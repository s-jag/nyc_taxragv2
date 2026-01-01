"""Tests for the Retriever class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import DocumentMetadata, EnrichedDocument
from app.query_engine import QueryAnalysis, QueryResult
from app.retriever import Retriever


class TestRetrieverLoading:
    """Tests for Retriever initialization and docstore loading."""

    @pytest.fixture
    def sample_parents(self) -> list[EnrichedDocument]:
        """Create sample parent documents."""
        return [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="§ 11-121 City collector duties for tax collection.",
                metadata=DocumentMetadata(
                    summary="City collector duties.",
                    keywords=["collector", "taxes"],
                ),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="§ 11-122 Assessment procedures for property.",
                metadata=DocumentMetadata(
                    summary="Assessment procedures.",
                    keywords=["assessment", "property"],
                ),
            ),
            EnrichedDocument(
                id="parent-3",
                section_id="11-123",
                text="§ 11-123 Penalties for late payment.",
                metadata=DocumentMetadata(
                    summary="Late payment penalties.",
                    keywords=["penalties", "late payment"],
                ),
            ),
        ]

    @pytest.fixture
    def sample_docstore(
        self, sample_parents: list[EnrichedDocument], tmp_path: Path
    ) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        with open(docstore_path, "w") as f:
            json.dump([p.model_dump() for p in sample_parents], f)
        return docstore_path

    @pytest.fixture
    def mock_vectorstore(self) -> MagicMock:
        """Create a mock vector store."""
        return MagicMock()

    def test_load_docstore(
        self,
        mock_vectorstore: MagicMock,
        sample_docstore: Path,
    ) -> None:
        """Test loading docstore creates parent lookup dict."""
        retriever = Retriever(mock_vectorstore, sample_docstore)

        assert len(retriever.parents) == 3
        assert "parent-1" in retriever.parents
        assert "parent-2" in retriever.parents
        assert "parent-3" in retriever.parents

    def test_load_docstore_with_correct_data(
        self,
        mock_vectorstore: MagicMock,
        sample_docstore: Path,
    ) -> None:
        """Test that loaded parents have correct data."""
        retriever = Retriever(mock_vectorstore, sample_docstore)

        parent = retriever.parents["parent-1"]
        assert parent.section_id == "11-121"
        assert "collector" in parent.text
        assert parent.metadata.summary == "City collector duties."

    def test_parent_count_property(
        self,
        mock_vectorstore: MagicMock,
        sample_docstore: Path,
    ) -> None:
        """Test parent_count property."""
        retriever = Retriever(mock_vectorstore, sample_docstore)
        assert retriever.parent_count == 3


class TestRetrieverSearch:
    """Tests for Retriever search methods."""

    @pytest.fixture
    def sample_parents(self) -> list[EnrichedDocument]:
        """Create sample parent documents."""
        return [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="§ 11-121 City collector duties.",
                metadata=DocumentMetadata(summary="Collector duties."),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="§ 11-122 Assessment procedures.",
                metadata=DocumentMetadata(summary="Assessment procedures."),
            ),
        ]

    @pytest.fixture
    def sample_docstore(
        self, sample_parents: list[EnrichedDocument], tmp_path: Path
    ) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        with open(docstore_path, "w") as f:
            json.dump([p.model_dump() for p in sample_parents], f)
        return docstore_path

    @pytest.fixture
    def mock_vectorstore(self) -> MagicMock:
        """Create a mock vector store with search results."""
        mock = MagicMock()

        # Create mock child documents
        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1", "section_id": "11-121"}

        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-2", "section_id": "11-122"}

        mock.similarity_search.return_value = [child1, child2]
        return mock

    @pytest.fixture
    def sample_query_result(self) -> QueryResult:
        """Create a sample query result."""
        return QueryResult(
            original_query="What are collector duties?",
            expanded_query="Under NYC tax law, the city collector is responsible for...",
            analysis=QueryAnalysis(is_relevant=True, relevance_score=0.9),
        )

    def test_search_returns_parents(
        self,
        mock_vectorstore: MagicMock,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test search returns parent documents from child matches."""
        retriever = Retriever(mock_vectorstore, sample_docstore)

        results = retriever.search(sample_query_result, k=5)

        assert len(results) == 2
        assert results[0].id == "parent-1"
        assert results[1].id == "parent-2"
        mock_vectorstore.similarity_search.assert_called_once_with(
            sample_query_result.expanded_query,
            k=5,
        )

    def test_search_uses_hyde_expanded_query(
        self,
        mock_vectorstore: MagicMock,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test that search uses the HyDE-expanded query."""
        retriever = Retriever(mock_vectorstore, sample_docstore)

        retriever.search(sample_query_result)

        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[0][0] == sample_query_result.expanded_query

    def test_search_deduplicates_parents(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test that multiple children from same parent return one parent."""
        mock_vs = MagicMock()

        # Two children from the same parent
        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1", "section_id": "11-121"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-1", "section_id": "11-121"}
        child3 = MagicMock()
        child3.metadata = {"parent_id": "parent-2", "section_id": "11-122"}

        mock_vs.similarity_search.return_value = [child1, child2, child3]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search(sample_query_result)

        assert len(results) == 2  # Only 2 unique parents
        parent_ids = [r.id for r in results]
        assert parent_ids.count("parent-1") == 1

    def test_search_preserves_order(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test that search preserves order of first occurrence."""
        mock_vs = MagicMock()

        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-2"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-1"}

        mock_vs.similarity_search.return_value = [child1, child2]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search(sample_query_result)

        assert results[0].id == "parent-2"  # First seen
        assert results[1].id == "parent-1"  # Second seen

    def test_search_handles_missing_parent(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test that search skips children with missing parents."""
        mock_vs = MagicMock()

        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "non-existent-parent"}

        mock_vs.similarity_search.return_value = [child1, child2]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search(sample_query_result)

        assert len(results) == 1
        assert results[0].id == "parent-1"


class TestRetrieverGetParent:
    """Tests for direct parent lookup."""

    @pytest.fixture
    def sample_docstore(self, tmp_path: Path) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        docs = [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="Test content",
                metadata=DocumentMetadata(),
            ),
        ]
        with open(docstore_path, "w") as f:
            json.dump([d.model_dump() for d in docs], f)
        return docstore_path

    def test_get_parent_found(
        self,
        sample_docstore: Path,
    ) -> None:
        """Test get_parent returns document when found."""
        retriever = Retriever(MagicMock(), sample_docstore)

        result = retriever.get_parent("parent-1")

        assert result is not None
        assert result.id == "parent-1"

    def test_get_parent_not_found(
        self,
        sample_docstore: Path,
    ) -> None:
        """Test get_parent returns None when not found."""
        retriever = Retriever(MagicMock(), sample_docstore)

        result = retriever.get_parent("non-existent")

        assert result is None


class TestRetrieverSearchBySection:
    """Tests for section-based search."""

    @pytest.fixture
    def sample_docstore(self, tmp_path: Path) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        docs = [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="§ 11-121 content",
                metadata=DocumentMetadata(),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="§ 11-122 content",
                metadata=DocumentMetadata(),
            ),
        ]
        with open(docstore_path, "w") as f:
            json.dump([d.model_dump() for d in docs], f)
        return docstore_path

    def test_search_by_section(
        self,
        sample_docstore: Path,
    ) -> None:
        """Test search_by_section uses metadata filter."""
        mock_vs = MagicMock()
        child = MagicMock()
        child.metadata = {"parent_id": "parent-1", "section_id": "11-121"}
        mock_vs.similarity_search.return_value = [child]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search_by_section("11-121")

        assert len(results) == 1
        assert results[0].section_id == "11-121"

        mock_vs.similarity_search.assert_called_once_with(
            "",
            k=5,
            filter={"section_id": "11-121"},
        )


class TestRetrieverSearchWithScores:
    """Tests for search with similarity scores."""

    @pytest.fixture
    def sample_docstore(self, tmp_path: Path) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        docs = [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="content 1",
                metadata=DocumentMetadata(),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="content 2",
                metadata=DocumentMetadata(),
            ),
        ]
        with open(docstore_path, "w") as f:
            json.dump([d.model_dump() for d in docs], f)
        return docstore_path

    @pytest.fixture
    def sample_query_result(self) -> QueryResult:
        """Create a sample query result."""
        return QueryResult(
            original_query="test",
            expanded_query="expanded test",
            analysis=QueryAnalysis(is_relevant=True, relevance_score=0.9),
        )

    def test_search_with_scores_returns_tuples(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test search_with_scores returns (document, score) tuples."""
        mock_vs = MagicMock()

        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-2"}

        mock_vs.similarity_search_with_score.return_value = [
            (child1, 0.1),
            (child2, 0.2),
        ]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search_with_scores(sample_query_result)

        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert results[0][0].id == "parent-1"
        assert results[0][1] == 0.1

    def test_search_with_scores_sorted_by_score(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test results are sorted by score (lowest first)."""
        mock_vs = MagicMock()

        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-2"}

        # Return in reverse order
        mock_vs.similarity_search_with_score.return_value = [
            (child2, 0.5),
            (child1, 0.1),
        ]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search_with_scores(sample_query_result)

        # Should be sorted by score
        assert results[0][0].id == "parent-1"  # score 0.1
        assert results[1][0].id == "parent-2"  # score 0.5

    def test_search_with_scores_keeps_best_score(
        self,
        sample_docstore: Path,
        sample_query_result: QueryResult,
    ) -> None:
        """Test that best score is kept when multiple children match same parent."""
        mock_vs = MagicMock()

        child1 = MagicMock()
        child1.metadata = {"parent_id": "parent-1"}
        child2 = MagicMock()
        child2.metadata = {"parent_id": "parent-1"}

        mock_vs.similarity_search_with_score.return_value = [
            (child1, 0.3),
            (child2, 0.1),  # Better score
        ]

        retriever = Retriever(mock_vs, sample_docstore)
        results = retriever.search_with_scores(sample_query_result)

        assert len(results) == 1
        assert results[0][1] == 0.1  # Best score kept
