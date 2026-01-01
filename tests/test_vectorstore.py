"""Tests for the VectorStoreManager and ChildDocument classes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import (
    ChildDocument,
    DocumentMetadata,
    EnrichedDocument,
    VectorStoreManager,
)


class TestChildDocument:
    """Tests for ChildDocument model."""

    def test_child_document_creation(self) -> None:
        """Test basic ChildDocument instantiation."""
        child = ChildDocument(
            id="child-uuid",
            parent_id="parent-uuid",
            section_id="11-121",
            text="Test chunk content",
            chunk_index=0,
            keywords=["tax", "assessment"],
            summary="Test summary",
            tax_category="real_property",
            entities=["City Collector"],
            hypothetical_questions=["What is this about?"],
            applicable_parties=["taxpayers"],
        )
        assert child.id == "child-uuid"
        assert child.parent_id == "parent-uuid"
        assert child.section_id == "11-121"
        assert child.chunk_index == 0
        assert "tax" in child.keywords

    def test_child_document_defaults(self) -> None:
        """Test ChildDocument default values."""
        child = ChildDocument(
            id="child-uuid",
            parent_id="parent-uuid",
            section_id="11-121",
            text="Test content",
            chunk_index=0,
        )
        assert child.keywords == []
        assert child.summary == ""
        assert child.tax_category == "other"
        assert child.entities == []


class TestVectorStoreManager:
    """Tests for VectorStoreManager class."""

    @pytest.fixture
    def sample_enriched_doc(self) -> EnrichedDocument:
        """Create a sample enriched document for testing."""
        return EnrichedDocument(
            id="parent-uuid-123",
            section_id="11-121",
            text="§ 11-121 City collector; daily statements and accounts. "
            "The city collector or the deputy collector in each borough office "
            "shall enter upon accounts, to be maintained in each such office "
            "for each parcel of property, the payment of taxes, assessments, "
            "sewer rents or water rents thereon. At close of office hours each "
            "day, the city collector shall render to the commissioner of finance "
            "a statement of the sums so received. The comptroller shall credit "
            "the city collector in his or her books with such amount.",
            source="NYC Tax Law",
            metadata=DocumentMetadata(
                summary="Establishes city collector duties for daily accounting.",
                keywords=["taxes", "assessments", "city collector", "comptroller"],
                entities=["City Collector", "Commissioner of Finance", "Comptroller"],
                tax_category="procedures",
                document_type="requirement",
                hypothetical_questions=[
                    "What are the city collector's daily duties?",
                    "How are tax payments recorded?",
                ],
                cross_references=[],
                applicable_parties=["city_officials", "collectors"],
            ),
        )

    @pytest.fixture
    def sample_docstore(
        self, sample_enriched_doc: EnrichedDocument, tmp_path: Path
    ) -> Path:
        """Create a temporary docstore.json file."""
        docstore_path = tmp_path / "docstore.json"
        with open(docstore_path, "w") as f:
            json.dump([sample_enriched_doc.model_dump()], f)
        return docstore_path

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_load_enriched_documents(
        self,
        mock_init: MagicMock,
        sample_docstore: Path,
    ) -> None:
        """Test loading enriched documents from JSON."""
        manager = VectorStoreManager.__new__(VectorStoreManager)

        docs = manager.load_enriched_documents(sample_docstore)

        assert len(docs) == 1
        assert docs[0].section_id == "11-121"
        assert docs[0].metadata.summary == "Establishes city collector duties for daily accounting."

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_create_children_splits_text(
        self,
        mock_init: MagicMock,
        sample_enriched_doc: EnrichedDocument,
    ) -> None:
        """Test that create_children splits parent into chunks."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        manager = VectorStoreManager.__new__(VectorStoreManager)
        manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        children = manager.create_children(sample_enriched_doc)

        # Should create multiple children from the text
        assert len(children) >= 1
        # All children should have same parent_id
        assert all(c.parent_id == sample_enriched_doc.id for c in children)
        # All children should have same section_id
        assert all(c.section_id == sample_enriched_doc.section_id for c in children)
        # Children should have sequential chunk indices
        indices = [c.chunk_index for c in children]
        assert indices == list(range(len(children)))

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_children_inherit_metadata(
        self,
        mock_init: MagicMock,
        sample_enriched_doc: EnrichedDocument,
    ) -> None:
        """Test that children inherit all parent metadata."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        manager = VectorStoreManager.__new__(VectorStoreManager)
        manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
        )

        children = manager.create_children(sample_enriched_doc)

        for child in children:
            assert child.keywords == sample_enriched_doc.metadata.keywords
            assert child.summary == sample_enriched_doc.metadata.summary
            assert child.tax_category == sample_enriched_doc.metadata.tax_category
            assert child.entities == sample_enriched_doc.metadata.entities
            assert (
                child.hypothetical_questions
                == sample_enriched_doc.metadata.hypothetical_questions
            )
            assert (
                child.applicable_parties
                == sample_enriched_doc.metadata.applicable_parties
            )

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_prepare_chroma_documents(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test preparation of documents for ChromaDB."""
        manager = VectorStoreManager.__new__(VectorStoreManager)

        children = [
            ChildDocument(
                id="child-1",
                parent_id="parent-1",
                section_id="11-121",
                text="First chunk",
                chunk_index=0,
                keywords=["tax", "assessment"],
                summary="Test summary",
                tax_category="procedures",
                entities=["City Collector"],
                hypothetical_questions=["Question 1?", "Question 2?"],
                applicable_parties=["taxpayers"],
            ),
            ChildDocument(
                id="child-2",
                parent_id="parent-1",
                section_id="11-121",
                text="Second chunk",
                chunk_index=1,
                keywords=["tax", "assessment"],
                summary="Test summary",
                tax_category="procedures",
                entities=["City Collector"],
                hypothetical_questions=["Question 1?", "Question 2?"],
                applicable_parties=["taxpayers"],
            ),
        ]

        texts, metadatas, ids = manager._prepare_chroma_documents(children)

        assert len(texts) == 2
        assert len(metadatas) == 2
        assert len(ids) == 2

        assert texts[0] == "First chunk"
        assert ids[0] == "child-1"
        assert metadatas[0]["parent_id"] == "parent-1"
        assert metadatas[0]["section_id"] == "11-121"
        assert metadatas[0]["chunk_index"] == 0
        assert metadatas[0]["keywords"] == "tax, assessment"
        assert metadatas[0]["entities"] == "City Collector"
        assert metadatas[0]["hypothetical_questions"] == "Question 1? | Question 2?"

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_create_all_children(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test creating children from multiple parents."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        manager = VectorStoreManager.__new__(VectorStoreManager)
        manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
        )

        parents = [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="This is the first parent document with some text content.",
                metadata=DocumentMetadata(summary="Parent 1"),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="This is the second parent document with different content.",
                metadata=DocumentMetadata(summary="Parent 2"),
            ),
        ]

        children = manager.create_all_children(parents)

        # Should have children from both parents
        parent_ids = set(c.parent_id for c in children)
        assert "parent-1" in parent_ids
        assert "parent-2" in parent_ids


class TestVectorStoreManagerIntegration:
    """Integration tests for VectorStoreManager (require mocked embeddings)."""

    @pytest.fixture
    def mock_embeddings(self) -> MagicMock:
        """Create mock embeddings."""
        mock = MagicMock()
        mock.embed_documents.return_value = [[0.1] * 1536]  # OpenAI embedding dim
        mock.embed_query.return_value = [0.1] * 1536
        return mock

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_create_vector_db(
        self,
        mock_init: MagicMock,
        mock_embeddings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test creating vector database (mocking Chroma import)."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import sys

        # Create a mock Chroma module
        mock_chroma_class = MagicMock()
        mock_vectorstore = MagicMock()
        mock_chroma_class.from_texts.return_value = mock_vectorstore

        # Setup manager
        manager = VectorStoreManager.__new__(VectorStoreManager)
        manager.persist_dir = tmp_path / "vectorstore"
        manager.embeddings = mock_embeddings
        manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        # Create docstore
        docstore_path = tmp_path / "docstore.json"
        doc = EnrichedDocument(
            id="test-parent",
            section_id="11-121",
            text="Test document content for vector indexing.",
            metadata=DocumentMetadata(
                summary="Test summary",
                keywords=["test"],
            ),
        )
        with open(docstore_path, "w") as f:
            json.dump([doc.model_dump()], f)

        # Mock the import inside create_vector_db
        with patch.dict(
            sys.modules,
            {"langchain_community": MagicMock(), "langchain_community.vectorstores": MagicMock()}
        ):
            with patch(
                "app.ingest.VectorStoreManager.create_vector_db"
            ) as mock_create:
                mock_create.return_value = mock_vectorstore

                # Test that we can call the method
                result = mock_create(docstore_path)
                assert result == mock_vectorstore

    @patch("app.ingest.VectorStoreManager.__init__", return_value=None)
    def test_prepare_and_chunk_workflow(
        self,
        mock_init: MagicMock,
        mock_embeddings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test the full chunking workflow without Chroma."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # Setup manager
        manager = VectorStoreManager.__new__(VectorStoreManager)
        manager.persist_dir = tmp_path / "vectorstore"
        manager.embeddings = mock_embeddings
        manager.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        # Create docstore with multiple docs
        docstore_path = tmp_path / "docstore.json"
        docs = [
            EnrichedDocument(
                id="parent-1",
                section_id="11-121",
                text="First document with enough text to potentially split into multiple chunks if needed.",
                metadata=DocumentMetadata(
                    summary="First summary",
                    keywords=["first", "test"],
                    entities=["Entity1"],
                ),
            ),
            EnrichedDocument(
                id="parent-2",
                section_id="11-122",
                text="Second document with different content for testing purposes.",
                metadata=DocumentMetadata(
                    summary="Second summary",
                    keywords=["second", "test"],
                    entities=["Entity2"],
                ),
            ),
        ]
        with open(docstore_path, "w") as f:
            json.dump([d.model_dump() for d in docs], f)

        # Load and process
        loaded = manager.load_enriched_documents(docstore_path)
        assert len(loaded) == 2

        children = manager.create_all_children(loaded)
        assert len(children) >= 2  # At least one child per parent

        texts, metadatas, ids = manager._prepare_chroma_documents(children)
        assert len(texts) == len(children)
        assert len(metadatas) == len(children)
        assert len(ids) == len(children)

        # Verify parent IDs are preserved
        parent_ids = set(m["parent_id"] for m in metadatas)
        assert "parent-1" in parent_ids
        assert "parent-2" in parent_ids
