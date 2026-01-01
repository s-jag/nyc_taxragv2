"""Tests for the ReRanker class."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import DocumentMetadata, EnrichedDocument
from app.ranker import GradedDocument, ReRanker


class TestGradedDocument:
    """Tests for GradedDocument model."""

    def test_graded_document_creation(self) -> None:
        """Test basic GradedDocument instantiation."""
        doc = EnrichedDocument(
            id="test-id",
            section_id="11-121",
            text="Test content",
            metadata=DocumentMetadata(),
        )
        graded = GradedDocument(
            document=doc,
            score=8.5,
            reasoning="Highly relevant to the query",
        )
        assert graded.score == 8.5
        assert graded.document.section_id == "11-121"
        assert "relevant" in graded.reasoning

    def test_graded_document_score_bounds(self) -> None:
        """Test that score is bounded 0-10."""
        doc = EnrichedDocument(
            id="test-id",
            section_id="11-121",
            text="Test",
            metadata=DocumentMetadata(),
        )
        # Should accept valid scores
        graded = GradedDocument(document=doc, score=0.0, reasoning="")
        assert graded.score == 0.0

        graded = GradedDocument(document=doc, score=10.0, reasoning="")
        assert graded.score == 10.0

    def test_graded_document_default_reasoning(self) -> None:
        """Test default reasoning is empty string."""
        doc = EnrichedDocument(
            id="test-id",
            section_id="11-121",
            text="Test",
            metadata=DocumentMetadata(),
        )
        graded = GradedDocument(document=doc, score=5.0)
        assert graded.reasoning == ""


class TestReRankerParsing:
    """Tests for ReRanker response parsing."""

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_parse_valid_json(self, mock_init: MagicMock) -> None:
        """Test parsing valid JSON response."""
        ranker = ReRanker.__new__(ReRanker)

        response = json.dumps({"score": 8.5, "reasoning": "Highly relevant"})
        score, reasoning = ranker._parse_grading_response(response)

        assert score == 8.5
        assert reasoning == "Highly relevant"

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_parse_json_with_markdown(self, mock_init: MagicMock) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        ranker = ReRanker.__new__(ReRanker)

        response = f"```json\n{json.dumps({'score': 7.0, 'reasoning': 'Good match'})}\n```"
        score, reasoning = ranker._parse_grading_response(response)

        assert score == 7.0
        assert reasoning == "Good match"

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_parse_invalid_json_returns_default(self, mock_init: MagicMock) -> None:
        """Test that invalid JSON returns default score."""
        ranker = ReRanker.__new__(ReRanker)

        response = "This is not JSON"
        score, reasoning = ranker._parse_grading_response(response)

        assert score == 5.0
        assert "Unable to parse" in reasoning

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_parse_clamps_score_high(self, mock_init: MagicMock) -> None:
        """Test that scores > 10 are clamped to 10."""
        ranker = ReRanker.__new__(ReRanker)

        response = json.dumps({"score": 15, "reasoning": "Too high"})
        score, _ = ranker._parse_grading_response(response)

        assert score == 10.0

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_parse_clamps_score_low(self, mock_init: MagicMock) -> None:
        """Test that scores < 0 are clamped to 0."""
        ranker = ReRanker.__new__(ReRanker)

        response = json.dumps({"score": -5, "reasoning": "Too low"})
        score, _ = ranker._parse_grading_response(response)

        assert score == 0.0


class TestReRankerGrading:
    """Tests for ReRanker grading functionality."""

    @pytest.fixture
    def sample_documents(self) -> list[EnrichedDocument]:
        """Create sample documents for testing."""
        return [
            EnrichedDocument(
                id="doc-1",
                section_id="11-121",
                text="§ 11-121 Property tax assessment procedures.",
                metadata=DocumentMetadata(
                    summary="Property tax assessment rules.",
                    keywords=["property", "assessment"],
                ),
            ),
            EnrichedDocument(
                id="doc-2",
                section_id="11-122",
                text="§ 11-122 Business license requirements.",
                metadata=DocumentMetadata(
                    summary="Business licensing procedures.",
                    keywords=["business", "license"],
                ),
            ),
            EnrichedDocument(
                id="doc-3",
                section_id="11-123",
                text="§ 11-123 Penalty for late payment.",
                metadata=DocumentMetadata(
                    summary="Late payment penalties.",
                    keywords=["penalty", "late payment"],
                ),
            ),
        ]

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_single_success(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test grading a single document successfully."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"score": 9.0, "reasoning": "Directly relevant"}
        )
        ranker.llm.invoke.return_value = mock_response

        result = ranker._grade_single("property tax question", sample_documents[0])

        assert isinstance(result, GradedDocument)
        assert result.score == 9.0
        assert result.document.id == "doc-1"
        ranker.llm.invoke.assert_called_once()

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_single_llm_error_returns_default(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that LLM errors result in default score."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.llm.invoke.side_effect = Exception("API Error")

        result = ranker._grade_single("test query", sample_documents[0])

        assert result.score == 5.0
        assert "failed" in result.reasoning.lower()


class TestReRankerFiltering:
    """Tests for ReRanker filtering functionality."""

    @pytest.fixture
    def sample_documents(self) -> list[EnrichedDocument]:
        """Create sample documents for testing."""
        return [
            EnrichedDocument(
                id=f"doc-{i}",
                section_id=f"11-12{i}",
                text=f"§ 11-12{i} Test content {i}",
                metadata=DocumentMetadata(summary=f"Summary {i}"),
            )
            for i in range(3)
        ]

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_documents_filters_by_threshold(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that documents below threshold are filtered out."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.threshold = 7.0
        ranker.enabled = True

        # Mock responses with different scores
        responses = [
            MagicMock(content=json.dumps({"score": 9.0, "reasoning": "High"})),
            MagicMock(content=json.dumps({"score": 5.0, "reasoning": "Low"})),
            MagicMock(content=json.dumps({"score": 8.0, "reasoning": "High"})),
        ]
        ranker.llm.invoke.side_effect = responses

        result = ranker.grade_documents("test query", sample_documents)

        assert result is not None
        assert len(result) == 2  # Only docs with score >= 7
        assert all(g.score >= 7.0 for g in result)

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_documents_returns_none_when_all_filtered(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that None is returned when all docs are filtered out."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.threshold = 7.0
        ranker.enabled = True

        # All scores below threshold
        responses = [
            MagicMock(content=json.dumps({"score": 3.0, "reasoning": "Low"})),
            MagicMock(content=json.dumps({"score": 2.0, "reasoning": "Low"})),
            MagicMock(content=json.dumps({"score": 4.0, "reasoning": "Low"})),
        ]
        ranker.llm.invoke.side_effect = responses

        result = ranker.grade_documents("test query", sample_documents)

        assert result is None  # Retrieval failure signal

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_documents_sorted_by_score(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that results are sorted by score descending."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.threshold = 7.0
        ranker.enabled = True

        responses = [
            MagicMock(content=json.dumps({"score": 7.5, "reasoning": "OK"})),
            MagicMock(content=json.dumps({"score": 9.0, "reasoning": "Best"})),
            MagicMock(content=json.dumps({"score": 8.0, "reasoning": "Good"})),
        ]
        ranker.llm.invoke.side_effect = responses

        result = ranker.grade_documents("test query", sample_documents)

        assert result is not None
        assert result[0].score == 9.0  # Highest first
        assert result[1].score == 8.0
        assert result[2].score == 7.5

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_grade_documents_empty_input(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test that empty input returns empty list."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.enabled = True

        result = ranker.grade_documents("test query", [])

        assert result == []


class TestReRankerToggle:
    """Tests for ReRanker enable/disable toggle."""

    @pytest.fixture
    def sample_documents(self) -> list[EnrichedDocument]:
        """Create sample documents for testing."""
        return [
            EnrichedDocument(
                id=f"doc-{i}",
                section_id=f"11-12{i}",
                text=f"Test {i}",
                metadata=DocumentMetadata(),
            )
            for i in range(2)
        ]

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_disabled_passes_all_documents(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that disabled re-ranker passes all documents through."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.enabled = False
        ranker.threshold = 7.0

        result = ranker.grade_documents("test query", sample_documents)

        assert result is not None
        assert len(result) == 2
        assert all(g.score == 10.0 for g in result)  # Max score when disabled
        assert all("disabled" in g.reasoning.lower() for g in result)

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_disabled_does_not_call_llm(
        self,
        mock_init: MagicMock,
        sample_documents: list[EnrichedDocument],
    ) -> None:
        """Test that disabled re-ranker doesn't call LLM."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.enabled = False
        ranker.threshold = 7.0

        ranker.grade_documents("test query", sample_documents)

        ranker.llm.invoke.assert_not_called()


class TestReRankerConfig:
    """Tests for ReRanker configuration."""

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_get_config(self, mock_init: MagicMock) -> None:
        """Test get_config returns current settings."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.model = "gpt-4o"
        ranker.threshold = 7.0
        ranker.enabled = True

        config = ranker.get_config()

        assert config["model"] == "gpt-4o"
        assert config["threshold"] == 7.0
        assert config["enabled"] is True

    @patch("app.ranker.ReRanker.__init__", return_value=None)
    def test_custom_threshold(self, mock_init: MagicMock) -> None:
        """Test that custom threshold is respected."""
        ranker = ReRanker.__new__(ReRanker)
        ranker.llm = MagicMock()
        ranker.threshold = 5.0  # Lower threshold
        ranker.enabled = True

        doc = EnrichedDocument(
            id="doc-1",
            section_id="11-121",
            text="Test",
            metadata=DocumentMetadata(),
        )

        # Score of 6 should pass with threshold of 5
        response = MagicMock(content=json.dumps({"score": 6.0, "reasoning": "OK"}))
        ranker.llm.invoke.return_value = response

        result = ranker.grade_documents("test", [doc])

        assert result is not None
        assert len(result) == 1
