"""Tests for the DocumentEnricher class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ingest import (
    DocumentEnricher,
    DocumentMetadata,
    EnrichedDocument,
    ParentDocument,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_default_values(self) -> None:
        """Test that DocumentMetadata has sensible defaults."""
        metadata = DocumentMetadata()
        assert metadata.summary == ""
        assert metadata.keywords == []
        assert metadata.entities == []
        assert metadata.tax_category == "other"
        assert metadata.document_type == "other"
        assert metadata.hypothetical_questions == []
        assert metadata.cross_references == []
        assert metadata.applicable_parties == []

    def test_with_values(self) -> None:
        """Test DocumentMetadata with provided values."""
        metadata = DocumentMetadata(
            summary="Test summary",
            keywords=["tax", "assessment"],
            entities=["City Collector"],
            tax_category="real_property",
            document_type="procedure",
            hypothetical_questions=["How do I pay taxes?"],
            cross_references=["11-121"],
            applicable_parties=["taxpayers"],
        )
        assert metadata.summary == "Test summary"
        assert len(metadata.keywords) == 2
        assert "City Collector" in metadata.entities


class TestEnrichedDocument:
    """Tests for EnrichedDocument model."""

    def test_creation(self) -> None:
        """Test EnrichedDocument creation."""
        doc = EnrichedDocument(
            id="test-uuid",
            section_id="11-121",
            text="Test content",
            source="NYC Tax Law",
            metadata=DocumentMetadata(summary="Test summary"),
        )
        assert doc.id == "test-uuid"
        assert doc.section_id == "11-121"
        assert doc.metadata.summary == "Test summary"

    def test_default_metadata(self) -> None:
        """Test EnrichedDocument has default metadata."""
        doc = EnrichedDocument(
            id="test-uuid",
            section_id="11-121",
            text="Test content",
        )
        assert doc.metadata is not None
        assert doc.metadata.summary == ""


class TestDocumentEnricher:
    """Tests for DocumentEnricher class."""

    @pytest.fixture
    def sample_parent_doc(self) -> ParentDocument:
        """Create a sample parent document for testing."""
        return ParentDocument(
            id="test-uuid-123",
            section_id="11-121",
            text="§ 11-121 City collector; daily statements and accounts. "
            "The city collector shall enter upon accounts for each parcel "
            "of property, the payment of taxes, assessments, sewer rents.",
            source="NYC Tax Law",
        )

    @pytest.fixture
    def mock_llm_response(self) -> dict:
        """Sample LLM response for testing."""
        return {
            "summary": "Establishes city collector duties for daily tax accounting.",
            "keywords": ["taxes", "assessments", "sewer rents", "city collector"],
            "entities": ["City Collector"],
            "tax_category": "procedures",
            "document_type": "requirement",
            "hypothetical_questions": [
                "What are the city collector's daily duties?",
                "How are tax payments recorded?",
            ],
            "cross_references": [],
            "applicable_parties": ["city_officials", "collectors"],
        }

    def test_parse_llm_response_valid_json(self) -> None:
        """Test parsing valid JSON response."""
        enricher = MagicMock(spec=DocumentEnricher)
        enricher._parse_llm_response = DocumentEnricher._parse_llm_response

        response = '{"summary": "Test", "keywords": ["tax"]}'
        result = enricher._parse_llm_response(enricher, response)
        assert result["summary"] == "Test"
        assert result["keywords"] == ["tax"]

    def test_parse_llm_response_with_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        enricher = MagicMock(spec=DocumentEnricher)
        enricher._parse_llm_response = DocumentEnricher._parse_llm_response

        response = '```json\n{"summary": "Test"}\n```'
        result = enricher._parse_llm_response(enricher, response)
        assert result["summary"] == "Test"

    def test_parse_llm_response_invalid_json(self) -> None:
        """Test parsing invalid JSON returns empty dict."""
        enricher = MagicMock(spec=DocumentEnricher)
        enricher._parse_llm_response = DocumentEnricher._parse_llm_response

        response = "This is not JSON"
        result = enricher._parse_llm_response(enricher, response)
        assert result == {}

    def test_create_default_metadata(self) -> None:
        """Test default metadata creation."""
        enricher = MagicMock(spec=DocumentEnricher)
        enricher._create_default_metadata = (
            DocumentEnricher._create_default_metadata
        )

        metadata = enricher._create_default_metadata(enricher)
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.summary == ""
        assert metadata.keywords == []

    @patch("app.ingest.DocumentEnricher.__init__", return_value=None)
    def test_enrich_document_success(
        self,
        mock_init: MagicMock,
        sample_parent_doc: ParentDocument,
        mock_llm_response: dict,
    ) -> None:
        """Test successful document enrichment."""
        enricher = DocumentEnricher.__new__(DocumentEnricher)
        enricher.llm = MagicMock()
        enricher.model = "gpt-4o"

        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_response)
        enricher.llm.invoke.return_value = mock_response

        result = enricher.enrich_document(sample_parent_doc)

        assert isinstance(result, EnrichedDocument)
        assert result.id == sample_parent_doc.id
        assert result.section_id == sample_parent_doc.section_id
        assert result.metadata.summary == mock_llm_response["summary"]
        assert "taxes" in result.metadata.keywords

    @patch("app.ingest.DocumentEnricher.__init__", return_value=None)
    def test_enrich_document_llm_failure_uses_defaults(
        self,
        mock_init: MagicMock,
        sample_parent_doc: ParentDocument,
    ) -> None:
        """Test that LLM failure results in default metadata."""
        enricher = DocumentEnricher.__new__(DocumentEnricher)
        enricher.llm = MagicMock()
        enricher.model = "gpt-4o"

        enricher.llm.invoke.side_effect = Exception("API Error")

        result = enricher.enrich_document(sample_parent_doc, max_retries=1)

        assert isinstance(result, EnrichedDocument)
        assert result.metadata.summary == ""
        assert result.metadata.keywords == []

    @patch("app.ingest.DocumentEnricher.__init__", return_value=None)
    def test_enrich_batch_saves_to_file(
        self,
        mock_init: MagicMock,
        sample_parent_doc: ParentDocument,
        mock_llm_response: dict,
    ) -> None:
        """Test batch enrichment saves to file."""
        enricher = DocumentEnricher.__new__(DocumentEnricher)
        enricher.llm = MagicMock()
        enricher.model = "gpt-4o"

        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_response)
        enricher.llm.invoke.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "docstore.json"

            result = enricher.enrich_batch(
                [sample_parent_doc],
                output_path,
                checkpoint_interval=1,
            )

            assert len(result) == 1
            assert output_path.exists()

            with open(output_path, "r") as f:
                saved_data = json.load(f)
            assert len(saved_data) == 1
            assert saved_data[0]["section_id"] == "11-121"

    @patch("app.ingest.DocumentEnricher.__init__", return_value=None)
    def test_enrich_batch_resumes_from_checkpoint(
        self,
        mock_init: MagicMock,
        mock_llm_response: dict,
    ) -> None:
        """Test batch enrichment resumes from existing checkpoint."""
        enricher = DocumentEnricher.__new__(DocumentEnricher)
        enricher.llm = MagicMock()
        enricher.model = "gpt-4o"

        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_response)
        enricher.llm.invoke.return_value = mock_response

        doc1 = ParentDocument(
            id="uuid-1", section_id="11-121", text="Section 1"
        )
        doc2 = ParentDocument(
            id="uuid-2", section_id="11-122", text="Section 2"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "docstore.json"

            # Create existing checkpoint with doc1
            existing = [
                EnrichedDocument(
                    id="uuid-1",
                    section_id="11-121",
                    text="Section 1",
                    metadata=DocumentMetadata(summary="Existing"),
                ).model_dump()
            ]
            with open(output_path, "w") as f:
                json.dump(existing, f)

            # Process both docs - should only enrich doc2
            result = enricher.enrich_batch(
                [doc1, doc2], output_path, checkpoint_interval=1
            )

            # Should have 2 docs total
            assert len(result) == 2

            # LLM should only be called once (for doc2)
            assert enricher.llm.invoke.call_count == 1
