"""Tests for the TaxAdvisor generator class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.generator import (
    DebugInfo,
    SourceInfo,
    TaxAdvisor,
    TaxAdvisorResponse,
)
from app.ingest import DocumentMetadata, EnrichedDocument
from app.query_engine import QueryAnalysis, QueryResult
from app.ranker import GradedDocument


class TestSourceInfo:
    """Tests for SourceInfo model."""

    def test_source_info_creation(self) -> None:
        """Test basic SourceInfo instantiation."""
        source = SourceInfo(
            section_id="11-121",
            summary="City collector duties.",
            relevance_score=9.0,
            text_preview="§ 11-121 The city collector shall...",
        )
        assert source.section_id == "11-121"
        assert source.relevance_score == 9.0
        assert "collector" in source.summary

    def test_source_info_defaults(self) -> None:
        """Test SourceInfo default values."""
        source = SourceInfo(section_id="11-121")
        assert source.summary == ""
        assert source.relevance_score == 10.0
        assert source.text_preview == ""


class TestDebugInfo:
    """Tests for DebugInfo model."""

    def test_debug_info_creation(self) -> None:
        """Test basic DebugInfo instantiation."""
        debug = DebugInfo(
            query_analysis={"is_relevant": True},
            hyde_query="Under NYC tax law...",
            documents_retrieved=5,
            documents_after_rerank=3,
            reranker_enabled=True,
            model_used="o3-mini",
            processing_time_ms=1500.0,
        )
        assert debug.documents_retrieved == 5
        assert debug.documents_after_rerank == 3
        assert debug.model_used == "o3-mini"

    def test_debug_info_defaults(self) -> None:
        """Test DebugInfo default values."""
        debug = DebugInfo()
        assert debug.query_analysis == {}
        assert debug.hyde_query == ""
        assert debug.documents_retrieved == 0
        assert debug.reranker_enabled is True


class TestTaxAdvisorResponse:
    """Tests for TaxAdvisorResponse model."""

    def test_response_creation(self) -> None:
        """Test basic TaxAdvisorResponse instantiation."""
        response = TaxAdvisorResponse(
            answer="Based on § 11-121, the taxpayer must...",
            sources=[
                SourceInfo(section_id="11-121", summary="Test"),
            ],
            warnings=["Jurisdiction: This is NYC-only"],
            debug_info=DebugInfo(model_used="o3-mini"),
        )
        assert "§ 11-121" in response.answer
        assert len(response.sources) == 1
        assert len(response.warnings) == 1

    def test_response_defaults(self) -> None:
        """Test TaxAdvisorResponse default values."""
        response = TaxAdvisorResponse(answer="Test answer")
        assert response.sources == []
        assert response.warnings == []
        assert response.debug_info.model_used == ""


class TestTaxAdvisorFormatting:
    """Tests for TaxAdvisor formatting methods."""

    @pytest.fixture
    def sample_graded_docs(self) -> list[GradedDocument]:
        """Create sample graded documents."""
        return [
            GradedDocument(
                document=EnrichedDocument(
                    id="doc-1",
                    section_id="11-121",
                    text="§ 11-121 City collector duties for tax collection.",
                    metadata=DocumentMetadata(
                        summary="City collector duties.",
                        tax_category="procedures",
                        document_type="requirement",
                    ),
                ),
                score=9.0,
                reasoning="Highly relevant",
            ),
            GradedDocument(
                document=EnrichedDocument(
                    id="doc-2",
                    section_id="11-122",
                    text="§ 11-122 Assessment procedures.",
                    metadata=DocumentMetadata(
                        summary="Assessment rules.",
                        tax_category="real_property",
                        document_type="procedure",
                    ),
                ),
                score=7.5,
                reasoning="Related topic",
            ),
        ]

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_format_context(
        self,
        mock_init: MagicMock,
        sample_graded_docs: list[GradedDocument],
    ) -> None:
        """Test context formatting for LLM."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        context = advisor._format_context(sample_graded_docs)

        assert "§ 11-121" in context
        assert "§ 11-122" in context
        assert "City collector duties" in context
        assert "9.0/10" in context
        assert "procedures" in context

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_create_sources(
        self,
        mock_init: MagicMock,
        sample_graded_docs: list[GradedDocument],
    ) -> None:
        """Test source info creation."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        sources = advisor._create_sources(sample_graded_docs)

        assert len(sources) == 2
        assert sources[0].section_id == "11-121"
        assert sources[0].relevance_score == 9.0
        assert sources[1].section_id == "11-122"


class TestTaxAdvisorWarnings:
    """Tests for TaxAdvisor warning collection."""

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_collect_warnings_federal(self, mock_init: MagicMock) -> None:
        """Test warning collection for federal queries."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        query_result = QueryResult(
            original_query="When is my IRS 1040 due?",
            expanded_query="Federal tax filing...",
            analysis=QueryAnalysis(
                is_relevant=True,
                relevance_score=0.8,
                jurisdiction="federal",
                jurisdiction_warning="This system covers NYC tax law only.",
            ),
        )

        warnings = advisor._collect_warnings(query_result)

        assert len(warnings) >= 1
        assert any("Federal" in w for w in warnings)
        assert any("NYC" in w for w in warnings)

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_collect_warnings_mixed(self, mock_init: MagicMock) -> None:
        """Test warning collection for mixed jurisdiction."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        query_result = QueryResult(
            original_query="How do NYC and federal taxes differ?",
            expanded_query="Tax comparison...",
            analysis=QueryAnalysis(
                is_relevant=True,
                relevance_score=0.7,
                jurisdiction="mixed",
            ),
        )

        warnings = advisor._collect_warnings(query_result)

        assert any("both Federal and NYC" in w for w in warnings)

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_collect_warnings_none(self, mock_init: MagicMock) -> None:
        """Test no warnings for clean NYC query."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        query_result = QueryResult(
            original_query="What are NYC property tax rates?",
            expanded_query="NYC property tax...",
            analysis=QueryAnalysis(
                is_relevant=True,
                relevance_score=0.9,
                jurisdiction="nyc",
            ),
        )

        warnings = advisor._collect_warnings(query_result)

        assert len(warnings) == 0


class TestTaxAdvisorSilenceResponse:
    """Tests for silence response when no relevant docs found."""

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_create_silence_response(self, mock_init: MagicMock) -> None:
        """Test silence response creation."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)
        advisor.SILENCE_RESPONSE = TaxAdvisor.SILENCE_RESPONSE
        advisor.reranker = MagicMock()
        advisor.reranker.enabled = True
        advisor.generation_model = "o3-mini"

        query_result = QueryResult(
            original_query="Test query",
            expanded_query="Expanded query",
            analysis=QueryAnalysis(is_relevant=True, relevance_score=0.5),
        )

        response = advisor._create_silence_response(
            query_result,
            documents_retrieved=5,
            start_time=0.0,
        )

        assert "could not find regulations" in response.answer
        assert "consult" in response.answer.lower()
        assert response.sources == []
        assert "No documents passed" in str(response.warnings)
        assert response.debug_info.documents_retrieved == 5
        assert response.debug_info.documents_after_rerank == 0


class TestTaxAdvisorPipeline:
    """Tests for the full TaxAdvisor pipeline."""

    @pytest.fixture
    def mock_vectorstore(self) -> MagicMock:
        """Create mock vectorstore."""
        return MagicMock()

    @pytest.fixture
    def sample_docstore(self, tmp_path: Path) -> Path:
        """Create sample docstore.json."""
        docs = [
            EnrichedDocument(
                id="doc-1",
                section_id="11-121",
                text="§ 11-121 City collector duties.",
                metadata=DocumentMetadata(summary="Collector duties."),
            ),
        ]
        docstore_path = tmp_path / "docstore.json"
        with open(docstore_path, "w") as f:
            json.dump([d.model_dump() for d in docs], f)
        return docstore_path

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_answer_question_normal_flow(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test normal answer generation flow."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        # Setup mocks
        advisor.processor = MagicMock()
        advisor.retriever = MagicMock()
        advisor.reranker = MagicMock()
        advisor.llm = MagicMock()
        advisor.fallback_llm = MagicMock()
        advisor.generation_model = "o3-mini"
        advisor.fallback_model = "gpt-5.2-pro"
        advisor.confidence_threshold = 8.0
        advisor.SYSTEM_PROMPT = TaxAdvisor.SYSTEM_PROMPT
        advisor.USER_PROMPT_TEMPLATE = TaxAdvisor.USER_PROMPT_TEMPLATE
        advisor.SILENCE_RESPONSE = TaxAdvisor.SILENCE_RESPONSE
        advisor.FALLBACK_SYSTEM_PROMPT = TaxAdvisor.FALLBACK_SYSTEM_PROMPT
        advisor.FALLBACK_DISCLAIMER = TaxAdvisor.FALLBACK_DISCLAIMER

        # Mock query processing
        query_result = QueryResult(
            original_query="What are property tax rates?",
            expanded_query="Under NYC law, property taxes...",
            analysis=QueryAnalysis(is_relevant=True, relevance_score=0.9),
        )
        advisor.processor.expand_query.return_value = query_result

        # Mock retrieval
        parent_doc = EnrichedDocument(
            id="doc-1",
            section_id="11-121",
            text="§ 11-121 Property tax rates are...",
            metadata=DocumentMetadata(summary="Tax rates."),
        )
        advisor.retriever.search.return_value = [parent_doc]

        # Mock re-ranking
        graded = GradedDocument(document=parent_doc, score=9.0, reasoning="Relevant")
        advisor.reranker.grade_documents.return_value = [graded]
        advisor.reranker.enabled = True

        # Mock LLM response
        llm_response = MagicMock()
        llm_response.content = "Based on § 11-121, property tax rates are..."
        advisor.llm.invoke.return_value = llm_response

        # Run pipeline
        response = advisor.answer_question("What are property tax rates?")

        assert isinstance(response, TaxAdvisorResponse)
        assert "§ 11-121" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].section_id == "11-121"
        assert response.debug_info.documents_retrieved == 1

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_answer_question_silence_on_no_docs(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test silence response when no documents pass re-ranking."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        # Setup mocks
        advisor.processor = MagicMock()
        advisor.retriever = MagicMock()
        advisor.reranker = MagicMock()
        advisor.generation_model = "o3-mini"
        advisor.SILENCE_RESPONSE = TaxAdvisor.SILENCE_RESPONSE

        query_result = QueryResult(
            original_query="How do I bake a cake?",
            expanded_query="Baking instructions...",
            analysis=QueryAnalysis(
                is_relevant=False,
                relevance_score=0.1,
                relevance_warning="Not a tax question.",
            ),
        )
        advisor.processor.expand_query.return_value = query_result

        # Mock retrieval returns docs
        advisor.retriever.search.return_value = [MagicMock()]

        # Mock re-ranking returns None (all filtered)
        advisor.reranker.grade_documents.return_value = None
        advisor.reranker.enabled = True

        response = advisor.answer_question("How do I bake a cake?")

        assert "could not find regulations" in response.answer
        assert response.debug_info.documents_after_rerank == 0

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_answer_question_empty_retrieval(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test silence response when retrieval returns empty."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)

        advisor.processor = MagicMock()
        advisor.retriever = MagicMock()
        advisor.reranker = MagicMock()
        advisor.generation_model = "o3-mini"
        advisor.SILENCE_RESPONSE = TaxAdvisor.SILENCE_RESPONSE

        query_result = QueryResult(
            original_query="Test",
            expanded_query="Test",
            analysis=QueryAnalysis(is_relevant=True, relevance_score=0.5),
        )
        advisor.processor.expand_query.return_value = query_result

        # Empty retrieval
        advisor.retriever.search.return_value = []
        advisor.reranker.enabled = True

        response = advisor.answer_question("Test query")

        assert "could not find regulations" in response.answer
        assert response.debug_info.documents_retrieved == 0


class TestTaxAdvisorConversation:
    """Tests for conversation history management."""

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_add_to_history(self, mock_init: MagicMock) -> None:
        """Test adding to conversation history."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)
        advisor.processor = MagicMock()

        advisor.add_to_history("Test query", "Test response")

        advisor.processor.add_to_history.assert_called_once_with(
            "Test query", "Test response"
        )

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_clear_history(self, mock_init: MagicMock) -> None:
        """Test clearing conversation history."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)
        advisor.processor = MagicMock()

        advisor.clear_history()

        advisor.processor.clear_history.assert_called_once()


class TestTaxAdvisorConfig:
    """Tests for TaxAdvisor configuration."""

    @patch("app.generator.TaxAdvisor.__init__", return_value=None)
    def test_get_config(self, mock_init: MagicMock) -> None:
        """Test getting configuration."""
        advisor = TaxAdvisor.__new__(TaxAdvisor)
        advisor.generation_model = "o3-mini"
        advisor.fallback_model = "gpt-5.2-pro"
        advisor.confidence_threshold = 8.0
        advisor.reranker = MagicMock()
        advisor.reranker.get_config.return_value = {
            "enabled": True,
            "threshold": 7.0,
        }
        advisor.retriever = MagicMock()
        advisor.retriever.parent_count = 100

        config = advisor.get_config()

        assert config["generation_model"] == "o3-mini"
        assert config["fallback_model"] == "gpt-5.2-pro"
        assert config["confidence_threshold"] == 8.0
        assert config["reranker"]["enabled"] is True
        assert config["parent_count"] == 100
