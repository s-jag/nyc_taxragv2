"""Tests for the QueryProcessor and HyDE implementation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.query_engine import QueryAnalysis, QueryProcessor, QueryResult


class TestQueryAnalysis:
    """Tests for QueryAnalysis model."""

    def test_query_analysis_creation(self) -> None:
        """Test basic QueryAnalysis instantiation."""
        analysis = QueryAnalysis(
            is_relevant=True,
            relevance_score=0.9,
            relevance_warning=None,
            jurisdiction="nyc",
            jurisdiction_warning=None,
            detected_intent="asking about property tax",
        )
        assert analysis.is_relevant is True
        assert analysis.relevance_score == 0.9
        assert analysis.jurisdiction == "nyc"

    def test_query_analysis_with_warnings(self) -> None:
        """Test QueryAnalysis with warning messages."""
        analysis = QueryAnalysis(
            is_relevant=False,
            relevance_score=0.2,
            relevance_warning="This appears to be a cooking question",
            jurisdiction="unknown",
            jurisdiction_warning=None,
            detected_intent="baking instructions",
        )
        assert analysis.is_relevant is False
        assert "cooking" in analysis.relevance_warning

    def test_query_analysis_jurisdiction_warning(self) -> None:
        """Test QueryAnalysis with jurisdiction warning."""
        analysis = QueryAnalysis(
            is_relevant=True,
            relevance_score=0.8,
            relevance_warning=None,
            jurisdiction="federal",
            jurisdiction_warning="This system covers NYC tax law only",
            detected_intent="IRS filing requirements",
        )
        assert analysis.jurisdiction == "federal"
        assert analysis.jurisdiction_warning is not None

    def test_query_analysis_defaults(self) -> None:
        """Test QueryAnalysis default values."""
        analysis = QueryAnalysis(
            is_relevant=True,
            relevance_score=0.5,
        )
        assert analysis.jurisdiction == "unknown"
        assert analysis.relevance_warning is None
        assert analysis.jurisdiction_warning is None
        assert analysis.detected_intent == ""


class TestQueryResult:
    """Tests for QueryResult model."""

    def test_query_result_creation(self) -> None:
        """Test basic QueryResult instantiation."""
        analysis = QueryAnalysis(
            is_relevant=True,
            relevance_score=0.9,
        )
        result = QueryResult(
            original_query="Do I pay taxes on rental income?",
            expanded_query="Under NYC tax law, rental income is subject to...",
            analysis=analysis,
            conversation_context="",
        )
        assert result.original_query == "Do I pay taxes on rental income?"
        assert "rental income" in result.expanded_query
        assert result.analysis.is_relevant is True

    def test_query_result_with_context(self) -> None:
        """Test QueryResult with conversation context."""
        analysis = QueryAnalysis(is_relevant=True, relevance_score=0.8)
        result = QueryResult(
            original_query="What about businesses?",
            expanded_query="For business entities in NYC...",
            analysis=analysis,
            conversation_context="User: What are property taxes?\nAssistant: Property taxes in NYC...",
        )
        assert "property taxes" in result.conversation_context


class TestQueryProcessor:
    """Tests for QueryProcessor class."""

    @pytest.fixture
    def mock_llm_analysis_response(self) -> dict:
        """Sample LLM analysis response."""
        return {
            "is_relevant": True,
            "relevance_score": 0.9,
            "relevance_warning": None,
            "jurisdiction": "nyc",
            "jurisdiction_warning": None,
            "detected_intent": "asking about property tax rates",
        }

    @pytest.fixture
    def mock_hyde_response(self) -> str:
        """Sample HyDE hypothetical answer."""
        return """Under NYC tax regulations, property taxes are assessed by the
Department of Finance pursuant to Section 11-201 of the NYC Administrative Code.
The Commissioner of Finance is responsible for determining the assessed value
of real property within the city. Property owners are required to pay taxes
based on the assessed value multiplied by the applicable tax rate for their
property class."""

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_parse_analysis_response_valid_json(
        self,
        mock_init: MagicMock,
        mock_llm_analysis_response: dict,
    ) -> None:
        """Test parsing valid JSON analysis response."""
        processor = QueryProcessor.__new__(QueryProcessor)

        response_text = json.dumps(mock_llm_analysis_response)
        analysis = processor._parse_analysis_response(response_text)

        assert analysis.is_relevant is True
        assert analysis.relevance_score == 0.9
        assert analysis.jurisdiction == "nyc"
        assert analysis.detected_intent == "asking about property tax rates"

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_parse_analysis_response_with_markdown(
        self,
        mock_init: MagicMock,
        mock_llm_analysis_response: dict,
    ) -> None:
        """Test parsing JSON wrapped in markdown code blocks."""
        processor = QueryProcessor.__new__(QueryProcessor)

        response_text = f"```json\n{json.dumps(mock_llm_analysis_response)}\n```"
        analysis = processor._parse_analysis_response(response_text)

        assert analysis.is_relevant is True
        assert analysis.jurisdiction == "nyc"

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_parse_analysis_response_invalid_json(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test parsing invalid JSON returns permissive defaults."""
        processor = QueryProcessor.__new__(QueryProcessor)

        response_text = "This is not JSON"
        analysis = processor._parse_analysis_response(response_text)

        # Should return permissive defaults
        assert analysis.is_relevant is True
        assert analysis.relevance_score == 0.5
        assert analysis.jurisdiction == "unknown"

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_get_context_section_empty(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test context section with no history."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []

        context = processor._get_context_section()
        assert context == ""

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_get_context_section_with_history(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test context section with conversation history."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.MAX_HISTORY = 5
        processor.conversation_history = [
            {"role": "user", "content": "What are property taxes?"},
            {"role": "assistant", "content": "Property taxes in NYC are..."},
        ]

        context = processor._get_context_section()
        assert "User: What are property taxes?" in context
        assert "Assistant: Property taxes in NYC" in context

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_add_to_history(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test adding to conversation history."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.MAX_HISTORY = 5
        processor.conversation_history = []

        processor.add_to_history("What is sales tax?", "Sales tax in NYC is...")

        assert len(processor.conversation_history) == 2
        assert processor.conversation_history[0]["role"] == "user"
        assert processor.conversation_history[1]["role"] == "assistant"

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_add_to_history_trims_old(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test that history is trimmed when exceeding max."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.MAX_HISTORY = 2
        processor.conversation_history = []

        # Add 3 exchanges (6 messages, exceeds MAX_HISTORY * 2 = 4)
        processor.add_to_history("Q1", "A1")
        processor.add_to_history("Q2", "A2")
        processor.add_to_history("Q3", "A3")

        # Should only keep last MAX_HISTORY * 2 = 4 messages
        assert len(processor.conversation_history) == 4
        assert processor.conversation_history[0]["content"] == "Q2"

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_clear_history(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test clearing conversation history."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        processor.clear_history()
        assert processor.conversation_history == []

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_get_history(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test getting history returns a copy."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = [
            {"role": "user", "content": "Test"},
        ]

        history = processor.get_history()
        history.append({"role": "test", "content": "modified"})

        # Original should be unchanged
        assert len(processor.conversation_history) == 1

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_analyze_query_success(
        self,
        mock_init: MagicMock,
        mock_llm_analysis_response: dict,
    ) -> None:
        """Test successful query analysis."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_llm_analysis_response)
        processor.llm.invoke.return_value = mock_response

        analysis = processor.analyze_query("What are NYC property tax rates?")

        assert analysis.is_relevant is True
        assert analysis.jurisdiction == "nyc"
        processor.llm.invoke.assert_called_once()

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_analyze_query_error_returns_defaults(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test query analysis error returns permissive defaults."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()
        processor.llm.invoke.side_effect = Exception("API Error")

        analysis = processor.analyze_query("Test query")

        assert analysis.is_relevant is True  # Permissive default
        assert analysis.relevance_score == 0.5

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_generate_hypothetical_answer_success(
        self,
        mock_init: MagicMock,
        mock_hyde_response: str,
    ) -> None:
        """Test successful HyDE generation."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = mock_hyde_response
        processor.llm.invoke.return_value = mock_response

        result = processor.generate_hypothetical_answer("What are property taxes?")

        assert "Department of Finance" in result
        assert "Section 11-201" in result
        processor.llm.invoke.assert_called_once()

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_generate_hypothetical_answer_error_returns_query(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test HyDE error falls back to original query."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()
        processor.llm.invoke.side_effect = Exception("API Error")

        original_query = "What are property taxes?"
        result = processor.generate_hypothetical_answer(original_query)

        assert result == original_query  # Falls back to original

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_expand_query_full_flow(
        self,
        mock_init: MagicMock,
        mock_llm_analysis_response: dict,
        mock_hyde_response: str,
    ) -> None:
        """Test full expand_query flow."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        # Mock two LLM calls - analysis then HyDE
        analysis_response = MagicMock()
        analysis_response.content = json.dumps(mock_llm_analysis_response)

        hyde_response = MagicMock()
        hyde_response.content = mock_hyde_response

        processor.llm.invoke.side_effect = [analysis_response, hyde_response]

        result = processor.expand_query("What are NYC property taxes?")

        assert isinstance(result, QueryResult)
        assert result.original_query == "What are NYC property taxes?"
        assert "Department of Finance" in result.expanded_query
        assert result.analysis.is_relevant is True
        assert result.analysis.jurisdiction == "nyc"


class TestQueryProcessorJurisdictionDetection:
    """Tests specifically for jurisdiction detection."""

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_federal_jurisdiction_detected(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test that federal/IRS queries are flagged."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        federal_response = {
            "is_relevant": True,
            "relevance_score": 0.8,
            "relevance_warning": None,
            "jurisdiction": "federal",
            "jurisdiction_warning": "This system covers NYC tax law only. For IRS questions, consult irs.gov.",
            "detected_intent": "asking about federal tax filing",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(federal_response)
        processor.llm.invoke.return_value = mock_response

        analysis = processor.analyze_query("When is my IRS 1040 due?")

        assert analysis.jurisdiction == "federal"
        assert analysis.jurisdiction_warning is not None
        assert "NYC" in analysis.jurisdiction_warning

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_mixed_jurisdiction_detected(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test that mixed federal/local queries are flagged."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        mixed_response = {
            "is_relevant": True,
            "relevance_score": 0.7,
            "relevance_warning": None,
            "jurisdiction": "mixed",
            "jurisdiction_warning": "This query mentions both federal and NYC taxes.",
            "detected_intent": "comparing federal and NYC tax obligations",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(mixed_response)
        processor.llm.invoke.return_value = mock_response

        analysis = processor.analyze_query(
            "How do NYC taxes differ from federal taxes?"
        )

        assert analysis.jurisdiction == "mixed"


class TestQueryProcessorRelevanceFiltering:
    """Tests specifically for relevance filtering."""

    @patch("app.query_engine.QueryProcessor.__init__", return_value=None)
    def test_irrelevant_query_soft_warning(
        self,
        mock_init: MagicMock,
    ) -> None:
        """Test that irrelevant queries get soft warning."""
        processor = QueryProcessor.__new__(QueryProcessor)
        processor.conversation_history = []
        processor.llm = MagicMock()

        irrelevant_response = {
            "is_relevant": False,
            "relevance_score": 0.1,
            "relevance_warning": "This appears to be a cooking question, not related to NYC tax law.",
            "jurisdiction": "unknown",
            "jurisdiction_warning": None,
            "detected_intent": "asking for baking instructions",
        }

        mock_response = MagicMock()
        mock_response.content = json.dumps(irrelevant_response)
        processor.llm.invoke.return_value = mock_response

        analysis = processor.analyze_query("How do I bake a cake?")

        assert analysis.is_relevant is False
        assert analysis.relevance_warning is not None
        assert "cooking" in analysis.relevance_warning.lower()
