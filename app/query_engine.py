"""Query processing engine with HyDE for NYC Tax Law RAG.

This module provides query analysis, expansion using HyDE (Hypothetical
Document Embeddings), and conversation context management for improved
legal document retrieval.
"""

from __future__ import annotations

import json
from typing import List, Literal, Optional, Any

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Analysis results for a user query.

    Attributes:
        is_relevant: Whether the query is related to tax/finance/legal matters.
        relevance_score: Confidence score (0-1) for relevance.
        relevance_warning: Warning message if query seems irrelevant.
        jurisdiction: Detected tax jurisdiction context.
        jurisdiction_warning: Warning if federal/IRS detected in NYC-only system.
        detected_intent: Brief description of what the user is asking.
    """

    is_relevant: bool = Field(description="Is query tax/finance related?")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance confidence")
    relevance_warning: Optional[str] = Field(
        default=None, description="Warning if not relevant"
    )
    jurisdiction: Literal["nyc", "federal", "mixed", "unknown"] = Field(
        default="unknown", description="Detected jurisdiction"
    )
    jurisdiction_warning: Optional[str] = Field(
        default=None, description="Warning for federal/IRS queries"
    )
    detected_intent: str = Field(default="", description="What user is asking about")


class QueryResult(BaseModel):
    """Complete result from query processing.

    Attributes:
        original_query: The user's original query text.
        expanded_query: HyDE-generated hypothetical answer for embedding.
        analysis: Query analysis results.
        conversation_context: Formatted conversation history for context.
    """

    original_query: str = Field(description="Original user query")
    expanded_query: str = Field(description="HyDE hypothetical answer")
    analysis: QueryAnalysis = Field(description="Query analysis results")
    conversation_context: str = Field(
        default="", description="Formatted conversation history"
    )


class QueryProcessor:
    """Processes user queries using HyDE and conversation context.

    This processor analyzes queries for relevance and jurisdiction,
    generates hypothetical answers using HyDE for better embedding
    similarity, and maintains conversation history for follow-up queries.

    Attributes:
        MAX_HISTORY: Maximum number of conversation exchanges to retain.
    """

    MAX_HISTORY = 5

    ANALYSIS_PROMPT = """Analyze this query for an NYC Tax Law system.

Query: "{query}"
{context_section}

Return a JSON object with these exact fields:
{{
  "is_relevant": true or false (is this about tax, finance, legal, or government matters?),
  "relevance_score": 0.0 to 1.0 (confidence that this is a valid tax question),
  "relevance_warning": "message explaining why this may not be relevant" or null,
  "jurisdiction": "nyc" or "federal" or "mixed" or "unknown",
  "jurisdiction_warning": "message if IRS/federal tax detected" or null,
  "detected_intent": "brief description of what the user wants to know"
}}

Guidelines:
- "federal" = mentions IRS, federal tax, 1040, W-2 in federal context
- "nyc" = specifically mentions NYC, New York City, city taxes
- "mixed" = mentions both federal and local aspects
- "unknown" = general tax question without specific jurisdiction
- For federal queries: warn that this system covers NYC tax law only
- Be lenient on relevance - tax-adjacent topics (permits, fees, assessments) are OK

IMPORTANT: Return ONLY valid JSON, no markdown formatting."""

    HYDE_PROMPT = """You are an expert on NYC Tax Law. Given the user's question, write a
detailed hypothetical answer AS IF you found it in the NYC Administrative Code or NYC tax regulations.

Your hypothetical answer should:
- Use specific legal terminology found in NYC tax law
- Reference section numbers (like "Section 11-xxx" even if approximate)
- Mention relevant NYC entities (Commissioner of Finance, Department of Finance, Tax Commission)
- Include procedural details where relevant
- Sound like it came from an official legal document

User Question: {query}
{context_section}

Write a 2-3 paragraph hypothetical answer that would help find relevant legal sections:"""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize the QueryProcessor.

        Args:
            model: OpenAI model name to use for processing.
        """
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.model = model
        self.conversation_history: List[dict] = []

    def _get_context_section(self) -> str:
        """Format conversation history for prompts.

        Returns:
            Formatted context string, or empty if no history.
        """
        if not self.conversation_history:
            return ""

        history_text = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:500]}"
            for h in self.conversation_history[-self.MAX_HISTORY:]
        )
        return f"\nConversation Context:\n{history_text}\n"

    def _parse_analysis_response(self, response_text: str) -> QueryAnalysis:
        """Parse LLM response into QueryAnalysis.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Parsed QueryAnalysis object.
        """
        # Clean response - remove markdown code blocks if present
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return QueryAnalysis(
                is_relevant=data.get("is_relevant", True),
                relevance_score=data.get("relevance_score", 0.5),
                relevance_warning=data.get("relevance_warning"),
                jurisdiction=data.get("jurisdiction", "unknown"),
                jurisdiction_warning=data.get("jurisdiction_warning"),
                detected_intent=data.get("detected_intent", ""),
            )
        except (json.JSONDecodeError, Exception):
            # Default to permissive analysis on parse failure
            return QueryAnalysis(
                is_relevant=True,
                relevance_score=0.5,
                relevance_warning=None,
                jurisdiction="unknown",
                jurisdiction_warning=None,
                detected_intent="Unable to analyze query",
            )

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a query for relevance and jurisdiction.

        Args:
            query: The user's query text.

        Returns:
            QueryAnalysis with relevance and jurisdiction information.
        """
        context_section = self._get_context_section()
        prompt = self.ANALYSIS_PROMPT.format(
            query=query, context_section=context_section
        )

        try:
            response = self.llm.invoke(prompt)
            return self._parse_analysis_response(response.content)
        except Exception:
            # On error, return permissive defaults
            return QueryAnalysis(
                is_relevant=True,
                relevance_score=0.5,
                relevance_warning=None,
                jurisdiction="unknown",
                jurisdiction_warning=None,
                detected_intent="Query analysis unavailable",
            )

    def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer using HyDE.

        Creates a detailed hypothetical answer that sounds like it came
        from NYC tax law documentation. This hypothetical answer is then
        embedded for similarity search, yielding better results than
        embedding the original query.

        Args:
            query: The user's query text.

        Returns:
            Hypothetical answer text for embedding.
        """
        context_section = self._get_context_section()
        prompt = self.HYDE_PROMPT.format(
            query=query, context_section=context_section
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception:
            # Fall back to original query if HyDE fails
            return query

    def expand_query(self, query: str) -> QueryResult:
        """Process a query with analysis and HyDE expansion.

        Main entry point for query processing. Analyzes the query for
        relevance and jurisdiction, then generates a hypothetical answer
        for improved retrieval.

        Args:
            query: The user's query text.

        Returns:
            QueryResult with analysis and expanded query.
        """
        # Analyze the query
        analysis = self.analyze_query(query)

        # Generate hypothetical answer (HyDE)
        hypothetical = self.generate_hypothetical_answer(query)

        # Get context for the result
        context = self._get_context_section()

        return QueryResult(
            original_query=query,
            expanded_query=hypothetical,
            analysis=analysis,
            conversation_context=context,
        )

    def add_to_history(self, query: str, response: str) -> None:
        """Add a query-response pair to conversation history.

        Args:
            query: The user's query.
            response: The system's response.
        """
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Trim to max history
        if len(self.conversation_history) > self.MAX_HISTORY * 2:
            self.conversation_history = self.conversation_history[-(self.MAX_HISTORY * 2):]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[dict]:
        """Get current conversation history.

        Returns:
            List of conversation exchanges.
        """
        return self.conversation_history.copy()
