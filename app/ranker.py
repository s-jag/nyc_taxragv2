"""Re-ranking guardrail for filtering irrelevant documents.

This module provides the ReRanker class that grades retrieved documents
for relevance and filters out irrelevant results to prevent hallucination.
"""

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, Field

from app.ingest import EnrichedDocument


class GradedDocument(BaseModel):
    """A document with its relevance grade.

    Attributes:
        document: The original enriched document.
        score: Relevance score from 0-10.
        reasoning: Explanation for the score.
    """

    document: EnrichedDocument = Field(description="The graded document")
    score: float = Field(ge=0.0, le=10.0, description="Relevance score 0-10")
    reasoning: str = Field(default="", description="Explanation for score")


class ReRanker:
    """Grades and filters documents for relevance to prevent hallucination.

    This class implements a re-ranking guardrail that:
    1. Uses an LLM to score each document for relevance (0-10)
    2. Filters out documents below the threshold
    3. Returns None if no documents pass (signals retrieval failure)

    The re-ranker can be toggled on/off for A/B testing.

    Attributes:
        DEFAULT_THRESHOLD: Default minimum score for a document to pass.
        enabled: Whether re-ranking is active.
        threshold: Minimum score for documents to pass filtering.
    """

    DEFAULT_THRESHOLD = 7.0

    GRADING_PROMPT = """You are grading the relevance of a legal document to a user's question.

User Question: {query}

Document Section ID: {section_id}
Document Summary: {summary}
Document Text (first 1000 chars):
{text}

Grade this document's relevance to the user's question on a scale of 0-10:
- 10: Directly answers the question with specific legal provisions
- 7-9: Highly relevant, discusses the same legal topic
- 4-6: Somewhat relevant, related topic but not directly applicable
- 1-3: Marginally relevant, different tax topic
- 0: Completely irrelevant, discusses unrelated matters

Return a JSON object with exactly these fields:
{{
  "score": <number 0-10>,
  "reasoning": "<brief explanation>"
}}

IMPORTANT: If the document discusses a DIFFERENT tax topic than what the user is asking about, score it 0-3.
Return ONLY valid JSON, no markdown formatting."""

    def __init__(
        self,
        model: str = "gpt-4o",
        threshold: float = DEFAULT_THRESHOLD,
        enabled: bool = True,
    ) -> None:
        """Initialize the ReRanker.

        Args:
            model: OpenAI model name to use for grading.
            threshold: Minimum score for documents to pass (0-10).
            enabled: Whether re-ranking is active. If False, all docs pass through.
        """
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.model = model
        self.threshold = threshold
        self.enabled = enabled

    def _parse_grading_response(self, response_text: str) -> tuple[float, str]:
        """Parse LLM grading response.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Tuple of (score, reasoning).
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
            score = float(data.get("score", 5.0))
            # Clamp score to valid range
            score = max(0.0, min(10.0, score))
            reasoning = data.get("reasoning", "")
            return score, reasoning
        except (json.JSONDecodeError, ValueError, TypeError):
            # Default to middle score on parse failure
            return 5.0, "Unable to parse grading response"

    def _grade_single(
        self,
        query: str,
        document: EnrichedDocument,
    ) -> GradedDocument:
        """Grade a single document for relevance.

        Args:
            query: The user's query.
            document: The document to grade.

        Returns:
            GradedDocument with score and reasoning.
        """
        prompt = self.GRADING_PROMPT.format(
            query=query,
            section_id=document.section_id,
            summary=document.metadata.summary,
            text=document.text[:1000],
        )

        try:
            response = self.llm.invoke(prompt)
            score, reasoning = self._parse_grading_response(response.content)
        except Exception:
            # On error, give benefit of doubt with middle score
            score = 5.0
            reasoning = "Grading failed, using default score"

        return GradedDocument(
            document=document,
            score=score,
            reasoning=reasoning,
        )

    def grade_documents(
        self,
        query: str,
        documents: List[EnrichedDocument],
    ) -> Optional[List[GradedDocument]]:
        """Grade and filter documents for relevance.

        Grades each document and filters out those below the threshold.
        Returns None if no documents pass, signaling retrieval failure.

        Args:
            query: The user's query.
            documents: List of documents to grade.

        Returns:
            List of GradedDocument that passed filtering, sorted by score descending.
            Returns None if no documents pass the threshold (retrieval failure).
            Returns empty list if input is empty.
        """
        # Handle empty input
        if not documents:
            return []

        # If disabled, pass all documents through with max score
        if not self.enabled:
            return [
                GradedDocument(
                    document=doc,
                    score=10.0,
                    reasoning="Re-ranking disabled, passing through",
                )
                for doc in documents
            ]

        # Grade each document
        graded: List[GradedDocument] = []
        for doc in documents:
            graded_doc = self._grade_single(query, doc)
            graded.append(graded_doc)

        # Filter by threshold
        passed = [g for g in graded if g.score >= self.threshold]

        # Return None if nothing passed (retrieval failure signal)
        if not passed:
            return None

        # Sort by score descending
        passed.sort(key=lambda x: x.score, reverse=True)
        return passed

    def get_config(self) -> dict:
        """Get current configuration.

        Returns:
            Dictionary with current settings.
        """
        return {
            "model": self.model,
            "threshold": self.threshold,
            "enabled": self.enabled,
        }
