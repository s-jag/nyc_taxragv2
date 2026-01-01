"""TaxAdvisor - Final answer generation with chain of verification.

This module provides the TaxAdvisor class that orchestrates the full RAG
pipeline and generates professional tax guidance with strict citations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.ingest import EnrichedDocument
from app.query_engine import QueryAnalysis, QueryProcessor, QueryResult
from app.ranker import GradedDocument, ReRanker
from app.retriever import Retriever


class SourceInfo(BaseModel):
    """Information about a cited source.

    Attributes:
        section_id: Legal section identifier (e.g., "11-121").
        summary: Brief description of the section.
        relevance_score: Score from ReRanker (0-10), or 10.0 if disabled.
        text_preview: First 200 characters of the section text.
    """

    section_id: str = Field(description="Legal section ID")
    summary: str = Field(default="", description="Section summary")
    relevance_score: float = Field(default=10.0, description="Relevance score 0-10")
    text_preview: str = Field(default="", description="First 200 chars of text")


class DebugInfo(BaseModel):
    """Pipeline debugging information.

    Attributes:
        query_analysis: Analysis results from QueryProcessor.
        hyde_query: The HyDE-expanded query used for retrieval.
        documents_retrieved: Number of documents from initial retrieval.
        documents_after_rerank: Number after re-ranking filter.
        reranker_enabled: Whether re-ranking was active.
        model_used: Model name used for generation.
        processing_time_ms: Total processing time in milliseconds.
        best_score: Best relevance score among retrieved documents.
        avg_score: Average relevance score of documents after reranking.
        fallback_model: Model used for fallback (if applicable).
    """

    query_analysis: Dict[str, Any] = Field(default_factory=dict)
    hyde_query: str = Field(default="")
    documents_retrieved: int = Field(default=0)
    documents_after_rerank: int = Field(default=0)
    reranker_enabled: bool = Field(default=True)
    model_used: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    best_score: float = Field(default=0.0)
    avg_score: float = Field(default=0.0)
    fallback_model: str = Field(default="")


class TaxAdvisorResponse(BaseModel):
    """Complete response from the TaxAdvisor.

    Attributes:
        answer: The full response text with citations.
        sources: List of cited sections with metadata.
        warnings: Any jurisdiction or relevance warnings.
        debug_info: Pipeline diagnostics for troubleshooting.
        is_fallback: Whether this response used the fallback model.
        confidence_level: Confidence level - "high", "low", or "none".
    """

    answer: str = Field(description="Full response with citations")
    sources: List[SourceInfo] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    debug_info: DebugInfo = Field(default_factory=DebugInfo)
    is_fallback: bool = Field(default=False)
    confidence_level: str = Field(default="high")


class TaxAdvisor:
    """Orchestrates the full RAG pipeline for NYC tax guidance.

    This class combines all pipeline components to provide professional
    tax guidance with strict citations:
    - QueryProcessor for HyDE expansion and query analysis
    - Retriever for parent-child document retrieval
    - ReRanker for relevance filtering (optional, toggleable)
    - LLM (o3-mini) for final answer generation

    Attributes:
        SILENCE_RESPONSE: Message returned when no relevant docs found.
    """

    SILENCE_RESPONSE = (
        "I analyzed NYC Tax Laws but could not find regulations "
        "specific to your query. This may be because:\n\n"
        "- The topic falls under Federal/IRS jurisdiction (not NYC)\n"
        "- The specific provision isn't in our database\n"
        "- The query relates to a different area of law\n\n"
        "Please consult a licensed tax professional or the NYC "
        "Department of Finance directly."
    )

    FALLBACK_SYSTEM_PROMPT = """You are a helpful tax assistant providing general guidance.

IMPORTANT CONTEXT:
- The user asked about NYC taxes
- Our legal database found some potentially related sections, but none were highly relevant
- You should provide helpful general guidance based on your knowledge

RULES:
1. DO NOT cite specific NYC Tax Law section numbers as authoritative
2. Clearly indicate this is general guidance, not legal advice
3. Be helpful but recommend consulting official sources
4. You may reference the retrieved sections as "potentially related" context

Provide a helpful, informative response that:
- Answers the user's question to the best of your general knowledge
- Acknowledges the limitations of this guidance
- Suggests next steps for getting authoritative information"""

    FALLBACK_DISCLAIMER = (
        "**General Guidance Notice**\n\n"
        "This response provides general tax guidance based on common knowledge. "
        "The NYC Tax Law database did not contain highly relevant sections for "
        "your specific question.\n\n"
        "**The sections shown below may be tangentially related but are not "
        "directly applicable.**\n\n"
        "Please consult a licensed tax professional or the NYC Department of "
        "Finance for authoritative guidance."
    )

    SYSTEM_PROMPT = """You are an NYC Tax Law Assistant for professional tax preparers. Your role is to provide accurate, actionable guidance based ONLY on the provided legal context.

## Your Audience
Professional tax preparers preparing client returns. They need:
- Precise legal citations for documentation
- Clear applicability criteria
- Practical filing implications
- Relevant exemptions and deductions

## Critical Rules

1. **CITE EVERYTHING**: Every legal claim MUST include the section reference (e.g., "pursuant to § 11-121" or "under Section 11-503.2").

3. **REASONING FIRST**: Before answering, analyze:
   - Which sections are relevant and why
   - How they apply to this specific situation
   - Any exceptions or qualifications that apply

4. **PRACTICAL GUIDANCE**: For each applicable provision:
   - State what it requires/allows
   - Note any thresholds, deadlines, or conditions
   - Identify related deductions or exemptions mentioned
   - Flag potential edge cases

5. **ACKNOWLEDGE LIMITS**: If the context discusses a related but different topic, clearly state what IS covered vs what the user is actually asking about.

## Response Structure

### Analysis
[Your reasoning about which sections apply and how]

### Answer
[Direct answer with inline citations: "The taxpayer is required to... (§ 11-XXX)"]

### Applicable Sections
[List each relevant section with a one-line summary of its relevance]

### Practical Notes
[Filing implications, related deductions, common pitfalls, edge cases]

### Limitations
[What this answer does NOT cover, when to consult further]"""

    USER_PROMPT_TEMPLATE = """## User Question
{query}

## Legal Context
The following NYC Tax Law sections have been retrieved as potentially relevant:

{context}

---

Based on the above legal context along with your knowledge and reasoning, provide professional tax guidance following the response structure. Remember to cite every claim with the specific section reference (§ XX-XXX)."""

    def __init__(
        self,
        vectorstore: Any,
        docstore_path: str | Path,
        generation_model: str = "o3-mini",
        fallback_model: str = "gpt-5.2-pro",
        reranker_enabled: bool = True,
        reranker_threshold: float = 7.0,
        confidence_threshold: float = 8.0,
    ) -> None:
        """Initialize the TaxAdvisor.

        Args:
            vectorstore: ChromaDB vector store instance.
            docstore_path: Path to the docstore.json file.
            generation_model: OpenAI model for answer generation.
            fallback_model: OpenAI model for low-confidence fallback.
            reranker_enabled: Whether to use re-ranking filter.
            reranker_threshold: Minimum score for documents to pass re-ranking.
            confidence_threshold: Minimum best score for high-confidence answers.
        """
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        load_dotenv()

        self.processor = QueryProcessor()
        self.retriever = Retriever(vectorstore, docstore_path)
        self.reranker = ReRanker(
            enabled=reranker_enabled,
            threshold=reranker_threshold,
        )
        self.generation_model = generation_model
        self.fallback_model = fallback_model
        self.confidence_threshold = confidence_threshold

        # o3-mini doesn't support temperature parameter
        if "o3" in generation_model:
            self.llm = ChatOpenAI(model=generation_model)
        else:
            self.llm = ChatOpenAI(model=generation_model, temperature=0)

        # Initialize fallback LLM
        self.fallback_llm = ChatOpenAI(model=fallback_model, temperature=0.7)

    def _format_context(self, graded_docs: List[GradedDocument]) -> str:
        """Format graded documents into context for the LLM.

        Args:
            graded_docs: List of graded documents from ReRanker.

        Returns:
            Formatted context string.
        """
        sections = []
        for graded in graded_docs:
            doc = graded.document
            section = f"""═══════════════════════════════════════════════════
§ {doc.section_id}
Category: {doc.metadata.tax_category} | Type: {doc.metadata.document_type}
Summary: {doc.metadata.summary}
Relevance Score: {graded.score:.1f}/10
───────────────────────────────────────────────────
{doc.text}
═══════════════════════════════════════════════════"""
            sections.append(section)

        return "\n\n".join(sections)

    def _create_sources(self, graded_docs: List[GradedDocument]) -> List[SourceInfo]:
        """Create source info list from graded documents.

        Args:
            graded_docs: List of graded documents.

        Returns:
            List of SourceInfo objects.
        """
        return [
            SourceInfo(
                section_id=g.document.section_id,
                summary=g.document.metadata.summary,
                relevance_score=g.score,
                text_preview=g.document.text[:200],
            )
            for g in graded_docs
        ]

    def _collect_warnings(self, query_result: QueryResult) -> List[str]:
        """Collect any warnings from query analysis.

        Args:
            query_result: Result from query processing.

        Returns:
            List of warning strings.
        """
        warnings = []

        analysis = query_result.analysis
        if analysis.relevance_warning:
            warnings.append(f"Relevance: {analysis.relevance_warning}")
        if analysis.jurisdiction_warning:
            warnings.append(f"Jurisdiction: {analysis.jurisdiction_warning}")
        if analysis.jurisdiction == "federal":
            warnings.append(
                "This query appears to be about Federal/IRS taxes. "
                "This system covers NYC tax law only."
            )
        elif analysis.jurisdiction == "mixed":
            warnings.append(
                "This query involves both Federal and NYC taxes. "
                "Only NYC-specific guidance is provided."
            )

        return warnings

    def _create_silence_response(
        self,
        query_result: QueryResult,
        documents_retrieved: int,
        start_time: float,
    ) -> TaxAdvisorResponse:
        """Create response when no relevant documents found.

        Args:
            query_result: Result from query processing.
            documents_retrieved: Number of docs initially retrieved.
            start_time: Processing start time.

        Returns:
            TaxAdvisorResponse with silence message.
        """
        warnings = self._collect_warnings(query_result)
        warnings.append("No documents passed relevance filtering.")

        return TaxAdvisorResponse(
            answer=self.SILENCE_RESPONSE,
            sources=[],
            warnings=warnings,
            debug_info=DebugInfo(
                query_analysis=query_result.analysis.model_dump(),
                hyde_query=query_result.expanded_query,
                documents_retrieved=documents_retrieved,
                documents_after_rerank=0,
                reranker_enabled=self.reranker.enabled,
                model_used=self.generation_model,
                processing_time_ms=(time.time() - start_time) * 1000,
            ),
        )

    def _generate_answer(
        self,
        user_query: str,
        query_result: QueryResult,
        graded_docs: List[GradedDocument],
        documents_retrieved: int,
        start_time: float,
    ) -> TaxAdvisorResponse:
        """Generate the final answer using the LLM.

        Args:
            user_query: Original user query.
            query_result: Result from query processing.
            graded_docs: Documents that passed re-ranking.
            documents_retrieved: Number of docs initially retrieved.
            start_time: Processing start time.

        Returns:
            Complete TaxAdvisorResponse.
        """
        # Format context
        context = self._format_context(graded_docs)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_PROMPT_TEMPLATE.format(
                    query=user_query,
                    context=context,
                ),
            },
        ]

        # Generate response
        try:
            response = self.llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = (
                f"An error occurred while generating the response: {str(e)}\n\n"
                "Please try again or consult the source documents directly."
            )

        # Collect metadata
        sources = self._create_sources(graded_docs)
        warnings = self._collect_warnings(query_result)

        # Calculate confidence scores
        best_score = max(g.score for g in graded_docs) if graded_docs else 0.0
        avg_score = (
            sum(g.score for g in graded_docs) / len(graded_docs)
            if graded_docs
            else 0.0
        )

        return TaxAdvisorResponse(
            answer=answer,
            sources=sources,
            warnings=warnings,
            debug_info=DebugInfo(
                query_analysis=query_result.analysis.model_dump(),
                hyde_query=query_result.expanded_query,
                documents_retrieved=documents_retrieved,
                documents_after_rerank=len(graded_docs),
                reranker_enabled=self.reranker.enabled,
                model_used=self.generation_model,
                processing_time_ms=(time.time() - start_time) * 1000,
                best_score=best_score,
                avg_score=avg_score,
            ),
            is_fallback=False,
            confidence_level="high",
        )

    def _format_fallback_context(self, graded_docs: List[GradedDocument]) -> str:
        """Format graded documents as potentially related context for fallback.

        Args:
            graded_docs: List of graded documents from ReRanker.

        Returns:
            Formatted context string for fallback prompt.
        """
        sections = []
        for graded in graded_docs:
            doc = graded.document
            section = f"""Section § {doc.section_id} (Relevance: {graded.score:.1f}/10)
Summary: {doc.metadata.summary}
Preview: {doc.text[:500]}..."""
            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def _generate_fallback_answer(
        self,
        user_query: str,
        query_result: QueryResult,
        graded_docs: List[GradedDocument],
        documents_retrieved: int,
        start_time: float,
        best_score: float,
        avg_score: float,
    ) -> TaxAdvisorResponse:
        """Generate fallback answer using GPT-5.2 pro for low-confidence cases.

        Args:
            user_query: Original user query.
            query_result: Result from query processing.
            graded_docs: Documents that passed re-ranking but with low confidence.
            documents_retrieved: Number of docs initially retrieved.
            start_time: Processing start time.
            best_score: Best relevance score among documents.
            avg_score: Average relevance score of documents.

        Returns:
            TaxAdvisorResponse with fallback answer and disclaimer.
        """
        # Format context for fallback
        context = self._format_fallback_context(graded_docs)

        # Build messages for fallback LLM
        messages = [
            {"role": "system", "content": self.FALLBACK_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {user_query}\n\nPotentially related sections from NYC Tax Law:\n\n{context}",
            },
        ]

        # Generate response
        try:
            response = self.fallback_llm.invoke(messages)
            raw_answer = response.content
        except Exception as e:
            raw_answer = (
                f"An error occurred while generating the response: {str(e)}\n\n"
                "Please try again or consult a tax professional directly."
            )

        # Prepend disclaimer to answer
        answer = f"{self.FALLBACK_DISCLAIMER}\n\n---\n\n{raw_answer}"

        # Collect metadata
        sources = self._create_sources(graded_docs)
        warnings = self._collect_warnings(query_result)
        warnings.append("Low confidence - general guidance provided")

        return TaxAdvisorResponse(
            answer=answer,
            sources=sources,
            warnings=warnings,
            debug_info=DebugInfo(
                query_analysis=query_result.analysis.model_dump(),
                hyde_query=query_result.expanded_query,
                documents_retrieved=documents_retrieved,
                documents_after_rerank=len(graded_docs),
                reranker_enabled=self.reranker.enabled,
                model_used=self.fallback_model,
                processing_time_ms=(time.time() - start_time) * 1000,
                best_score=best_score,
                avg_score=avg_score,
                fallback_model=self.fallback_model,
            ),
            is_fallback=True,
            confidence_level="low",
        )

    def answer_question(
        self,
        user_query: str,
        k: int = 5,
    ) -> TaxAdvisorResponse:
        """Answer a tax question using the full RAG pipeline.

        This method orchestrates the complete pipeline:
        1. Query expansion with HyDE
        2. Document retrieval with parent-child swapping
        3. Re-ranking and filtering (if enabled)
        4. Confidence check and routing:
           - High confidence (best_score >= confidence_threshold): o3-mini with citations
           - Low confidence (best_score < confidence_threshold): GPT-5.2 pro fallback
           - No confidence (no docs pass reranker): silence response

        Args:
            user_query: The user's tax question.
            k: Number of documents to retrieve.

        Returns:
            TaxAdvisorResponse with answer, sources, warnings, and debug info.
        """
        start_time = time.time()

        # Step 1: Expand query with HyDE
        query_result = self.processor.expand_query(user_query)

        # Step 2: Retrieve parent documents
        parents = self.retriever.search(query_result, k=k)
        documents_retrieved = len(parents)

        # Handle empty retrieval
        if not parents:
            return self._create_silence_response(
                query_result, documents_retrieved, start_time
            )

        # Step 3: Re-rank and filter
        graded_docs = self.reranker.grade_documents(user_query, parents)

        # Step 4: Silence check - no documents passed reranker
        if graded_docs is None:
            return self._create_silence_response(
                query_result, documents_retrieved, start_time
            )

        # Step 5: Calculate confidence metrics
        best_score = max(g.score for g in graded_docs)
        avg_score = sum(g.score for g in graded_docs) / len(graded_docs)

        # Step 6: Route based on confidence
        if best_score >= self.confidence_threshold:
            # HIGH CONFIDENCE → o3-mini with citations
            return self._generate_answer(
                user_query,
                query_result,
                graded_docs,
                documents_retrieved,
                start_time,
            )
        else:
            # LOW CONFIDENCE → GPT-5.2 pro fallback
            return self._generate_fallback_answer(
                user_query,
                query_result,
                graded_docs,
                documents_retrieved,
                start_time,
                best_score,
                avg_score,
            )

    def add_to_history(self, query: str, response: str) -> None:
        """Add a query-response pair to conversation history.

        Args:
            query: The user's query.
            response: The system's response.
        """
        self.processor.add_to_history(query, response)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.processor.clear_history()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.

        Returns:
            Dictionary with current settings.
        """
        return {
            "generation_model": self.generation_model,
            "fallback_model": self.fallback_model,
            "confidence_threshold": self.confidence_threshold,
            "reranker": self.reranker.get_config(),
            "parent_count": self.retriever.parent_count,
        }
