"""Retriever for parent documents from child search results.

This module provides the Retriever class that implements the "super chunk swapper"
pattern: search small children, return full parent documents for LLM context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingest import EnrichedDocument
from app.query_engine import QueryResult


class Retriever:
    """Retrieves full parent documents from child search results.

    This class implements parent-child retrieval: searching the vector store
    for matching children, then returning their full parent documents for
    complete legal context.

    Attributes:
        vectorstore: ChromaDB vector store containing child documents.
        parents: Dictionary mapping parent IDs to EnrichedDocument objects.
    """

    def __init__(
        self,
        vectorstore: Any,
        docstore_path: str | Path,
    ) -> None:
        """Initialize the Retriever.

        Args:
            vectorstore: ChromaDB vector store instance.
            docstore_path: Path to the docstore.json file containing parents.
        """
        self.vectorstore = vectorstore
        self.parents: Dict[str, EnrichedDocument] = self._load_docstore(docstore_path)

    def _load_docstore(self, path: str | Path) -> Dict[str, EnrichedDocument]:
        """Load parent documents indexed by ID.

        Args:
            path: Path to docstore.json file.

        Returns:
            Dictionary mapping parent ID to EnrichedDocument.

        Raises:
            FileNotFoundError: If docstore file doesn't exist.
        """
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {doc["id"]: EnrichedDocument(**doc) for doc in data}

    def search(
        self,
        query_result: QueryResult,
        k: int = 5,
    ) -> List[EnrichedDocument]:
        """Search for relevant parents using HyDE-expanded query.

        Searches the vector store for matching children using the
        hypothetical answer from HyDE, then returns unique parent documents.

        Args:
            query_result: QueryResult containing the expanded query.
            k: Number of child results to retrieve.

        Returns:
            List of unique EnrichedDocument parents.
        """
        # Search children with HyDE-expanded query
        children = self.vectorstore.similarity_search(
            query_result.expanded_query,
            k=k,
        )

        # Get unique parent IDs while preserving order
        seen: set[str] = set()
        parent_ids: List[str] = []
        for doc in children:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent_ids.append(parent_id)

        # Return full parent documents
        return [
            self.parents[pid]
            for pid in parent_ids
            if pid in self.parents
        ]

    def search_with_scores(
        self,
        query_result: QueryResult,
        k: int = 5,
    ) -> List[tuple[EnrichedDocument, float]]:
        """Search for relevant parents with similarity scores.

        Like search(), but also returns the best similarity score
        for each parent (from its highest-scoring child).

        Args:
            query_result: QueryResult containing the expanded query.
            k: Number of child results to retrieve.

        Returns:
            List of (EnrichedDocument, score) tuples, sorted by score.
        """
        # Search children with scores
        results = self.vectorstore.similarity_search_with_score(
            query_result.expanded_query,
            k=k,
        )

        # Track best score for each parent
        parent_scores: Dict[str, float] = {}
        for doc, score in results:
            parent_id = doc.metadata.get("parent_id")
            if parent_id:
                # Keep the best (lowest) score for each parent
                if parent_id not in parent_scores or score < parent_scores[parent_id]:
                    parent_scores[parent_id] = score

        # Return parents with scores, sorted by score
        parent_results = [
            (self.parents[pid], score)
            for pid, score in parent_scores.items()
            if pid in self.parents
        ]
        return sorted(parent_results, key=lambda x: x[1])

    def get_parent(self, parent_id: str) -> Optional[EnrichedDocument]:
        """Get a parent document by ID.

        Args:
            parent_id: UUID of the parent document.

        Returns:
            EnrichedDocument if found, None otherwise.
        """
        return self.parents.get(parent_id)

    def search_by_section(
        self,
        section_id: str,
        k: int = 5,
    ) -> List[EnrichedDocument]:
        """Search for documents by section ID.

        Uses metadata filtering to find children from a specific section,
        then returns their parents.

        Args:
            section_id: Legal section ID (e.g., "11-121").
            k: Maximum number of results.

        Returns:
            List of matching EnrichedDocument parents.
        """
        # Use metadata filter for exact section matching
        children = self.vectorstore.similarity_search(
            "",  # Empty query - we're filtering by metadata
            k=k,
            filter={"section_id": section_id},
        )

        # Get unique parent IDs
        seen: set[str] = set()
        parent_ids: List[str] = []
        for doc in children:
            parent_id = doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen:
                seen.add(parent_id)
                parent_ids.append(parent_id)

        return [
            self.parents[pid]
            for pid in parent_ids
            if pid in self.parents
        ]

    @property
    def parent_count(self) -> int:
        """Return the number of loaded parent documents."""
        return len(self.parents)
