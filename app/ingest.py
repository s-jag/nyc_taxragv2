"""Legal document parser and enricher for NYC Tax Law.

This module provides functionality to parse legal documents by section,
creating structured Parent Documents, and enriching them with AI-generated
metadata for improved RAG retrieval.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Union, List, Any

from pydantic import BaseModel, Field
from tqdm import tqdm


class ParentDocument(BaseModel):
    """Represents a complete legal section as a parent document.

    Attributes:
        id: Unique identifier (UUID) for the document.
        section_id: Legal section code (e.g., "11-121").
        text: Full text content of the section.
        source: Source identifier for the document.
    """

    id: str = Field(description="Unique UUID for the document")
    section_id: str = Field(description="Legal section code (e.g., '11-121')")
    text: str = Field(description="Full text content of the section")
    source: str = Field(default="NYC Tax Law", description="Source identifier")


class DocumentMetadata(BaseModel):
    """AI-generated metadata for enhanced RAG retrieval.

    Attributes:
        summary: One-sentence summary of the section.
        keywords: Technical tax/legal terms found in the text.
        entities: Roles and parties mentioned (e.g., "City Collector").
        tax_category: Topic area classification.
        document_type: Type of legal content.
        hypothetical_questions: Questions this section might answer.
        cross_references: Other section IDs mentioned.
        applicable_parties: Who this section applies to.
    """

    summary: str = Field(default="", description="One-sentence summary")
    keywords: List[str] = Field(default_factory=list, description="5-10 tax terms")
    entities: List[str] = Field(default_factory=list, description="Roles mentioned")
    tax_category: str = Field(default="other", description="Topic classification")
    document_type: str = Field(default="other", description="Content type")
    hypothetical_questions: List[str] = Field(
        default_factory=list, description="Questions this section answers"
    )
    cross_references: List[str] = Field(
        default_factory=list, description="Referenced section IDs"
    )
    applicable_parties: List[str] = Field(
        default_factory=list, description="Who this applies to"
    )


class EnrichedDocument(BaseModel):
    """Parent document enriched with AI-generated metadata.

    Attributes:
        id: Unique identifier (UUID) for the document.
        section_id: Legal section code (e.g., "11-121").
        text: Full text content of the section.
        source: Source identifier for the document.
        metadata: AI-generated metadata for RAG.
    """

    id: str = Field(description="Unique UUID for the document")
    section_id: str = Field(description="Legal section code (e.g., '11-121')")
    text: str = Field(description="Full text content of the section")
    source: str = Field(default="NYC Tax Law", description="Source identifier")
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata, description="AI-generated metadata"
    )


class ChildDocument(BaseModel):
    """A chunk of a parent document for vector indexing.

    Child documents are smaller pieces of parent documents that are
    embedded and stored in the vector database. Each child maintains
    a link to its parent and inherits relevant metadata.

    Attributes:
        id: Unique identifier (UUID) for the child.
        parent_id: UUID of the parent document.
        section_id: Legal section code from parent.
        text: Chunk text content.
        chunk_index: Position of this chunk in the parent (0-indexed).
        keywords: Inherited from parent metadata.
        summary: Inherited from parent metadata.
        tax_category: Inherited from parent metadata.
        entities: Inherited from parent metadata.
        hypothetical_questions: Inherited from parent metadata.
        applicable_parties: Inherited from parent metadata.
    """

    id: str = Field(description="Unique UUID for the child")
    parent_id: str = Field(description="UUID of parent document")
    section_id: str = Field(description="Legal section code")
    text: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Position in parent (0-indexed)")
    # Inherited metadata
    keywords: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    tax_category: str = Field(default="other")
    entities: List[str] = Field(default_factory=list)
    hypothetical_questions: List[str] = Field(default_factory=list)
    applicable_parties: List[str] = Field(default_factory=list)


class LegalParser:
    """Parser for legal documents that splits by section symbol (§).

    This parser identifies legal sections using the section symbol (§) and
    extracts structured information including section IDs and full text content.
    """

    # Pattern to match section IDs like "11-121" or "11-207.1"
    # Handles both regular hyphen (-) and other dash variants (‐)
    SECTION_ID_PATTERN = re.compile(r'§\s*(\d+[-‐]\d+(?:\.\d+)?)')

    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace in text.

        Args:
            text: Raw text with potentially irregular whitespace.

        Returns:
            Text with normalized whitespace (single spaces, trimmed).
        """
        # Replace multiple whitespace characters (including newlines) with single space
        cleaned = re.sub(r'\s+', ' ', text)
        return cleaned.strip()

    def _extract_section_id(self, section_text: str) -> str | None:
        """Extract the section ID from section text.

        Args:
            section_text: Text starting with § symbol.

        Returns:
            Section ID (e.g., "11-121") or None if not found.
        """
        match = self.SECTION_ID_PATTERN.match(section_text)
        if match:
            # Normalize any dash variants to standard hyphen
            return match.group(1).replace('‐', '-')
        return None

    def load_data(self, file_path: str | Path) -> list[ParentDocument]:
        """Load and parse a legal document into Parent Documents.

        Reads the file and splits it by section symbol (§), creating a
        ParentDocument for each valid section.

        Args:
            file_path: Path to the legal document file.

        Returns:
            List of ParentDocument objects, one per legal section.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by § symbol, keeping the delimiter with each section
        # Uses lookahead to keep § with the following text
        raw_sections = re.split(r'(?=§)', content)

        documents: list[ParentDocument] = []

        for raw_section in raw_sections:
            # Skip empty sections or sections without §
            if not raw_section.strip() or not raw_section.strip().startswith('§'):
                continue

            section_id = self._extract_section_id(raw_section)

            if section_id is None:
                # Skip sections where we can't extract a valid ID
                continue

            cleaned_text = self._clean_whitespace(raw_section)

            document = ParentDocument(
                id=str(uuid.uuid4()),
                section_id=section_id,
                text=cleaned_text,
                source="NYC Tax Law"
            )
            documents.append(document)

        return documents


class DocumentEnricher:
    """Enriches documents with AI-generated metadata using LLM.

    This class uses OpenAI's GPT models to analyze legal text and generate
    structured metadata for improved RAG retrieval.
    """

    ENRICHMENT_PROMPT = """Analyze this NYC Tax Law section and return a JSON object with the following fields:

{{
  "summary": "One sentence summarizing what this section establishes or requires",
  "keywords": ["5-10 technical tax/legal terms found in this text"],
  "entities": ["Specific roles mentioned like 'City Collector', 'Commissioner of Finance', 'Comptroller'"],
  "tax_category": "one of: real_property | business | income | sales | penalties | exemptions | procedures | definitions | other",
  "document_type": "one of: definition | procedure | requirement | penalty | exemption | timeline | authority | reporting | other",
  "hypothetical_questions": ["3-5 questions a user might ask that this section answers"],
  "cross_references": ["Other section IDs mentioned, e.g., '11-203', '11-208.1'"],
  "applicable_parties": ["Who this applies to, from: property_owners | businesses | city_officials | taxpayers | assessors | collectors | finance_department"]
}}

IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text.

Section Text:
{text}"""

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize the enricher with specified LLM model.

        Args:
            model: OpenAI model name to use for enrichment.
        """
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv

        load_dotenv()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.model = model

    def _parse_llm_response(self, response_text: str) -> dict[str, Any]:
        """Parse LLM response into structured metadata.

        Args:
            response_text: Raw text response from LLM.

        Returns:
            Parsed dictionary with metadata fields.
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
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {}

    def _create_default_metadata(self) -> DocumentMetadata:
        """Create default metadata when LLM fails.

        Returns:
            DocumentMetadata with empty/default values.
        """
        return DocumentMetadata()

    def enrich_document(
        self, doc: ParentDocument, max_retries: int = 3
    ) -> EnrichedDocument:
        """Enrich a single document with AI-generated metadata.

        Args:
            doc: Parent document to enrich.
            max_retries: Maximum retry attempts on failure.

        Returns:
            EnrichedDocument with AI-generated metadata.
        """
        prompt = self.ENRICHMENT_PROMPT.format(text=doc.text)

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                parsed = self._parse_llm_response(response.content)

                if parsed:
                    metadata = DocumentMetadata(
                        summary=parsed.get("summary", ""),
                        keywords=parsed.get("keywords", []),
                        entities=parsed.get("entities", []),
                        tax_category=parsed.get("tax_category", "other"),
                        document_type=parsed.get("document_type", "other"),
                        hypothetical_questions=parsed.get(
                            "hypothetical_questions", []
                        ),
                        cross_references=parsed.get("cross_references", []),
                        applicable_parties=parsed.get("applicable_parties", []),
                    )
                else:
                    metadata = self._create_default_metadata()

                return EnrichedDocument(
                    id=doc.id,
                    section_id=doc.section_id,
                    text=doc.text,
                    source=doc.source,
                    metadata=metadata,
                )

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    # Final attempt failed, use default metadata
                    return EnrichedDocument(
                        id=doc.id,
                        section_id=doc.section_id,
                        text=doc.text,
                        source=doc.source,
                        metadata=self._create_default_metadata(),
                    )

        # Should not reach here, but satisfy type checker
        return EnrichedDocument(
            id=doc.id,
            section_id=doc.section_id,
            text=doc.text,
            source=doc.source,
            metadata=self._create_default_metadata(),
        )

    def enrich_batch(
        self,
        documents: List[ParentDocument],
        output_path: str | Path,
        checkpoint_interval: int = 10,
    ) -> List[EnrichedDocument]:
        """Enrich a batch of documents with progress tracking and checkpointing.

        Args:
            documents: List of parent documents to enrich.
            output_path: Path to save the enriched documents JSON.
            checkpoint_interval: Save checkpoint every N documents.

        Returns:
            List of enriched documents.
        """
        output_path = Path(output_path)
        enriched_docs: List[EnrichedDocument] = []

        # Check for existing checkpoint
        existing_ids: set[str] = set()
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    enriched_docs = [
                        EnrichedDocument(**doc) for doc in existing_data
                    ]
                    existing_ids = {doc.id for doc in enriched_docs}
                    print(f"Resuming from checkpoint: {len(enriched_docs)} docs loaded")
            except (json.JSONDecodeError, Exception):
                enriched_docs = []

        # Filter out already processed documents
        docs_to_process = [doc for doc in documents if doc.id not in existing_ids]

        if not docs_to_process:
            print("All documents already enriched.")
            return enriched_docs

        print(f"Enriching {len(docs_to_process)} documents with {self.model}...")

        for i, doc in enumerate(
            tqdm(docs_to_process, desc="Enriching documents", unit="doc")
        ):
            enriched_doc = self.enrich_document(doc)
            enriched_docs.append(enriched_doc)

            # Checkpoint save
            if (i + 1) % checkpoint_interval == 0:
                self._save_documents(enriched_docs, output_path)

        # Final save
        self._save_documents(enriched_docs, output_path)
        print(f"Saved {len(enriched_docs)} enriched documents to {output_path}")

        return enriched_docs

    def _save_documents(
        self, documents: List[EnrichedDocument], output_path: Path
    ) -> None:
        """Save enriched documents to JSON file.

        Args:
            documents: List of enriched documents.
            output_path: Path to save the JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                [doc.model_dump() for doc in documents],
                f,
                indent=2,
                ensure_ascii=False,
            )


class VectorStoreManager:
    """Manages the vector store for Parent-Child document indexing.

    This class handles:
    - Loading enriched parent documents
    - Splitting parents into child chunks
    - Creating and persisting vector store (ChromaDB local or Qdrant cloud)
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_COLLECTION_NAME = "nyc_tax_law"

    def __init__(
        self,
        persist_dir: str | Path = "vectorstore",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        use_qdrant: bool = False,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        """Initialize the VectorStoreManager.

        Args:
            persist_dir: Directory for ChromaDB persistence (local mode).
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
            use_qdrant: Whether to use Qdrant cloud instead of local ChromaDB.
            qdrant_url: Qdrant Cloud cluster URL.
            qdrant_api_key: Qdrant Cloud API key.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from dotenv import load_dotenv

        load_dotenv()

        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_qdrant = use_qdrant
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        self.embeddings = OpenAIEmbeddings()

    def load_enriched_documents(
        self, docstore_path: str | Path
    ) -> List[EnrichedDocument]:
        """Load enriched documents from JSON file.

        Args:
            docstore_path: Path to the docstore.json file.

        Returns:
            List of EnrichedDocument objects.

        Raises:
            FileNotFoundError: If docstore file doesn't exist.
        """
        docstore_path = Path(docstore_path)

        with open(docstore_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [EnrichedDocument(**doc) for doc in data]

    def create_children(
        self, parent: EnrichedDocument
    ) -> List[ChildDocument]:
        """Split a parent document into child chunks.

        Args:
            parent: The enriched parent document to split.

        Returns:
            List of ChildDocument objects with inherited metadata.
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(parent.text)

        children: List[ChildDocument] = []
        for i, chunk_text in enumerate(chunks):
            child = ChildDocument(
                id=str(uuid.uuid4()),
                parent_id=parent.id,
                section_id=parent.section_id,
                text=chunk_text,
                chunk_index=i,
                # Inherit all metadata from parent
                keywords=parent.metadata.keywords,
                summary=parent.metadata.summary,
                tax_category=parent.metadata.tax_category,
                entities=parent.metadata.entities,
                hypothetical_questions=parent.metadata.hypothetical_questions,
                applicable_parties=parent.metadata.applicable_parties,
            )
            children.append(child)

        return children

    def create_all_children(
        self, parents: List[EnrichedDocument]
    ) -> List[ChildDocument]:
        """Create child documents from all parents.

        Args:
            parents: List of enriched parent documents.

        Returns:
            List of all child documents.
        """
        all_children: List[ChildDocument] = []

        for parent in tqdm(parents, desc="Creating child chunks", unit="parent"):
            children = self.create_children(parent)
            all_children.extend(children)

        return all_children

    def _prepare_chroma_documents(
        self, children: List[ChildDocument]
    ) -> tuple[List[str], List[dict], List[str]]:
        """Prepare documents for ChromaDB ingestion.

        Args:
            children: List of child documents.

        Returns:
            Tuple of (texts, metadatas, ids) for ChromaDB.
        """
        texts: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []

        for child in children:
            texts.append(child.text)
            ids.append(child.id)

            # Create metadata dict - join lists as strings for ChromaDB filtering
            metadata = {
                "parent_id": child.parent_id,
                "section_id": child.section_id,
                "chunk_index": child.chunk_index,
                "keywords": ", ".join(child.keywords),
                "summary": child.summary,
                "tax_category": child.tax_category,
                "entities": ", ".join(child.entities),
                "hypothetical_questions": " | ".join(child.hypothetical_questions),
                "applicable_parties": ", ".join(child.applicable_parties),
            }
            metadatas.append(metadata)

        return texts, metadatas, ids

    def create_vector_db(
        self,
        docstore_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> Any:
        """Create and persist the vector store (ChromaDB or Qdrant).

        Args:
            docstore_path: Path to the enriched documents JSON.
            collection_name: Name for the collection.

        Returns:
            The vector store instance.
        """
        print(f"Loading enriched documents from {docstore_path}...")
        parents = self.load_enriched_documents(docstore_path)
        print(f"Loaded {len(parents)} parent documents")

        print("Creating child chunks...")
        children = self.create_all_children(parents)
        print(f"Created {len(children)} child chunks")

        print("Preparing documents...")
        texts, metadatas, ids = self._prepare_chroma_documents(children)

        if self.use_qdrant:
            return self._create_qdrant_db(texts, metadatas, ids, collection_name)
        else:
            return self._create_chroma_db(texts, metadatas, ids, collection_name)

    def _create_chroma_db(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: List[str],
        collection_name: str,
    ) -> Any:
        """Create ChromaDB vector store (local mode).

        Args:
            texts: List of text chunks.
            metadatas: List of metadata dicts.
            ids: List of document IDs.
            collection_name: Name for the collection.

        Returns:
            The ChromaDB vector store instance.
        """
        from langchain_community.vectorstores import Chroma

        print(f"Creating ChromaDB vector store at {self.persist_dir}...")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
        )

        print(f"Vector store created with {len(texts)} documents")
        print(f"Persisted to {self.persist_dir}")

        return vectorstore

    def _create_qdrant_db(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: List[str],
        collection_name: str,
        force_recreate: bool = True,
    ) -> Any:
        """Create Qdrant vector store (cloud mode).

        Args:
            texts: List of text chunks.
            metadatas: List of metadata dicts.
            ids: List of document IDs.
            collection_name: Name for the collection.
            force_recreate: Whether to recreate if collection exists.

        Returns:
            The Qdrant vector store instance.
        """
        from langchain_qdrant import QdrantVectorStore

        print(f"Creating Qdrant vector store at {self.qdrant_url}...")

        vectorstore = QdrantVectorStore.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            force_recreate=force_recreate,
        )

        print(f"Vector store created with {len(texts)} documents")
        print(f"Uploaded to Qdrant Cloud: {self.qdrant_url}")

        return vectorstore

    def load_vector_db(
        self, collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> Any:
        """Load an existing vector store (ChromaDB or Qdrant).

        Args:
            collection_name: Name of the collection.

        Returns:
            The vector store instance.
        """
        if self.use_qdrant:
            return self._load_qdrant_db(collection_name)
        else:
            return self._load_chroma_db(collection_name)

    def _load_chroma_db(self, collection_name: str) -> Any:
        """Load ChromaDB vector store (local mode).

        Args:
            collection_name: Name of the collection.

        Returns:
            The ChromaDB vector store instance.
        """
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def _load_qdrant_db(self, collection_name: str) -> Any:
        """Load Qdrant vector store (cloud mode).

        Args:
            collection_name: Name of the collection.

        Returns:
            The Qdrant vector store instance.
        """
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
