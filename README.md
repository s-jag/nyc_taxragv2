# NYC Tax Law RAG System

An Advanced RAG system for NYC Tax Law with HyDE query expansion, parent-child document retrieval, and optional re-ranking guardrails.

## Features

| Feature | Description |
|---------|-------------|
| **HyDE Query Expansion** | Generates hypothetical legal answers for better semantic matching |
| **Parent-Child Indexing** | Search small chunks, retrieve full legal sections |
| **AI Metadata Enrichment** | LLM-generated summaries, keywords, and question mappings |
| **Re-ranking Guardrail** | Optional LLM-based relevance filtering (toggleable) |
| **Jurisdiction Detection** | Warns when queries are about Federal vs NYC taxes |

## Quick Start

```bash
# Setup
git clone <repository-url> && cd nyc_taxragv2
./setup.sh

# Configure
cp .env.example .env  # Add your OPENAI_API_KEY

# Run
source venv/bin/activate
streamlit run app/main.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                             │
│                                                                 │
│  tax_law.txt                                                    │
│       ↓                                                         │
│  LegalParser ──→ 819 Parent Sections (by § symbol)              │
│       ↓                                                         │
│  DocumentEnricher ──→ AI metadata (summary, keywords, entities) │
│       ↓                                                         │
│  VectorStoreManager ──→ Child chunks (1000 char) → ChromaDB     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  QUERY PIPELINE                                                 │
│                                                                 │
│  User Query: "Do I pay taxes on Etsy income?"                   │
│       ↓                                                         │
│  QueryProcessor                                                 │
│  ├─ analyze_query() → relevance check, jurisdiction detection   │
│  └─ generate_hypothetical_answer() → HyDE expansion             │
│       ↓                                                         │
│  Retriever ──→ Search children, return full parent sections     │
│       ↓                                                         │
│  ReRanker (optional) ──→ Filter irrelevant docs, or return None │
│       ↓                                                         │
│  Answer Generation                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
app/
├── ingest.py        # LegalParser, DocumentEnricher, VectorStoreManager
├── query_engine.py  # QueryProcessor (HyDE, conversation memory)
├── retriever.py     # Retriever (parent-child document swapper)
└── ranker.py        # ReRanker (optional relevance filtering)

tests/               # 80+ tests
data/
├── tax_law.txt      # Raw NYC tax law text
└── docstore.json    # Enriched parent documents (generated)
```

## Core Components

### 1. LegalParser
Parses raw tax law text by § symbol into structured sections.

### 2. DocumentEnricher
Uses GPT-4o to generate metadata for each section:
- Summary, keywords, entities
- Hypothetical questions (improves retrieval)
- Tax category, applicable parties

### 3. VectorStoreManager
Splits parents into 1000-char child chunks with metadata inheritance. Stores in ChromaDB.

### 4. QueryProcessor (HyDE)
Expands user queries into hypothetical legal answers for better embedding similarity.

### 5. Retriever
Searches child chunks, returns full parent documents for complete context.

### 6. ReRanker (Optional)
Grades each document 0-10 for relevance. Filters docs below threshold. Returns `None` if no relevant docs found (prevents hallucination).

```python
# Toggle for A/B testing
reranker = ReRanker(enabled=True, threshold=7.0)
result = reranker.grade_documents(query, documents)
if result is None:
    return "I couldn't find relevant information."
```

## Development

```bash
# Run tests
pytest tests/ -v

# Code standards
# - Python 3.10+ with type hints
# - PEP 8 style
```

## Progress

| Component | Status |
|-----------|--------|
| LegalParser | Done |
| DocumentEnricher | Done |
| VectorStoreManager | Done |
| QueryProcessor (HyDE) | Done |
| Retriever | Done |
| ReRanker | Done |

## License

MIT
