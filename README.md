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
| **TaxAdvisor (o3-mini)** | Reasoning-based answer generation with strict citations |

## Quick Start

```bash
# Setup
git clone <repository-url> && cd nyc_taxragv2
./setup.sh

# Configure
cp .env.example .env  # Add your OPENAI_API_KEY

# Run the UI
source venv/bin/activate
streamlit run app/main_ui.py
```

## UI Features

The Streamlit interface provides:

| Feature | Description |
|---------|-------------|
| **System Internals Sidebar** | View HyDE-expanded queries, retrieved sections with scores, and pipeline stats |
| **Re-ranker Toggle** | Enable/disable re-ranking for A/B testing |
| **Threshold Slider** | Adjust minimum relevance score (0-10) |
| **Re-build Database** | One-click database rebuild from source files |
| **Silence Handling** | Yellow warning box when no relevant docs found (prevents hallucination) |

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
│  TaxAdvisor (o3-mini) ──→ Cited answer with reasoning           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
app/
├── main_ui.py       # Streamlit web interface
├── ingest.py        # LegalParser, DocumentEnricher, VectorStoreManager
├── query_engine.py  # QueryProcessor (HyDE, conversation memory)
├── retriever.py     # Retriever (parent-child document swapper)
├── ranker.py        # ReRanker (optional relevance filtering)
└── generator.py     # TaxAdvisor (full pipeline orchestration)

tests/               # 106 tests
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

### 7. TaxAdvisor (Full Pipeline)
Orchestrates the complete RAG pipeline with o3-mini reasoning model. Designed for professional tax preparers with strict citation requirements.

```python
advisor = TaxAdvisor(
    vectorstore=vectorstore,
    docstore_path="data/docstore.json",
    generation_model="o3-mini",
    reranker_enabled=True,
)

response = advisor.answer_question("What are property tax rates?")
# Returns: TaxAdvisorResponse with:
#   - answer: Full response with § citations
#   - sources: List of cited sections
#   - warnings: Jurisdiction/relevance warnings
#   - debug_info: Pipeline diagnostics
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
| TaxAdvisor | Done |

## License

MIT
