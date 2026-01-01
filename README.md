# NYC Tax Law RAG System

An Advanced Retrieval-Augmented Generation (RAG) system for NYC Tax Law, built with LangChain, ChromaDB, and hybrid search capabilities.

## Features

- **HyDE Query Expansion**: Generates hypothetical answers for better retrieval
- **Parent-Child Indexing**: Search small chunks, retrieve full legal sections
- **AI Metadata Enrichment**: LLM-generated keywords, summaries, and question mappings
- **Jurisdiction Detection**: Warns when queries are about Federal vs NYC taxes
- **Conversation Memory**: Supports follow-up questions with context
- **ChromaDB Vector Store**: Persistent vector storage for efficient retrieval
- **Type-Safe**: Full Python type hints throughout the codebase

## Requirements

- Python 3.10+
- OpenAI API key

## Project Structure

```
nyc_taxragv2/
├── app/
│   ├── ingest.py        # LegalParser, DocumentEnricher, VectorStoreManager
│   ├── query_engine.py  # QueryProcessor with HyDE
│   └── retriever.py     # Retriever (parent-child document swapper)
├── data/
│   ├── tax_law.txt      # Raw NYC tax law text
│   └── docstore.json    # Enriched parent documents (generated)
├── vectorstore/         # ChromaDB persistence (generated)
├── tests/               # Test suite (70 tests)
├── setup.sh             # Environment setup script
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── README.md            # This file
```

## Architecture

```
                        NYC Tax RAG Pipeline
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INGESTION                                                      │
│  ─────────                                                      │
│  tax_law.txt → LegalParser → 819 Parent Sections                │
│                     ↓                                           │
│              DocumentEnricher (gpt-4o)                          │
│                     ↓                                           │
│              EnrichedDocuments with:                            │
│              • summary, keywords, entities                      │
│              • hypothetical_questions ← improves retrieval      │
│              • tax_category, applicable_parties                 │
│                     ↓                                           │
│              VectorStoreManager                                 │
│                     ↓                                           │
│              Child Chunks (1000 char) → ChromaDB                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  QUERY PROCESSING                                               │
│  ────────────────                                               │
│  User: "Do I pay taxes on Etsy income?"                         │
│                     ↓                                           │
│              QueryProcessor                                     │
│              ├─ analyze_query() → relevance, jurisdiction       │
│              └─ generate_hypothetical_answer() → HyDE           │
│                     ↓                                           │
│              "Under NYC tax law, self-employment income         │
│               from e-commerce activities is subject to..."      │
│                     ↓                                           │
│              Embed hypothetical → Search ChromaDB               │
│                     ↓                                           │
│              Retriever                                          │
│              ├─ Search children with HyDE embedding             │
│              └─ Swap children → Full parent sections            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd nyc_taxragv2
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

3. **Add your data**:
   Place your `tax_law.txt` file in the `data/` directory.

4. **Run the application**:
   ```bash
   source venv/bin/activate
   streamlit run app/main.py
   ```

## Development

### Code Standards

- Python 3.10+ with type hints required
- Follow PEP 8 style guidelines
- All functions must have type annotations

### Running Tests

```bash
source venv/bin/activate
pytest tests/
```

---

## Progress Log

| Task | Status | Date |
|------|--------|------|
| Project Init | Completed | 2026-01-01 |
| Semantic Parser (LegalParser) | Completed | 2026-01-01 |
| AI Metadata Enrichment (DocumentEnricher) | Completed | 2026-01-01 |
| Parent-Child Indexing (VectorStoreManager) | Completed | 2026-01-01 |
| Query Engine with HyDE (QueryProcessor) | Completed | 2026-01-01 |
| Parent-Child Retriever (Retriever) | Completed | 2026-01-01 |

---

## License

MIT License
