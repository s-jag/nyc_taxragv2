# How the NYC Tax RAG System Works

## Abstract

RAG systems for legal documents face a fundamental tradeoff: small chunks enable precise retrieval but lose context, while large chunks preserve context but reduce retrieval accuracy.

This system solves this with a **Parent-Child architecture**: we search small chunks but return full legal sections. Combined with HyDE query expansion and a confidence-based fallback, we achieve high-precision retrieval for professional tax preparers.

Key contributions:
- **Parent-Child Indexing**: Search 1000-char chunks, retrieve full § sections
- **HyDE Query Expansion**: Generate hypothetical legal answers to improve semantic matching
- **ReRanker Guardrail**: LLM-based relevance filtering to prevent hallucination
- **Confidence-based Fallback**: Route low-confidence queries to a general-purpose model with appropriate disclaimers

## The Problem with Legal RAG

Legal documents have a unique structure. NYC Tax Law is organized into sections (§ 11-121, § 11-122, etc.), and each section is a self-contained unit of law. When a user asks "What are property tax rates?", we need to return the *entire* relevant section, not a fragment.

Naive chunking breaks this:

```
§ 11-121 Property Tax Assessment.
The commissioner shall assess all real property...
[800 more characters]
...and shall file such assessment with the clerk.

§ 11-122 Collection Procedures.
The city collector shall collect all taxes...
```

=>

### Chunk 1 (1000 chars)
```
§ 11-121 Property Tax Assessment.
The commissioner shall assess all real property...
[truncated mid-sentence]
```

### Chunk 2 (1000 chars)
```
[continuation from previous section]
...and shall file such assessment with the clerk.

§ 11-122 Collection Procedures.
The city collector shall collect all taxes...
[truncated]
```

If the user's query matches Chunk 2, they get a fragment starting mid-sentence, plus the beginning of an unrelated section. Useless for a tax preparer who needs the complete legal provision.

## Parent-Child Architecture

Our solution: **index the children, return the parents**.

```
                    INGESTION

§ 11-121 (Parent)                    § 11-122 (Parent)
Full section text                    Full section text
      |                                    |
      v                                    v
  ┌───────┬───────┬───────┐          ┌───────┬───────┐
  │Child 1│Child 2│Child 3│          │Child 1│Child 2│
  │1000ch │1000ch │1000ch │          │1000ch │1000ch │
  └───────┴───────┴───────┘          └───────┴───────┘
      |       |       |                  |       |
      v       v       v                  v       v
  ┌─────────────────────────────────────────────────┐
  │              ChromaDB Vector Store              │
  │         (5430 child chunks indexed)             │
  └─────────────────────────────────────────────────┘
```

At query time:

```
User: "What are property tax rates?"
                |
                v
        Search ChromaDB
                |
                v
    Match: Child 2 of § 11-121
                |
                v
    Lookup parent_id in docstore
                |
                v
    Return: Full § 11-121 section
```

This gives us the best of both worlds:
- **Precise retrieval** via small chunk embeddings
- **Complete context** via parent section return

The implementation is simple. Each child chunk stores its `parent_id`:

```python
child = {
    "id": "child-11-121-2",
    "parent_id": "11-121",
    "text": "...chunk content...",
}
```

After vector search, we swap children for parents:

```python
def search(query, k=5):
    child_results = vectorstore.similarity_search(query, k=k*3)

    # Deduplicate by parent
    seen_parents = set()
    parents = []
    for child in child_results:
        parent_id = child.metadata["parent_id"]
        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parents.append(docstore[parent_id])

    return parents[:k]
```

## HyDE Query Expansion

User queries are short and colloquial. Legal text is dense and formal. The embedding similarity between them is poor.

```
User query: "Do I pay taxes on Etsy income?"

Legal text: "§ 11-503 Imposition of tax on unincorporated
business conducted within the city. A tax is hereby imposed
for each taxable year on the unincorporated business taxable
income of every unincorporated business wholly or partly
carried on within the city..."
```

These have low semantic similarity despite being directly relevant.

**HyDE** (Hypothetical Document Embeddings) solves this. We ask an LLM to generate what a *legal answer* to the user's question might look like, then use *that* for retrieval:

```
User: "Do I pay taxes on Etsy income?"
        |
        v
    LLM generates hypothetical answer:
        |
        v
"Under NYC tax law, income from online marketplace
activities such as Etsy would typically be considered
unincorporated business income subject to the
Unincorporated Business Tax (UBT) pursuant to § 11-503,
provided the business is conducted within the city..."
        |
        v
    Embed THIS for retrieval
```

Now the query embedding is in the same semantic space as the legal corpus. Retrieval accuracy improves dramatically.

## ReRanker Guardrail

Vector similarity is not semantic relevance. A document about "property tax *exemptions*" might embed similarly to a query about "property tax *rates*", but citing the wrong section is worse than citing nothing.

The ReRanker is an LLM-based filter that grades each retrieved document:

```
Query: "What are property tax rates?"

Document § 11-243.1:
"Tax abatement for storm-damaged properties..."

ReRanker prompt:
"Grade this document's relevance to the query (0-10):
- 10: Directly answers the question
- 7-9: Highly relevant, same topic
- 4-6: Related but not applicable
- 0-3: Different topic"

Response: {"score": 3.0, "reasoning": "Discusses abatements, not rates"}
```

Documents below the threshold (default 7.0) are filtered out. If *all* documents fail, we return a "silence" response rather than hallucinate:

```python
def grade_documents(query, documents):
    graded = [grade_single(query, doc) for doc in documents]
    passed = [g for g in graded if g.score >= threshold]

    if not passed:
        return None  # Signals retrieval failure

    return sorted(passed, key=lambda x: x.score, reverse=True)
```

This is toggleable for A/B testing.

## Confidence-based Fallback

What happens when documents pass the ReRanker threshold (7.0) but aren't highly confident (< 8.0)?

We route to a **fallback model** that provides general guidance without authoritative citations:

```
                        Query
                          |
                          v
                    ┌──────────┐
                    │ Retriever│
                    └────┬─────┘
                         |
                         v
                    ┌──────────┐
                    │ ReRanker │
                    └────┬─────┘
                         |
            ┌────────────┼────────────┐
            |            |            |
     best >= 8.0    7.0 <= best    best < 7.0
            |         < 8.0           |
            v            |            v
     ┌──────────┐        |     ┌──────────┐
     │ o3-mini  │        |     │ SILENCE  │
     │ + cites  │        v     │ response │
     └──────────┘  ┌──────────┐└──────────┘
                   │GPT-5.2pro│
                   │ general  │
                   │ guidance │
                   └──────────┘
```

The fallback response includes a clear disclaimer:

```
**General Guidance Notice**

This response provides general tax guidance based on common
knowledge. The NYC Tax Law database did not contain highly
relevant sections for your specific question.

The sections shown below may be tangentially related but are
not directly applicable.

---

[General answer without authoritative citations]

Potentially Related Sections:
- § 11-121 (7.3/10) - May be tangentially related
```

This prevents the failure mode where users receive confidently-stated but unreliable information.

## Full Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  User: "What are property tax rates?"                        │
└──────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: HyDE Query Expansion                                │
│                                                              │
│  LLM generates hypothetical legal answer                     │
│  "Under NYC law, property tax rates are determined..."       │
└──────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Parent-Child Retrieval                              │
│                                                              │
│  Search 5430 child chunks → Return 5 parent sections         │
└──────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: ReRanker Filtering                                  │
│                                                              │
│  Grade each section 0-10, filter below 7.0                   │
│  § 11-121: 7.5/10 ✓                                          │
│  § 11-243: 3.0/10 ✗                                          │
└──────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: Confidence Routing                                  │
│                                                              │
│  best_score = 7.5 < confidence_threshold (8.0)               │
│  → Route to fallback model                                   │
└──────────────────────────────────────────────────────────────┘
                              |
                              v
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: Response Generation                                 │
│                                                              │
│  GPT-5.2-pro generates general guidance                      │
│  + disclaimer + "Potentially Related Sections"               │
└──────────────────────────────────────────────────────────────┘
```

## Benchmarks

| Component | Count |
|-----------|-------|
| Parent sections (§) | 819 |
| Child chunks | 5,430 |
| Avg children per parent | 6.6 |
| Child chunk size | 1000 chars |

| Setting | Default |
|---------|---------|
| ReRanker threshold | 7.0 |
| Confidence threshold | 8.0 |
| Primary model | o3-mini |
| Fallback model | gpt-5.2-pro |
| Retrieval k | 5 |

## Summary

The NYC Tax RAG system addresses the core challenges of legal document retrieval:

1. **Context preservation**: Parent-child indexing searches small chunks but returns complete sections
2. **Query-corpus mismatch**: HyDE generates hypothetical legal answers for better embedding similarity
3. **Precision over recall**: ReRanker filters irrelevant documents to prevent hallucination
4. **Graceful degradation**: Confidence-based fallback provides general guidance with clear disclaimers

The result is a system that professional tax preparers can trust: authoritative when confident, transparent when not.
