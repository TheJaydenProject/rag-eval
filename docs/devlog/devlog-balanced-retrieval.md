# Devlog: Balanced Retrieval for Multi-Source RAG Queries

**Project:** rag-eval
**Date:** 17 June 2026
**Environment:** Windows 11, Python 3.12, ChromaDB 1.5.9, gemini-embedding-001
**Files changed:** `src/retriever.py`, `app.py`, `main.py`

---

## Background

rag-eval is a document Q&A pipeline. PDFs are split into fixed-size text chunks
(`CHUNK_SIZE=1500`, `CHUNK_OVERLAP=100`), embedded via Gemini's embedding API, and
stored in a local ChromaDB vector store. At query time, the user's question is embedded
using the same model, and ChromaDB returns the top-K most similar chunks by L2 distance.
Those chunks are injected into a generation prompt, and an LLM produces an answer
grounded strictly in the retrieved context. If the answer cannot be found in the
provided chunks, the model is instructed to return "I don't know" rather than
hallucinate.

The corpus for this evaluation run contained three documents:

- NVIDIA Code of Conduct (~130 pages)
- Apple Business Conduct Policy (~30 pages)
- Microsoft 2023 Annual Report (10-K, several hundred pages)

---

## Problem

Cross-document comparison queries returned "I don't know" despite both source documents
being indexed correctly in ChromaDB.

**Query (Q2 in the eval set):**

> "Compare how NVIDIA and Apple each define and handle employee conflicts of interest."

**Expected:** A comparative answer synthesised from relevant chunks across both the
NVIDIA and Apple documents.

**Actual response from the system:**

> "I don't know. The context provides separate policies for Apple and NVIDIA, but it
> does not contain a direct comparison or analysis of how each defines and handles
> employee conflicts of interest."

**Eval scores:** Faithfulness 1.00, Relevance 0.80.

The high faithfulness score is the key signal here. The model did not hallucinate. It
correctly declined because the context window passed to the generation prompt only
contained chunks from one company. The system worked as designed — the issue was
upstream, in what the retriever sent to the LLM.

---

## Root Cause

`retrieve()` in `src/retriever.py` issues a single global query to ChromaDB and returns
the top-K chunks by L2 distance, with no constraint on which source documents those
chunks come from:

```python
results = collection.query(
    query_embeddings=[query_vec],
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
)
```

With `TOP_K=4`, there are only four slots available in the context window. Inspecting
the `source` field on each returned chunk confirmed that all four were from
`NVIDIA-Code-of-Conduct-external.pdf`. Apple's document had zero representation.

This is not a bug. `retrieve()` behaved exactly as implemented. The problem is
structural: document size directly determines how many chunks a source contributes to
the collection. NVIDIA's document is roughly four times the length of Apple's, so it
produces roughly four times as many chunks. More chunks means a statistically higher
chance that NVIDIA's content occupies the top positions in L2 space for any query
touching topics both documents cover, such as conflicts of interest. With only four
retrieval slots and a heavily skewed chunk distribution, NVIDIA filled all of them.

The LLM never saw Apple's content. Its "I don't know" was the correct and faithful
response given what it received.

---

## Fix

Added `retrieve_balanced()` to `src/retriever.py` as a second retrieval function. The
original `retrieve()` was not modified or removed, as it remains appropriate for
single-document queries and corpora where document sizes are roughly equal.

`retrieve_balanced()` queries ChromaDB once per unique source file using ChromaDB's
`where` metadata filter. It retrieves `top_k` chunks from each source independently,
then merges and re-sorts the full result set by distance ascending before returning.
This guarantees every indexed document gets representation in the context window,
regardless of how many chunks it contributed to the collection.

```python
def retrieve_balanced(query: str, collection: chromadb.Collection, top_k: int) -> list[dict]:
    """
    Retrieves top_k chunks per unique source file, then merges and sorts by
    distance ascending.

    Prevents a single large document from monopolising all TOP_K retrieval slots
    when the query spans multiple sources of unequal size.
    """
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run embedder.build_collection() first.")

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources: list[str] = list({m["source"] for m in all_meta})

    query_vec: list[float] = get_query_embedding(query)
    chunks: list[dict] = []

    for source in sources:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where={"source": source},
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "distance": dist,
            })

    return sorted(chunks, key=lambda c: c["distance"])
```

The `where={"source": source}` filter is a built-in ChromaDB metadata filter. Each
source file is stored in the collection with a `source` metadata field set to its
filename at embed time (handled in `src/embedder.py`). Filtering by this field scopes
each query to one document's chunks, which is what makes per-source capping possible.

`app.py` and `main.py` were updated to call `retrieve_balanced` in place of `retrieve`.
The call signature is identical, so no other changes were needed.

---

## Result

Re-running Q2 after the fix returned a comparative answer drawing on chunks from both
the NVIDIA and Apple documents. Faithfulness and Relevance scores both returned at or
above 0.90.

---

## Known Tradeoff

`retrieve_balanced()` issues N queries to ChromaDB (one per source document), where N
scales with the number of documents in the corpus. For this project's current scale of
three documents against a local ChromaDB instance, the overhead is negligible. For a
corpus with hundreds of documents, the N-query pattern could become a bottleneck and
would need a different approach (for example, querying globally and post-filtering, or
chunking documents to equalise their contribution at ingest time). That is not a problem
this project has today, and the implementation should not be pre-optimised for it.

---

## When to Use Each Function

| Scenario | Function |
|---|---|
| Single-document corpus, or all documents are similar in length and topic density | `retrieve()` |
| Multi-document corpus with unequal document sizes, or query explicitly spans multiple sources | `retrieve_balanced()` |

---

## Change Summary

| File | Change |
|---|---|
| `src/retriever.py` | Added `retrieve_balanced()` alongside existing `retrieve()` |
| `app.py` | Swapped `retrieve` call to `retrieve_balanced` |
| `main.py` | Swapped `retrieve` call to `retrieve_balanced` |
