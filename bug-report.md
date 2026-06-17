# RAG-Eval: Embedding Setup Bug Report

**Project:** rag-eval  
**Date:** 17 June 2026  
**Environment:** Windows 11, Python 3.x, `google-genai==2.8.0`, Gemini free tier + DeepSeek  
**Files affected:** `src/config.py`, `src/llm_client.py`, `src/embedder.py`

---

## Overview

Three sequential bugs were encountered while configuring Gemini free tier embeddings for a RAG pipeline that indexes PDFs into ChromaDB. All three errors surfaced during the initial collection build — before any queries could be run. Each had a distinct root cause and required a targeted fix.

The bugs are documented in the order they appeared. Resolving one revealed the next.

---

## Bug 1: Embedding Model Not Available on Free Tier

**Symptom**

The app crashed immediately on startup when attempting to build the ChromaDB collection:

```
google.genai.errors.ClientError: 404 NOT_FOUND.
'models/text-embedding-004 is not found for API version v1beta,
or is not supported for embedContent.'
```

**What was expected**

`text-embedding-004` would be called via `_gemini_client.models.embed_content()` and return a vector for each document chunk.

**What actually happened**

The Gemini API returned a 404. `text-embedding-004` is not available on the free tier.

**Root cause**

`config.py` had `text-embedding-004` hardcoded as the embedding model for both the `gemini` and `deepseek` providers. This model requires a paid Gemini API plan.

```python
# src/config.py — before fix
EMBEDDING_MODELS: dict[str, str] = {
    "deepseek": "text-embedding-004",
    "gemini": "text-embedding-004",
    "openai": "text-embedding-3-small",
}
```

**Diagnosis**

Listed available models against the actual API key:

```python
for m in client.models.list():
    if "embed" in m.name.lower():
        print(m.name)
```

Output confirmed only two embedding models were accessible on this key: `gemini-embedding-001` and `gemini-embedding-2`. `text-embedding-004` was not present.

`gemini-embedding-2` was skipped as it had already failed in an earlier attempt. `gemini-embedding-001` was selected.

**Fix**

Updated `config.py` to use the correct model:

```python
# src/config.py — after fix
EMBEDDING_MODELS: dict[str, str] = {
    "deepseek": "gemini-embedding-001",
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}
```

The existing `chroma_store/` directory was deleted before re-running. Vectors built under the wrong model are dimensionally incompatible and would produce silently incorrect retrieval results if left in place:

```powershell
Remove-Item -Recurse -Force chroma_store
```

---

## Bug 2: SDK Routing Requests to Deprecated API Version

**Symptom**

After fixing the model name, a 404 persisted with a slightly different message:

```
RuntimeError: Embedding request failed. Verify gemini-embedding-001 is active.
Details: 404 NOT_FOUND.
'models/text-embedding-004 is not found for API version v1beta,
or is not supported for embedContent.'
```

**What was expected**

With the correct model name set, embedding calls would succeed.

**What actually happened**

The 404 continued. The error message still referenced `v1beta`, indicating the SDK was hitting the wrong API endpoint regardless of the model name.

**Root cause**

`google-genai==2.8.0` defaults to the `v1beta` endpoint for `embed_content` calls. `gemini-embedding-001` only exists under `v1`. The client was initialised without an explicit API version:

```python
# src/llm_client.py — before fix
_gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
```

**Fix**

Forced `v1` via `HttpOptions` at client initialisation:

```python
# src/llm_client.py — after fix
from google.genai.types import HttpOptions

_gemini_client = genai.Client(
    api_key=config.GEMINI_API_KEY,
    http_options=HttpOptions(api_version="v1"),
)
```

Confirmed resolved when the next error no longer mentioned `v1beta`.

---

## Bug 3: Free Tier Rate Limit Exceeded During Bulk Indexing

**Symptom**

Embedding calls began succeeding but the process crashed partway through collection build:

```
RuntimeError: Embedding request failed. Verify gemini-embedding-001 is active.
Details: 429 RESOURCE_EXHAUSTED.
'Quota exceeded for metric: embed_content_free_tier_requests,
limit: 100, model: gemini-embedding-1.0. Please retry in 42.571344403s.'
```

**What was expected**

The existing rate limit buffer in `embedder.py` would keep requests within the free tier quota.

**What actually happened**

The pipeline exceeded the 100 requests-per-minute cap and was rejected mid-build.

**Root cause**

The free tier enforces a hard limit of 100 RPM for `gemini-embedding-001`. The corpus being indexed consisted of three PDFs:

| Document | Pages |
|---|---|
| Microsoft 2023 Annual Report (10-K) | ~16 |
| Apple Business Conduct Policy | ~30 |
| NVIDIA Code of Conduct | ~130 |

At `CHUNK_SIZE=500` with `CHUNK_OVERLAP=50`, these produced approximately 800-1000 chunks. The existing rate limit buffer only paused once every 50 chunks:

```python
# src/embedder.py — before fix
if i % 50 == 0 and i > 0:
    time.sleep(1)
```

This allowed bursts of 50 rapid back-to-back requests before each 1-second pause — easily saturating the 100 RPM cap.

**Fix**

Two changes applied together:

Increased chunk size in `config.py` to reduce total chunk count by roughly 3x, lowering the total number of embedding calls required:

```python
# src/config.py — after fix
CHUNK_SIZE: int = 1500   # was 500
CHUNK_OVERLAP: int = 100  # was 50
```

Replaced the burst-based sleep with a steady per-request delay in `embedder.py`. At 0.7 seconds per request, throughput sits at ~85 RPM — safely under the 100 RPM cap with headroom for network variance:

```python
# src/embedder.py — after fix
for i, record in enumerate(records):
    vector: list[float] = get_embedding(record["text"])

    collection.upsert(
        ids=[f"{record['source']}__chunk_{record['chunk_index']}"],
        embeddings=[vector],
        documents=[record["text"]],
        metadatas=[{"source": record["source"], "chunk_index": record["chunk_index"]}],
    )

    # 100 RPM free tier = 1 request per 0.6s; 0.7s provides headroom
    time.sleep(0.7)

    if i % 10 == 0:
        print(f"  Embedded {i}/{len(records)} chunks...")
```

`chroma_store/` was deleted and rebuilt after this change, as the chunk boundaries changed with the new `CHUNK_SIZE`.

---

## Additional Note: Conflicting Environment Variables

During diagnosis, the SDK printed a warning:

```
Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
```

This did not affect the running app — `llm_client.py` passes the key explicitly via `genai.Client(api_key=config.GEMINI_API_KEY)`, bypassing the SDK's ambient key resolution entirely. However, the redundant `GOOGLE_API_KEY` entry in `.env` is worth removing to prevent confusion if the explicit key is ever removed in future.

---

## Change Summary

| File | Change |
|---|---|
| `src/config.py` | `EMBEDDING_MODEL` → `gemini-embedding-001`; `CHUNK_SIZE` → 1500; `CHUNK_OVERLAP` → 100 |
| `src/llm_client.py` | Gemini client initialised with `HttpOptions(api_version="v1")` |
| `src/embedder.py` | Per-request `time.sleep(0.7)` replacing burst-based sleep every 50 chunks |
| `chroma_store/` | Deleted and rebuilt after Bug 1 fix and again after Bug 3 fix |
