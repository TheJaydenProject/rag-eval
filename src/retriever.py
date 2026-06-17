import chromadb

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from llm_client import get_query_embedding


def retrieve(query: str, collection: chromadb.Collection, top_k: int) -> list[dict]:
    if collection.count() == 0:
        raise ValueError("Collection is empty. Run embedder.build_collection() first.")

    query_vec: list[float] = get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            # ChromaDB returns L2 distance by default — lower means more similar.
            "distance": dist,
        })

    return chunks


def retrieve_balanced(query: str, collection: chromadb.Collection, top_k: int) -> list[dict]:
    """
    For cross-document queries: retrieves top_k chunks per unique source file,
    then returns all of them sorted by distance ascending.

    Prevents a single high-scoring document from monopolising all TOP_K slots.
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
