import os
import sys
import time
from pathlib import Path

import chromadb

sys.path.insert(0, os.path.dirname(__file__))
import config
from llm_client import get_embedding


def build_collection(records: list[dict]) -> chromadb.Collection:
    if not records:
        raise ValueError("No records to embed. Check that PDFs are in data/raw/.")

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    collection = client.get_or_create_collection(name=config.COLLECTION_NAME)

    existing_ids = set(collection.get(include=[])["ids"])

    pending = [
        r
        for r in records
        if f"{r['source']}__chunk_{r['chunk_index']}" not in existing_ids
    ]

    if not pending:
        print(
            f"Collection '{config.COLLECTION_NAME}' already complete — {len(records)} chunks."
        )
        return collection

    print(f"Resuming: {len(existing_ids)} already embedded, {len(pending)} remaining.")

    for i, record in enumerate(pending):
        vector: list[float] = get_embedding(record["text"])

        collection.upsert(
            ids=[f"{record['source']}__chunk_{record['chunk_index']}"],
            embeddings=[vector],
            documents=[record["text"]],
            metadatas=[
                {"source": record["source"], "chunk_index": record["chunk_index"]}
            ],
        )

        # Throttle to stay under the embedding provider's rate limit. Gemini's
        # free tier (100 RPM) needs ~0.7s/request; OpenAI embeddings set this to
        # 0.0. Controlled by config.EMBEDDING_RATE_LIMIT_DELAY.
        if config.EMBEDDING_RATE_LIMIT_DELAY > 0:
            time.sleep(config.EMBEDDING_RATE_LIMIT_DELAY)

        if i % 10 == 0:
            print(f"  Embedded {len(existing_ids) + i}/{len(records)} chunks...")

    print(
        f"Collection '{config.COLLECTION_NAME}' ready — {len(records)} chunks indexed."
    )
    return collection


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(name=config.COLLECTION_NAME)
