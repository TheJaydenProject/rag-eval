import time
from pathlib import Path

import chromadb

import sys, os
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
        r for r in records
        if f"{r['source']}__chunk_{r['chunk_index']}" not in existing_ids
    ]

    if not pending:
        print(f"Collection '{config.COLLECTION_NAME}' already complete — {len(records)} chunks.")
        return collection

    print(f"Resuming: {len(existing_ids)} already embedded, {len(pending)} remaining.")

    for i, record in enumerate(pending):
        vector: list[float] = get_embedding(record["text"])

        collection.upsert(
            ids=[f"{record['source']}__chunk_{record['chunk_index']}"],
            embeddings=[vector],
            documents=[record["text"]],
            metadatas=[{"source": record["source"], "chunk_index": record["chunk_index"]}],
        )

        # 100 RPM free tier = 1 req/0.6s; 0.7s keeps us safely under across 800-1000 chunk builds.
        time.sleep(0.7)

        if i % 10 == 0:
            print(f"  Embedded {len(existing_ids) + i}/{len(records)} chunks...")

    print(f"Collection '{config.COLLECTION_NAME}' ready — {len(records)} chunks indexed.")
    return collection


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(name=config.COLLECTION_NAME)
