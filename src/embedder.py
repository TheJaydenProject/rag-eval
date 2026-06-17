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

    for i, record in enumerate(records):
        vector: list[float] = get_embedding(record["text"])

        collection.upsert(
            ids=[f"{record['source']}__chunk_{record['chunk_index']}"],
            embeddings=[vector],
            documents=[record["text"]],
            metadatas=[{"source": record["source"], "chunk_index": record["chunk_index"]}],
        )

        # Free tier rate limit buffer — avoids burst rejections on large corpora.
        if i % 50 == 0 and i > 0:
            time.sleep(1)

        if i % 10 == 0:
            print(f"  Embedded {i}/{len(records)} chunks...")

    print(f"Collection '{config.COLLECTION_NAME}' ready — {len(records)} chunks indexed.")
    return collection


def load_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return client.get_or_create_collection(name=config.COLLECTION_NAME)
