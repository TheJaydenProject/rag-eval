import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import config
from parser import load_documents
from embedder import build_collection, load_collection
from retriever import retrieve_balanced
from evaluator import generate_answer, evaluate_answer

RAW_DIR = Path("data/raw")
CHROMA_PATH = Path(config.CHROMA_PATH)
LOG_PATH = Path("eval_results.json")


def get_or_build_collection():
    # If chroma_store exists and has content, skip re-embedding.
    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        print("ChromaDB store found — loading existing collection.")
        collection = load_collection()
        if collection.count() > 0:
            return collection
        print("Collection is empty — rebuilding.")

    print("Building ChromaDB collection from PDFs...")
    records = load_documents(RAW_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    if not records:
        print("No PDFs found in data/raw/. Add at least one PDF and rerun.")
        sys.exit(1)

    return build_collection(records)


def run_query(query: str, collection) -> dict:
    chunks = retrieve_balanced(query, collection, config.TOP_K)
    answer = generate_answer(query, chunks)
    scores = evaluate_answer(query, answer, chunks)

    return {
        "query": query,
        "answer": answer,
        "sources": [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
        "evaluation": scores,
    }


def main() -> None:
    print(f"Provider: {config.PROVIDER} | Embedding: {config.EMBEDDING_MODEL} | Generation: {config.GENERATION_MODEL}\n")

    collection = get_or_build_collection()

    queries: list[str] = [
        "What is the main topic of the document?",
        "What are the key findings or conclusions?",
        "What methods or approaches are described?",
        "What recommendations are made?",
        "What limitations or risks are identified?",
    ]

    all_results: list[dict] = []
    for query in queries:
        print(f"\nQuery: {query}")
        result = run_query(query, collection)
        print(f"Answer: {result['answer']}")
        print(f"Faithfulness: {result['evaluation']['faithfulness_score']:.2f} | Relevance: {result['evaluation']['relevance_score']:.2f}")
        all_results.append(result)

    LOG_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nEval results saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
