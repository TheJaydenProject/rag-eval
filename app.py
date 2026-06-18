import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, "src")

import config
from budget_tracker import BudgetExceededError, get_usage_today
from embedder import build_collection, load_collection
from evaluator import evaluate_answer, generate_answer
from parser import load_documents
from retriever import retrieve_balanced

RAW_DIR = Path("data/raw")
CHROMA_PATH = Path(config.CHROMA_PATH)


def get_or_build_collection():
    records = load_documents(RAW_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not records:
        st.error("No PDFs found in data/raw/. Add at least one PDF and restart.")
        st.stop()

    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        collection = load_collection()
        if collection.count() >= len(records):
            return collection

    with st.spinner(f"Indexing {len(records)} chunks — runs once, then cached..."):
        return build_collection(records)


def render_eval_metrics(ev: dict, lat: dict) -> None:
    with st.expander("Pipeline Metrics & Evaluation"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", f"{ev['faithfulness_score']:.2f}")
        col2.metric("Relevance", f"{ev['relevance_score']:.2f}")

        if lat:
            col3.metric("Total Latency", f"{lat.get('total', 0):.2f}s")

        st.divider()

        if ev.get('is_faithful'):
            st.success(f"Passed: {ev['reasoning']}")
        else:
            st.error(f"Warning: {ev['reasoning']}")

        if lat:
            st.caption(f"Retrieval: {lat.get('retrieval', 0):.2f}s | Generation: {lat.get('generation', 0):.2f}s | Eval: {lat.get('eval', 0):.2f}s")


st.set_page_config(page_title="rag-eval", page_icon="📄")
st.title("rag-eval")
st.caption(f"Provider: `{config.PROVIDER}` | Generation: `{config.GENERATION_MODEL}` | Embedding: `{config.EMBEDDING_MODEL}`")

if "collection" not in st.session_state:
    st.session_state.collection = get_or_build_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "eval" in msg:
            render_eval_metrics(msg["eval"], msg.get("latency", {}))

if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                t0 = time.time()
                chunks = retrieve_balanced(query, st.session_state.collection, config.TOP_K)
                t1 = time.time()

                answer = generate_answer(query, chunks)
                t2 = time.time()

                scores = evaluate_answer(query, answer, chunks)
                t3 = time.time()
            except BudgetExceededError as e:
                st.error(str(e))
                st.stop()

        latencies = {
            "total": t3 - t0,
            "retrieval": t1 - t0,
            "generation": t2 - t1,
            "eval": t3 - t2,
        }

        st.markdown(answer)
        render_eval_metrics(scores, latencies)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "eval": scores,
        "latency": latencies,
    })

with st.sidebar:
    st.header("Status")
    used = get_usage_today()
    st.metric("Tokens used today", f"{used:,}")
    st.progress(min(used / config.DAILY_TOKEN_BUDGET, 1.0))
    st.caption(f"Daily budget: {config.DAILY_TOKEN_BUDGET:,}")

    st.divider()
    st.header("Indexed documents")
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if pdfs:
        for pdf in pdfs:
            st.text(pdf.name)
    else:
        st.caption("None — add PDFs to data/raw/")
