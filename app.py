import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, "src")

import config
from budget_tracker import BudgetExceededError, get_usage_today
from embedder import build_collection, load_collection
from evaluator import evaluate_answer, generate_answer
from parser import load_documents
from retriever import retrieve

RAW_DIR = Path("data/raw")
CHROMA_PATH = Path(config.CHROMA_PATH)


def get_or_build_collection():
    if CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()):
        collection = load_collection()
        if collection.count() > 0:
            return collection

    records = load_documents(RAW_DIR, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not records:
        st.error("No PDFs found in data/raw/. Add at least one PDF and restart.")
        st.stop()

    with st.spinner(f"Indexing {len(records)} chunks — runs once, then cached..."):
        return build_collection(records)


st.set_page_config(page_title="rag-eval", page_icon="📄")
st.title("rag-eval")
st.caption(f"Provider: `{config.PROVIDER}` | Generation: `{config.GENERATION_MODEL}` | Embedding: `{config.EMBEDDING_MODEL}`")

# Load collection once per session.
if "collection" not in st.session_state:
    st.session_state.collection = get_or_build_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "eval" in msg:
            ev = msg["eval"]
            with st.expander("Eval scores"):
                col1, col2 = st.columns(2)
                col1.metric("Faithfulness", f"{ev['faithfulness_score']:.2f}")
                col2.metric("Relevance", f"{ev['relevance_score']:.2f}")
                st.caption(ev["reasoning"])

# Handle new input.
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                chunks = retrieve(query, st.session_state.collection, config.TOP_K)
                answer = generate_answer(query, chunks)
                scores = evaluate_answer(query, answer, chunks)
            except BudgetExceededError as e:
                st.error(str(e))
                st.stop()

        st.markdown(answer)
        with st.expander("Eval scores"):
            col1, col2 = st.columns(2)
            col1.metric("Faithfulness", f"{scores['faithfulness_score']:.2f}")
            col2.metric("Relevance", f"{scores['relevance_score']:.2f}")
            st.caption(scores["reasoning"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "eval": scores,
    })

# Sidebar: token usage and source list.
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
