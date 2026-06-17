# rag-eval

RAG pipeline that ingests PDF documents, retrieves grounded context, and scores every generated answer for faithfulness and relevance using an LLM-as-judge.

[![Tests](https://github.com/TheJaydenProject/rag-eval/actions/workflows/test.yml/badge.svg)](https://github.com/TheJaydenProject/rag-eval/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.58.0-red)
![ChromaDB](https://img.shields.io/badge/chromadb-1.5.9-orange)

---

<!-- Add a screen recording or GIF of the Streamlit UI here. Replace the line below with: ![Demo](docs/demo.gif) -->
> **Demo**: Add a screen recording or GIF showing a query through the Streamlit UI.

---

<!-- Replace the path below with your actual screenshot once captured. -->
![Q4 hallucination trap](docs/screenshots/q4_hallucination_trap.png)

*System returns 'I don't know' when the answer is not grounded in the source documents. Faithfulness score remains 1.00 — the answer makes no unsupported claims.*

---

## Evaluation Queries

Tested against three corporate documents: Apple's Business Conduct Policy, Microsoft's 2023 Annual Report (10-K), and NVIDIA's Code of Conduct.

| # | Query | What it proves |
|---|-------|----------------|
| 1 | What conflicts of interest must Apple employees disclose? | Retrieval precision against a single-source compliance document |
| 2 | What was Microsoft's total revenue in fiscal year 2023? | Numeric extraction from dense financial text |
| 3 | What should NVIDIA employees do if they observe a potential violation? | Multi-chunk synthesis across a reporting-procedure section |
| 4 | What is NVIDIA's formal process for appealing a termination decision? | **Guardrail handling: refuses to fabricate a process absent from the source** |
| 5 | Across all three documents, which provides the most detailed anti-bribery guidance? | Cross-document retrieval and comparative reasoning |

---

## How It Works

1. **Ingest**: `load_documents()` reads all PDFs from `data/raw/`, extracts full text via pypdf, and splits into 1,500-character chunks with 100-character overlap.
2. **Embed**: Each chunk is embedded using Gemini's `gemini-embedding-001` model. A 0.7s rate-limit delay is enforced between requests to stay within the free-tier cap of 100 RPM. Vectors are persisted to ChromaDB on disk.
3. **Retrieve**: The user query is embedded with `task_type=RETRIEVAL_QUERY` to use a query-optimized representation. ChromaDB performs L2 similarity search and returns the top-4 most relevant chunks.
4. **Generate**: Retrieved chunks are concatenated into a context window. The generation LLM receives a strict prompt: answer using only the provided context; return "I don't know" if the answer is absent. Supports DeepSeek, Gemini, and OpenAI via a single environment variable.
5. **Evaluate**: A second LLM call acts as judge, scoring the answer for faithfulness (is every claim grounded in the context?) and relevance (does the answer directly address the question?). Scores are returned as structured JSON with a one-sentence explanation.
6. **Budget**: Token counts from every API call are accumulated against a configurable daily cap (`DAILY_TOKEN_BUDGET`). `BudgetExceededError` is raised before any call that would breach the limit.

---

## Architecture

```
data/raw/*.pdf
      |
   [Parser]      pypdf extract -> whitespace clean -> 1500-char chunks, 100-char overlap
      |
  [Embedder]     gemini-embedding-001, 0.7s/request, RETRIEVAL_DOCUMENT task type
      |
  [ChromaDB]     persistent L2 vector store  (chroma_store/)
      |
  [Retriever]    embed query (RETRIEVAL_QUERY) -> top-4 L2 nearest chunks
      |
  [Generator]    context-grounded prompt -> DeepSeek / Gemini 1.5 Flash / GPT-4o Mini
      |
  [Evaluator]    LLM-as-judge -> faithfulness score + relevance score + reasoning
      |
  [Interface]    Streamlit UI (app.py)  |  CLI + JSON log (main.py)
```

---

## Built With

| Component | Library | Version |
|-----------|---------|---------|
| Language | Python | 3.12 |
| Web UI | Streamlit | 1.58.0 |
| Vector store | ChromaDB | 1.5.9 |
| Embeddings | google-genai | 2.8.0 |
| Generation clients | openai (OpenAI + DeepSeek) | 2.42.0 |
| PDF parsing | pypdf | 6.13.2 |
| Test framework | pytest + pytest-cov | 9.1.0 |

---

## Prerequisites

- Python 3.12 or later
- pip
- API key for at least one generation provider (DeepSeek, Gemini, or OpenAI)
- Gemini API key for embeddings regardless of generation provider

```bash
python --version   # must be 3.12+
pip --version
```

---

## Installation

1. Clone the repository.

```bash
git clone https://github.com/TheJaydenProject/rag-eval.git
cd rag-eval
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Copy the environment template and fill in your API keys.

```bash
cp .env.example .env
```

5. Place your PDF documents in `data/raw/`. Any number of `.pdf` files are supported.

6. Launch the Streamlit UI. The vector store is built automatically on first run.

```bash
streamlit run app.py
```

---

## Configuration

All configuration is read from environment variables. See `.env.example` for the full template.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PROVIDER` | string | `deepseek` | Generation provider. One of `deepseek`, `gemini`, `openai`. |
| `DEEPSEEK_API_KEY` | string | | Required when `PROVIDER=deepseek`. |
| `GEMINI_API_KEY` | string | | Required for embeddings in all configurations. Also required when `PROVIDER=gemini`. |
| `OPENAI_API_KEY` | string | | Required when `PROVIDER=openai`. |
| `DAILY_TOKEN_BUDGET` | integer | `500000` | Maximum tokens consumed per calendar day across all API calls. |
| `CHUNK_SIZE` | integer | `1500` | Characters per document chunk. |
| `CHUNK_OVERLAP` | integer | `100` | Overlap between adjacent chunks in characters. |
| `TOP_K` | integer | `4` | Number of chunks retrieved per query. |
| `CHROMA_PATH` | string | `chroma_store` | Directory for ChromaDB persistence. |
| `COLLECTION_NAME` | string | `rag_eval_docs` | Name of the ChromaDB collection. |

---

## Usage

**Streamlit UI**

```bash
streamlit run app.py
```

Enter a query in the chat input. The sidebar displays faithfulness score, relevance score, and per-stage latency (retrieval, generation, evaluation) for each response.

**CLI batch evaluation**

```bash
python main.py
```

Runs five queries against the loaded documents and writes structured results to `eval_results.json`.

---

## Sample Output

Guardrail response for query 4, where the source documents contain no relevant context:

```json
{
  "query": "What is NVIDIA's formal process for appealing a termination decision?",
  "answer": "I don't know.",
  "faithfulness_score": 1.0,
  "relevance_score": 0.0,
  "is_faithful": true,
  "reasoning": "The retrieved context contains no information about termination appeal procedures. The system correctly declined to speculate."
}
```

Grounded answer for query 3:

```json
{
  "query": "What should NVIDIA employees do if they observe a potential violation?",
  "answer": "NVIDIA employees who observe a potential violation should report it through the Ethics Helpline, available 24 hours a day, or by contacting their manager, Human Resources, or the Legal department.",
  "faithfulness_score": 0.97,
  "relevance_score": 0.95,
  "is_faithful": true,
  "reasoning": "All claims are directly supported by language in the retrieved Code of Conduct chunks."
}
```

---

## Roadmap

- [x] Multi-provider LLM support (DeepSeek, Gemini, OpenAI)
- [x] LLM-as-judge evaluation with faithfulness and relevance scoring
- [x] Token budget enforcement with hard daily cap
- [x] Streamlit UI with per-stage latency breakdown
- [x] GitHub Actions CI with isolated dummy credentials
- [ ] Async batch evaluation for large document corpora
- [ ] RAGAS integration for standardized benchmark comparison
- [ ] Support for additional document formats (DOCX, HTML, Markdown)
- [ ] Per-query latency chart in the Streamlit UI

---

## Contributing

1. Fork the repository and create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes and verify tests pass: `pytest tests/ -v`
3. Open a pull request against `main` with a description of what changed and why.

Bug reports and feature requests go in [GitHub Issues](https://github.com/TheJaydenProject/rag-eval/issues).

---

## Contact

[github.com/TheJaydenProject](https://github.com/TheJaydenProject)
