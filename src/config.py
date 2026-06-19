import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER: str = os.getenv("PROVIDER", "deepseek").lower()

if PROVIDER not in ("deepseek", "gemini", "openai"):
    raise ValueError(f"PROVIDER must be 'deepseek', 'gemini', or 'openai', got: {PROVIDER!r}")

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

if PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is required when PROVIDER=deepseek")

if PROVIDER == "deepseek" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required when PROVIDER=deepseek (used for embeddings)")

if PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required when PROVIDER=gemini")

if PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required when PROVIDER=openai")

# DeepSeek has no embedding API — embeddings route to Gemini's free tier instead.
EMBEDDING_MODELS: dict[str, str] = {
    "deepseek": "gemini-embedding-001",
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
}
GENERATION_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-4o-mini",
}

EMBEDDING_MODEL: str = EMBEDDING_MODELS[PROVIDER]
GENERATION_MODEL: str = GENERATION_MODELS[PROVIDER]

CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 100
TOP_K: int = 4
MAX_QUERY_LENGTH: int = 2000
CHROMA_PATH: str = "chroma_store"
COLLECTION_NAME: str = "rag_eval_docs"
DAILY_TOKEN_BUDGET: int = int(os.getenv("DAILY_TOKEN_BUDGET", "500000"))
