import os

from dotenv import load_dotenv

load_dotenv()

# PROVIDER selects the GENERATION/judge backend.
PROVIDER: str = os.getenv("PROVIDER", "openai").lower()

if PROVIDER not in ("deepseek", "gemini", "openai", "openrouter"):
    raise ValueError(
        f"PROVIDER must be 'deepseek', 'gemini', 'openai', or 'openrouter', got: {PROVIDER!r}"
    )

# EMBEDDING_PROVIDER selects the embeddings backend independently of PROVIDER.
# If unset, it defaults to whatever generation already needs a key for, so a
# single PROVIDER setting is enough to run with one API key.
_DEFAULT_EMBEDDING_PROVIDER: dict[str, str] = {
    "openai": "openai",
    "openrouter": "openrouter",
    "gemini": "gemini",
    "deepseek": "gemini",
}
EMBEDDING_PROVIDER: str = os.getenv(
    "EMBEDDING_PROVIDER", _DEFAULT_EMBEDDING_PROVIDER[PROVIDER]
).lower()

if EMBEDDING_PROVIDER not in ("openai", "gemini", "openrouter"):
    raise ValueError(
        f"EMBEDDING_PROVIDER must be 'openai', 'gemini', or 'openrouter', "
        f"got: {EMBEDDING_PROVIDER!r}"
    )

DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# --- Generation key validation ---
if PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY is required when PROVIDER=deepseek")

if PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required when PROVIDER=gemini")

if PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required when PROVIDER=openai")

if PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is required when PROVIDER=openrouter")

# --- Embedding key validation ---
if EMBEDDING_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

if EMBEDDING_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")

if EMBEDDING_PROVIDER == "openrouter" and not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is required when EMBEDDING_PROVIDER=openrouter")

EMBEDDING_MODELS: dict[str, str] = {
    "gemini": "gemini-embedding-001",
    "openai": "text-embedding-3-small",
    # OpenRouter embedding model slug is overridable via OPENROUTER_EMBEDDING_MODEL.
    "openrouter": os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
}
GENERATION_MODELS: dict[str, str] = {
    "deepseek": "deepseek-chat",
    "gemini": "gemini-1.5-flash",
    "openai": "gpt-5.4-nano",
    # OpenRouter model slug is overridable via OPENROUTER_MODEL.
    "openrouter": os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite"),
}

EMBEDDING_MODEL: str = EMBEDDING_MODELS[EMBEDDING_PROVIDER]
GENERATION_MODEL: str = GENERATION_MODELS[PROVIDER]

OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
# Optional attribution headers OpenRouter recommends (shown on their dashboard).
OPENROUTER_APP_URL: str = os.getenv(
    "OPENROUTER_APP_URL", "https://github.com/TheJaydenProject/rag-eval"
)
OPENROUTER_APP_TITLE: str = os.getenv("OPENROUTER_APP_TITLE", "rag-eval")

# Per-request delay enforced while building the vector store, to stay under the
# embedding provider's rate limit. Gemini's free tier caps at 100 RPM, so 0.7s
# keeps large builds safe; OpenAI/OpenRouter embeddings have far higher limits
# and need no throttle. Override with EMBEDDING_RATE_LIMIT_DELAY if your tier differs.
_DEFAULT_EMBEDDING_DELAY: dict[str, float] = {"gemini": 0.7, "openai": 0.0, "openrouter": 0.0}
EMBEDDING_RATE_LIMIT_DELAY: float = float(
    os.getenv(
        "EMBEDDING_RATE_LIMIT_DELAY", _DEFAULT_EMBEDDING_DELAY[EMBEDDING_PROVIDER]
    )
)

CHUNK_SIZE: int = 1500
CHUNK_OVERLAP: int = 100
TOP_K: int = 4
MAX_QUERY_LENGTH: int = 2000
CHROMA_PATH: str = "chroma_store"
COLLECTION_NAME: str = "rag_eval_docs"
DAILY_TOKEN_BUDGET: int = int(os.getenv("DAILY_TOKEN_BUDGET", "500000"))
