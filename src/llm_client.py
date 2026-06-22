import os
import sys

from google import genai
from google.genai import errors, types
from google.genai.types import HttpOptions
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
import config
from budget_tracker import check_and_record

# Initialise client(s) once at import time. Embeddings and generation can use
# different backends now (e.g. PROVIDER=openrouter for chat, EMBEDDING_PROVIDER=
# openai for embeddings), so each client is created if either half needs it.
_gemini_client: genai.Client | None = None
_openai_client: OpenAI | None = None
_deepseek_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None

if config.PROVIDER == "gemini" or config.EMBEDDING_PROVIDER == "gemini":
    _gemini_client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=HttpOptions(api_version="v1"),
    )

if config.PROVIDER == "openai" or config.EMBEDDING_PROVIDER == "openai":
    _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

if config.PROVIDER == "deepseek":
    _deepseek_client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

if config.PROVIDER == "openrouter" or config.EMBEDDING_PROVIDER == "openrouter":
    # OpenRouter is OpenAI-compatible; the attribution headers are optional but
    # recommended (they surface your app on the OpenRouter dashboard).
    _openrouter_client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": config.OPENROUTER_APP_URL,
            "X-Title": config.OPENROUTER_APP_TITLE,
        },
    )


def _gemini_embed(text: str, task_type: str) -> list[float]:
    if not text:
        raise ValueError("Input text for embedding cannot be empty.")

    try:
        result = _gemini_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        check_and_record(len(text) // 4)
        return result.embeddings[0].values
    except errors.ClientError as e:
        raise RuntimeError(
            f"Embedding request failed. Verify {config.EMBEDDING_MODEL} is active. Details: {e}"
        )


def _openai_compatible_embed(client: OpenAI, text: str) -> list[float]:
    if not text:
        raise ValueError("Input text for embedding cannot be empty.")

    result = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text,
    )
    check_and_record(result.usage.total_tokens)
    return result.data[0].embedding


def get_embedding(text: str) -> list[float]:
    if config.EMBEDDING_PROVIDER == "gemini":
        return _gemini_embed(text, "RETRIEVAL_DOCUMENT")
    client = _openrouter_client if config.EMBEDDING_PROVIDER == "openrouter" else _openai_client
    return _openai_compatible_embed(client, text)


def get_query_embedding(text: str) -> list[float]:
    if config.EMBEDDING_PROVIDER == "gemini":
        return _gemini_embed(text, "RETRIEVAL_QUERY")
    client = _openrouter_client if config.EMBEDDING_PROVIDER == "openrouter" else _openai_client
    return _openai_compatible_embed(client, text)


def generate_response(prompt: str) -> str:
    if config.PROVIDER == "gemini":
        response = _gemini_client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
        )
        tokens = getattr(response.usage_metadata, "total_token_count", len(prompt) // 4)
        check_and_record(tokens)
        return response.text.strip()

    # All remaining providers speak the OpenAI chat-completions protocol.
    if config.PROVIDER == "deepseek":
        client = _deepseek_client
    elif config.PROVIDER == "openrouter":
        client = _openrouter_client
    else:
        client = _openai_client

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    check_and_record(response.usage.total_tokens)
    return response.choices[0].message.content.strip()
