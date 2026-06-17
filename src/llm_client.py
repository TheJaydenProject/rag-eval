from google import genai
from google.genai import types as genai_types
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import config
from budget_tracker import check_and_record

# Initialise client(s) once at import time.
# PROVIDER=deepseek needs both: Gemini for embeddings, DeepSeek for generation.
_gemini_client: genai.Client | None = None
_openai_client: OpenAI | None = None
_deepseek_client: OpenAI | None = None

if config.PROVIDER in ("gemini", "deepseek"):
    # Gemini handles embeddings for both the pure-gemini and hybrid-deepseek paths.
    _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)

if config.PROVIDER == "openai":
    _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

if config.PROVIDER == "deepseek":
    _deepseek_client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )


def get_embedding(text: str) -> list[float]:
    # Gemini handles embeddings for both PROVIDER=gemini and PROVIDER=deepseek.
    if config.PROVIDER in ("gemini", "deepseek"):
        result = _gemini_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        # Gemini doesn't return token counts for embedding calls — estimate conservatively.
        check_and_record(len(text) // 4)
        return result.embeddings[0].values

    result = _openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text,
    )
    check_and_record(result.usage.total_tokens)
    return result.data[0].embedding


def get_query_embedding(text: str) -> list[float]:
    # Gemini distinguishes document vs query task types for better retrieval accuracy.
    if config.PROVIDER in ("gemini", "deepseek"):
        result = _gemini_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        check_and_record(len(text) // 4)
        return result.embeddings[0].values

    result = _openai_client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=text,
    )
    check_and_record(result.usage.total_tokens)
    return result.data[0].embedding


def generate_response(prompt: str) -> str:
    if config.PROVIDER == "gemini":
        response = _gemini_client.models.generate_content(
            model=config.GENERATION_MODEL,
            contents=prompt,
        )
        tokens = getattr(response.usage_metadata, "total_token_count", len(prompt) // 4)
        check_and_record(tokens)
        return response.text.strip()

    client = _deepseek_client if config.PROVIDER == "deepseek" else _openai_client
    response = client.chat.completions.create(
        model=config.GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    check_and_record(response.usage.total_tokens)
    return response.choices[0].message.content.strip()
