import json
import re

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from llm_client import generate_response


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    context_text: str = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt: str = f"""You are a precise document assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't know."

Context:
{context_text}

Question: {query}

Answer:"""
    return generate_response(prompt)


def evaluate_answer(query: str, answer: str, context_chunks: list[dict]) -> dict:
    """
    LLM-as-a-judge: enforces a strict JSON output schema via the prompt.
    re.search extracts the JSON block regardless of surrounding text or code
    fences the model may produce — more robust than string splitting.
    """
    context_text: str = "\n\n---\n\n".join(chunk["text"] for chunk in context_chunks)
    eval_prompt: str = f"""You are an impartial evaluation judge. Score the answer strictly against the source context.

Source Context:
{context_text}

Question: {query}

Answer: {answer}

Return ONLY a valid JSON object with this exact schema and no other text:
{{
  "is_faithful": <true if every claim in the answer is supported by the context, false otherwise>,
  "faithfulness_score": <float 0.0 to 1.0, where 1.0 means fully grounded in context>,
  "relevance_score": <float 0.0 to 1.0, where 1.0 means the answer directly addresses the question>,
  "reasoning": "<one sentence explaining your scores>"
}}"""

    raw: str = generate_response(eval_prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in evaluator response: {raw!r}")

    return json.loads(match.group())
