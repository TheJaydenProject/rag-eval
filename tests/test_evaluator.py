"""
Tests for the JSON extraction logic in src/evaluator.py

Strategy: isolate the re.search + json.loads pipeline by extracting it into
test-facing calls. We mock generate_response so no LLM calls are made — we
are testing the parser's resilience, not the model's output.
"""

import json
import re
import pytest
from unittest.mock import patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# The extraction logic lives inside evaluate_answer. We test it directly by
# replicating the exact regex from evaluator.py, and also by calling
# evaluate_answer with a mocked generate_response.
import src.evaluator as evaluator_module


# ---------------------------------------------------------------------------
# Helpers — mirror the extraction logic from evaluator.py exactly
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """Replicate the extraction path in evaluate_answer."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {raw!r}")
    return json.loads(match.group())


VALID_PAYLOAD = {
    "is_faithful": True,
    "faithfulness_score": 0.9,
    "relevance_score": 0.85,
    "reasoning": "The answer is fully grounded in the context.",
}


# ---------------------------------------------------------------------------
# Core extraction — happy path
# ---------------------------------------------------------------------------

def test_extracts_clean_json():
    raw = json.dumps(VALID_PAYLOAD)
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_json_wrapped_in_markdown_code_fence():
    """
    The most common LLM failure mode: model wraps JSON in ```json ... ```.
    The regex must strip the fence and still return valid data.
    """
    raw = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_json_wrapped_in_plain_code_fence():
    """Some models use ``` without the json language hint."""
    raw = f"```\n{json.dumps(VALID_PAYLOAD)}\n```"
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_json_with_leading_prose():
    raw = f"Sure, here is my evaluation:\n{json.dumps(VALID_PAYLOAD)}"
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_json_with_trailing_prose():
    raw = f"{json.dumps(VALID_PAYLOAD)}\nI hope this helps!"
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_json_surrounded_by_garbage():
    raw = f"GARBAGE TEXT\n```json\n{json.dumps(VALID_PAYLOAD)}\n```\nMORE GARBAGE"
    result = _extract_json(raw)
    assert result == VALID_PAYLOAD


def test_extracts_multiline_json():
    """re.DOTALL must match newlines inside the JSON object."""
    multiline = json.dumps(VALID_PAYLOAD, indent=2)
    result = _extract_json(multiline)
    assert result["faithfulness_score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Field integrity — the extracted dict must have the expected schema
# ---------------------------------------------------------------------------

def test_extracted_dict_has_all_required_keys():
    raw = json.dumps(VALID_PAYLOAD)
    result = _extract_json(raw)
    required_keys = {"is_faithful", "faithfulness_score", "relevance_score", "reasoning"}
    assert required_keys.issubset(result.keys())


def test_scores_are_floats_in_valid_range():
    raw = json.dumps(VALID_PAYLOAD)
    result = _extract_json(raw)
    assert 0.0 <= result["faithfulness_score"] <= 1.0
    assert 0.0 <= result["relevance_score"] <= 1.0


def test_is_faithful_is_boolean():
    raw = json.dumps(VALID_PAYLOAD)
    result = _extract_json(raw)
    assert isinstance(result["is_faithful"], bool)


# ---------------------------------------------------------------------------
# Failure modes — evaluate_answer must raise, not silently return None
# ---------------------------------------------------------------------------

def test_raises_value_error_when_no_json_present():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_json("The model just returned plain English with no JSON.")


def test_raises_value_error_on_empty_string():
    with pytest.raises(ValueError):
        _extract_json("")


def test_raises_json_decode_error_on_malformed_json():
    """
    If the regex matches something that looks like a JSON object but isn't
    (e.g. a Python dict with unquoted keys), json.loads must raise, not
    silently return garbage.
    """
    malformed = "{ is_faithful: True, faithfulness_score: 0.9 }"
    with pytest.raises(json.JSONDecodeError):
        _extract_json(malformed)


# ---------------------------------------------------------------------------
# Integration — evaluate_answer with mocked LLM
# ---------------------------------------------------------------------------

def test_evaluate_answer_returns_dict_on_clean_response():
    """evaluate_answer must return a dict when the mocked LLM gives clean JSON."""
    with patch.object(evaluator_module, "generate_response", return_value=json.dumps(VALID_PAYLOAD)):
        result = evaluator_module.evaluate_answer(
            query="What is the policy?",
            answer="The policy requires annual reviews.",
            context_chunks=[{"text": "Annual reviews are required."}],
        )
    assert result["is_faithful"] is True
    assert result["faithfulness_score"] == pytest.approx(0.9)


def test_evaluate_answer_handles_markdown_fence_from_llm():
    """evaluate_answer must survive a model that wraps JSON in a code fence."""
    fenced = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    with patch.object(evaluator_module, "generate_response", return_value=fenced):
        result = evaluator_module.evaluate_answer(
            query="What is the policy?",
            answer="Annual reviews are required.",
            context_chunks=[{"text": "The policy mandates annual reviews."}],
        )
    assert "faithfulness_score" in result


def test_evaluate_answer_raises_on_unparseable_llm_output():
    """If the LLM returns no JSON at all, evaluate_answer must raise ValueError."""
    with patch.object(evaluator_module, "generate_response", return_value="I cannot evaluate this."):
        with pytest.raises(ValueError, match="No JSON object found"):
            evaluator_module.evaluate_answer(
                query="Q",
                answer="A",
                context_chunks=[{"text": "context"}],
            )