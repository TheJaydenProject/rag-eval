"""
Tests for src/budget_tracker.py

The budget tracker is the only guardrail preventing runaway API spend.
These tests verify enforcement at the boundary level.

Patching strategy: budget_tracker.py uses `import config` via a sys.path
hack, so it binds to the bare 'config' module in sys.modules, not 'src.config'.
We must patch `budget_tracker.config.DAILY_TOKEN_BUDGET` (the reference
budget_tracker holds) rather than `src.config.DAILY_TOKEN_BUDGET`.
Similarly, BUDGET_FILE is a module-level Path — we patch `budget_tracker.BUDGET_FILE`
directly and point it at a tmp_path so tests never touch the real file on disk.
"""

import json
import pytest
from datetime import date
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import src.budget_tracker as budget_tracker
from src.budget_tracker import BudgetExceededError, check_and_record, get_usage_today


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_budget_file(tmp_path, monkeypatch):
    """
    Redirect BUDGET_FILE to a temp directory for every test.
    autouse=True means this runs for all tests in this module automatically.
    """
    mock_file = tmp_path / ".token_budget.json"
    monkeypatch.setattr(budget_tracker, "BUDGET_FILE", mock_file)
    return mock_file


@pytest.fixture()
def low_budget(monkeypatch):
    """Set a low budget (100 tokens) on the module config reference budget_tracker holds."""
    monkeypatch.setattr(budget_tracker.config, "DAILY_TOKEN_BUDGET", 100)


def _seed_spend(mock_file: Path, tokens: int) -> None:
    """Pre-populate today's spend directly so we can start tests mid-day."""
    today = str(date.today())
    mock_file.write_text(json.dumps({today: tokens}))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_first_call_within_budget_succeeds(low_budget):
    check_and_record(99)


def test_spend_is_persisted_to_file(tmp_path, monkeypatch, low_budget):
    mock_file = tmp_path / ".token_budget.json"
    monkeypatch.setattr(budget_tracker, "BUDGET_FILE", mock_file)

    check_and_record(50)

    today = str(date.today())
    assert json.loads(mock_file.read_text())[today] == 50


def test_cumulative_spend_accumulates_correctly(monkeypatch):
    monkeypatch.setattr(budget_tracker.config, "DAILY_TOKEN_BUDGET", 1000)

    check_and_record(300)
    check_and_record(400)

    today = str(date.today())
    assert json.loads(budget_tracker.BUDGET_FILE.read_text())[today] == 700


def test_get_usage_today_returns_zero_with_no_file():
    assert get_usage_today() == 0


def test_get_usage_today_reflects_existing_spend():
    _seed_spend(budget_tracker.BUDGET_FILE, 250)
    assert get_usage_today() == 250


# ---------------------------------------------------------------------------
# Budget enforcement — the security-critical path
# ---------------------------------------------------------------------------

def test_budget_exceeded_raises_error(low_budget):
    """
    Core guardrail: 90 + 11 = 101 on a 100-token cap must raise before
    any network call could be made by the caller.
    """
    check_and_record(90)

    with pytest.raises(BudgetExceededError):
        check_and_record(11)


def test_budget_exceeded_error_message_contains_limit(low_budget):
    """Error message must include the budget figure so callers can surface it."""
    check_and_record(90)

    with pytest.raises(BudgetExceededError, match="100"):
        check_and_record(11)


def test_spend_is_not_written_on_budget_breach(low_budget):
    """
    A failed check_and_record must not modify stored spend. If the overage
    were written, the tracker would permanently block all further calls even
    after a legitimate budget increase.
    """
    check_and_record(90)

    with pytest.raises(BudgetExceededError):
        check_and_record(11)

    today = str(date.today())
    assert json.loads(budget_tracker.BUDGET_FILE.read_text())[today] == 90


def test_exact_budget_limit_is_allowed(low_budget):
    """Spending exactly the cap (not one over) must succeed — > not >=."""
    check_and_record(100)


def test_one_token_over_limit_raises(low_budget):
    """101 on a 100-token cap must raise."""
    with pytest.raises(BudgetExceededError):
        check_and_record(101)


def test_pre_seeded_spend_contributes_to_limit(low_budget):
    """
    Budget enforced across process boundaries. 95 tokens recorded from a
    prior session + 10 attempted now = 105 > 100, must raise.
    """
    _seed_spend(budget_tracker.BUDGET_FILE, 95)

    with pytest.raises(BudgetExceededError):
        check_and_record(10)


# ---------------------------------------------------------------------------
# Resilience — corrupt / missing file
# ---------------------------------------------------------------------------

def test_corrupted_budget_file_is_treated_as_empty(monkeypatch):
    """
    A malformed file (e.g. truncated on a crash mid-write) must be treated as
    zero-spend rather than crashing the entire pipeline.
    """
    budget_tracker.BUDGET_FILE.write_text("{ not valid json %%%")
    monkeypatch.setattr(budget_tracker.config, "DAILY_TOKEN_BUDGET", 500)

    check_and_record(100)


def test_zero_tokens_does_not_raise(low_budget):
    """0-token calls (e.g. empty embedding response) must not crash."""
    check_and_record(0)