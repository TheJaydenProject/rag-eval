import json
from datetime import date
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config

BUDGET_FILE = Path(".token_budget.json")


def _load() -> dict:
    try:
        return json.loads(BUDGET_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(data: dict) -> None:
    BUDGET_FILE.write_text(json.dumps(data, indent=2))


def check_and_record(tokens: int) -> None:
    """
    Raises BudgetExceededError if recording `tokens` would breach the daily cap.
    Call this with actual usage from the API response wherever possible.
    """
    today = str(date.today())
    data = _load()
    used: int = data.get(today, 0)

    if used + tokens > config.DAILY_TOKEN_BUDGET:
        raise BudgetExceededError(
            f"Daily token budget exceeded: {used + tokens:,} would exceed {config.DAILY_TOKEN_BUDGET:,}. "
            f"Used today: {used:,}. Reset tomorrow or raise DAILY_TOKEN_BUDGET in .env."
        )

    data[today] = used + tokens
    _save(data)


def get_usage_today() -> int:
    today = str(date.today())
    return _load().get(today, 0)


class BudgetExceededError(Exception):
    pass
