"""
conftest.py — project-wide pytest configuration.

Adds the project root to sys.path so all test files can import from src/
without requiring a package install. Also sets the dummy environment
variables that config.py validates at import time, preventing ImportError
when the real .env is absent (e.g. in CI).
"""

import os
import sys
from pathlib import Path

# Insert repo root so `import src.x` works from any test file.
sys.path.insert(0, str(Path(__file__).parent.parent))

# config.py raises ValueError if these are missing — set dummies for test runs.
os.environ.setdefault("PROVIDER", "deepseek")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-dummy")
os.environ.setdefault("GEMINI_API_KEY", "gm-test-dummy")