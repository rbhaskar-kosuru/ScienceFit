"""Blank the LLM settings before src.config loads .env: tests use the code defaults and never hold the real key."""
import os

for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_PROMPT_VERSION", "LLM_API_KEY", "LLM_EXTRA_PARAMS"):
    os.environ[name] = ""
