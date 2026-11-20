"""All settings come from environment variables, read in one place."""
import os

from dotenv import load_dotenv

load_dotenv()  # copies values from .env into the environment, if the file exists

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")

# Ollama runs locally; OPENAI-compatible endpoint lives at /v1 on 11434.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Calendar backend. "fake" is used by tests and evals; "google" talks to the real API.
CALENDAR_BACKEND = os.getenv("CALENDAR_BACKEND", "fake")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# Telegram bot. Allowlist is a comma-separated list of numeric user IDs.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USER_IDS = tuple(
    int(x) for x in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",") if x.strip()
)
