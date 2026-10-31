import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT_DIR = ROOT / "telegram_bot"

# Real .env files must not leak into tests: explicit env vars win over both
# pydantic-settings env files and python-dotenv (which does not override).
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TELEGRAM_BOT_TOKEN"] = "123456:test-token"
os.environ["WEB_SECRET_KEY"] = "test-secret-key-for-pytest-0123456789"
os.environ["ADMIN_EMAIL"] = ""
os.environ["ADMIN_PASSWORD"] = ""
os.environ["DEBUG"] = "false"
os.environ["MODEL"] = "gpt-6.1-sol"
os.environ["REASONING_EFFORT"] = "low"

for path in (ROOT, BOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
