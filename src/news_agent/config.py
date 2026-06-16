import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")
DEFAULT_MODELS = os.getenv("DEFAULT_MODELS", "llama3.2:1b")
DEFAULT_PROMPTS = os.getenv("DEFAULT_PROMPTS", "abstract")
DEFAULT_NUM_AGENTS = int(os.getenv("DEFAULT_NUM_AGENTS", "1"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.2"))

# Enable verbose per-tool debug logging (input/output of each tool call).
DEBUG_LOGS = os.getenv("DEBUG_LOGS", "false").strip().lower() in ("1", "true", "yes", "on")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
