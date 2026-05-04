"""
Central configuration.
All Ollama and path settings live here.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_TIMEOUT_SECONDS = 120

OLLAMA_OPTIONS: dict = {
    "temperature": 0,
    "num_predict": 2048,
    "repeat_penalty": 1.2,
}

# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------
ERR_OLLAMA_UNAVAILABLE = (
    "Ollama unavailable. Start Ollama and make sure qwen2.5:3b is installed."
)
ERR_MODEL_MISSING = "Model qwen2.5:3b not found. Run: ollama pull qwen2.5:3b"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs" / "terminal_runs"
DEFAULT_CSV = DATA_DIR / "production_scheduling_basic_dataset.csv"

DEFAULT_USER_INPUT = (
    "A factory has 2 identical machines and works 480 minutes per day. "
    "Schedule all jobs from the CSV dataset and minimize makespan. "
    "Use production_qty multiplied by processing_time_min_per_unit as job duration. "
    "Check demand and inventory before scheduling."
)
