import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PUZZLE_DIR = DATA_DIR / "puzzles"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
OUTPUT_DIR = BASE_DIR / "output"

API_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or ""
MODEL = "deepseek-chat"
TEMPERATURE = 0.0
MAX_TOKENS = 2000

PUZZLE_SIZES = {
    "4x4": {"width": 2, "height": 2, "count": 5, "difficulty": 0.4},
    "6x6": {"width": 3, "height": 2, "count": 5, "difficulty": 0.5},
    "9x9": {"width": 3, "height": 3, "count": 5, "difficulty": 0.55},
    "9x9-evil": {"width": 3, "height": 3, "count": 3, "difficulty": 0.7},
}

MAX_TURNS_MULTIPLIER = 2
MAX_PARSE_RETRIES = 3
MAX_CONSECUTIVE_ERRORS = 5
MAX_API_RETRIES = 3

STRATEGIES = [
    "s1-direct",
    "s2-naked-singles",
    "s3-hidden-singles",
    "s4-full-logic",
    "s5-guess-verify",
]
RUNS_PER_STRATEGY = 3
MAX_WORKERS = 5
