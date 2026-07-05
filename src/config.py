import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PUZZLE_DIR = DATA_DIR / "puzzles"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
OUTPUT_DIR = BASE_DIR / "output"

TEMPERATURE = 0.0
MAX_TOKENS = 2000

# --- Puzzle source ---------------------------------------------------------
# "nikoli": Sakana AI's Sudoku-Bench Nikoli set (100 real 9x9 puzzles, fetched
#   from HF and leveled via src.nikoli). "generator": the original py-sudoku
#   procedural puzzles below, for unlimited/synthetic sizes (4x4/6x6/evil).
PUZZLE_SOURCE = os.getenv("SUDOKU_PUZZLE_SOURCE", "nikoli")  # "nikoli" | "generator"

# Which Nikoli difficulty levels to pull when PUZZLE_SOURCE == "nikoli".
# Options: "easy", "medium", "hard", "other". None/[] means all.
NIKOLI_LEVELS = ["easy"]
NIKOLI_LIMIT = 5  # cap puzzles per run; None for all matching puzzles

PUZZLE_SIZES = {
    "4x4": {"width": 2, "height": 2, "count": 5, "difficulty": 0.4},
    "6x6": {"width": 3, "height": 2, "count": 5, "difficulty": 0.5},
    "9x9": {"width": 3, "height": 3, "count": 5, "difficulty": 0.55},
    "9x9-evil": {"width": 3, "height": 3, "count": 3, "difficulty": 0.7},
}

# --- Models -----------------------------------------------------------------
# Model ids are passed straight to litellm (https://docs.litellm.ai/docs/providers),
# so any provider litellm supports works: "gpt-5", "anthropic/claude-opus-4-8",
# "gemini/gemini-2.5-pro", "deepseek/deepseek-chat", "ollama/llama3", etc.
# Each provider reads its key from its usual env var (OPENAI_API_KEY,
# ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, ...) — litellm handles
# that lookup itself, nothing to configure here beyond exporting the var.
#
# Fill in the frontier models you want to compare, e.g.:
#   MODELS = ["gpt-5", "anthropic/claude-opus-4-8", "gemini/gemini-2.5-pro"]
#
# A model id prefixed "local:<checkpoint-name>" is served locally via
# transformers instead of litellm — see src/train/ for training your own
# small-model checkpoints (tier1 LoRA SFT, tier2 LoRA+GRPO, tier3 full+GRPO)
# and src/train/local_model.py for how the prefix is resolved. If any MODELS
# entry uses this prefix, set MAX_WORKERS = 1 below (one GPU serving all
# local checkpoints; concurrent threads would fight over it).
MODELS = [
    "deepseek/deepseek-chat",
]

# --- Eval protocol -----------------------------------------------------------
# Mirrors Sakana's two Sudoku-Bench modes:
#   "multi-step": one digit per turn, run halts on first invalid placement.
#   "single-shot": model must return the full solved grid in one response.
PROTOCOLS = ["multi-step"]  # subset of ["multi-step", "single-shot"]

MAX_TURNS_MULTIPLIER = 2
MAX_PARSE_RETRIES = 3
MAX_CONSECUTIVE_ERRORS = 5
MAX_API_RETRIES = 3

# Multi-step prompting personas (Sakana itself uses one neutral prompt per
# protocol; these are this repo's own added dimension for richer animations).
STRATEGIES = [
    "s1-direct",
    "s2-naked-singles",
    "s3-hidden-singles",
    "s4-full-logic",
    "s5-guess-verify",
]
RUNS_PER_STRATEGY = 3
MAX_WORKERS = 5
