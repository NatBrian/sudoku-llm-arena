"""Fetch and cache the Sakana AI Sudoku-Bench Nikoli dataset.

Source: SakanaAI/sudoku-bench-nikoli on Hugging Face (CC-BY-4.0, attribution
to Nikoli required). Sakana's own eval harness and the larger CTC/Logic
Masters variant sets were pulled from GitHub in 2026-05; this Nikoli subset
is the only piece still hosted. All 100 puzzles are plain 9x9 sudoku.

Difficulty isn't a dedicated column — Nikoli's own puzzle IDs encode it as a
trailing letter (e=easy, m=medium, h=hard, sd=special/seasonal), so that's
what we key LEVELS off of.
"""
import json
import re
import urllib.request

from . import config

HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=SakanaAI%2Fsudoku-bench-nikoli&config=default&split=test"
    "&offset={offset}&length={length}"
)
DATASET_SIZE = 100
PAGE_SIZE = 100

LEVELS = {
    "easy": re.compile(r"e(-\d+)?$", re.IGNORECASE),
    "medium": re.compile(r"m(-\d+)?$", re.IGNORECASE),
    "hard": re.compile(r"h(-?\d+)?$", re.IGNORECASE),
}

CACHE_PATH_NAME = "nikoli_100.json"


def _cache_path():
    return config.DATA_DIR / CACHE_PATH_NAME


def _classify_level(puzzle_id):
    lower = puzzle_id.lower()
    for level in ("easy", "medium", "hard"):
        if level in lower:
            return level
    for level, pattern in LEVELS.items():
        if pattern.search(puzzle_id):
            return level
    return "other"


def _board_str_to_grid(s, size):
    return [
        [0 if s[r * size + c] == "." else int(s[r * size + c]) for c in range(size)]
        for r in range(size)
    ]


def _row_to_puzzle(row):
    size = row["rows"]
    puzzle_id = row["puzzle_id"]
    return {
        "puzzle_id": puzzle_id,
        "label": f"nikoli-{_classify_level(puzzle_id)}",
        "level": _classify_level(puzzle_id),
        "size": size,
        "box_width": 3,
        "box_height": 3,
        "clues": _board_str_to_grid(row["initial_board"], size),
        "solution": _board_str_to_grid(row["solution"], size),
        "clue_count": sum(1 for ch in row["initial_board"] if ch != "."),
        "rules": row["rules"].strip('"'),
        "author": row["author"],
        "title": row["title"],
        "sudokupad_url": row["sudokupad_url"],
        "source": "sakana-sudoku-bench-nikoli",
    }


def fetch_and_cache(force=False):
    path = _cache_path()
    if path.exists() and not force:
        with open(path) as f:
            return json.load(f)

    puzzles = []
    offset = 0
    while offset < DATASET_SIZE:
        url = HF_ROWS_URL.format(offset=offset, length=PAGE_SIZE)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        rows = data["rows"]
        if not rows:
            break
        for entry in rows:
            puzzles.append(_row_to_puzzle(entry["row"]))
        offset += len(rows)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(puzzles, f, indent=2)
    print(f"  Cached {len(puzzles)} Nikoli puzzles -> {path}")
    return puzzles


def load_nikoli(levels=None, limit=None):
    """levels: iterable of 'easy'/'medium'/'hard'/'other', or None for all."""
    puzzles = fetch_and_cache()
    if levels:
        levels = set(levels)
        puzzles = [p for p in puzzles if p["level"] in levels]
    if limit:
        puzzles = puzzles[:limit]
    return {p["puzzle_id"]: p for p in puzzles}


def level_counts():
    puzzles = fetch_and_cache()
    counts = {}
    for p in puzzles:
        counts[p["level"]] = counts.get(p["level"], 0) + 1
    return counts
