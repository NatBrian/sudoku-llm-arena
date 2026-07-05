"""Generates synthetic 9x9 training puzzles via py-sudoku's procedural
generator (src/puzzles.py) — deliberately NOT derived from Sakana's real
Nikoli-100 set. Even a held-out-split of Nikoli puzzles would still teach
Nikoli-specific structure that the real eval doesn't test frontier models
on, so training and eval must come from disjoint sources, not just disjoint
puzzle instances. All 100 real Nikoli puzzles are reserved for eval only.
"""
import random

from ..puzzles import generate_puzzle
from . import config as train_config


def generate_train_puzzles(seed=None, box_width=3, box_height=3, difficulty_mix=None):
    """Returns a flat list of synthetic puzzle dicts spanning a difficulty
    mix that approximates the real Nikoli-100 eval set's easy/medium/hard
    split (12/35/51), so training difficulty isn't skewed vs. what's tested.

    `box_width`/`box_height` and `difficulty_mix` default to the canonical
    9x9 setup (train_config.TRAIN_DIFFICULTY_MIX) but can be overridden to
    generate curriculum-stage puzzles at other board sizes (see
    train_config.TRAIN_CURRICULUM_STAGES)."""
    seed = train_config.DATA_SEED if seed is None else seed
    difficulty_mix = difficulty_mix or train_config.TRAIN_DIFFICULTY_MIX
    rng = random.Random(seed)

    out = []
    i = 0
    for label, (difficulty, count) in difficulty_mix.items():
        for _ in range(count):
            puzzle_seed = rng.randrange(0xFFFFFFFF)
            puzzle = generate_puzzle(box_width, box_height, difficulty=difficulty, seed=puzzle_seed)
            puzzle["puzzle_id"] = f"synth-{label}-{i:04d}"
            puzzle["level"] = label
            out.append(puzzle)
            i += 1
    rng.shuffle(out)
    return out
