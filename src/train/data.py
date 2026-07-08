"""Turn synthetic puzzles (src/train/synth.py) into training examples for
tier1 (SFT) and tier2/tier3 (GRPO).

Both share the same idea: take a puzzle's solution grid, reveal cells in a
random order to build partial-fill "snapshots", and use each snapshot as one
training prompt (current grid -> next correct move). This mirrors what the
model actually sees at eval time via loop.py's multi-step protocol.
"""
import json
import random
from copy import deepcopy

from ..validator import classify_move, count_empty
from ..strategies import build_prompt
from . import config as train_config
from .synth import generate_train_puzzles


def _snapshot_states(puzzle, n_snapshots, rng):
    """Yields (grid_state, target_row, target_col, target_value) pairs by
    revealing solution cells in a random order, stopping at n_snapshots
    random points along the way."""
    size = puzzle["size"]
    solution = puzzle["solution"]
    clues = puzzle["clues"]

    empty_cells = [(r, c) for r in range(size) for c in range(size) if clues[r][c] == 0]
    rng.shuffle(empty_cells)

    grid = deepcopy(clues)
    n_snapshots = min(n_snapshots, len(empty_cells))
    # Random indices into the reveal order, so snapshots span early/mid/late
    # game states rather than all being fresh-puzzle first moves.
    pick_at = set(rng.sample(range(len(empty_cells)), n_snapshots))

    for i, (r, c) in enumerate(empty_cells):
        if i in pick_at:
            target_value = solution[r][c]
            yield deepcopy(grid), r, c, target_value
        grid[r][c] = solution[r][c]


def _reasoning_for(grid, box_width, box_height, r, c, value):
    """Returns a real justification string if `value` at (r, c) is provable by
    an implemented solving technique (see validator.classify_move), else None.
    Cells with no such justification are excluded from SFT training rather
    than given a vacuous "matches the puzzle's unique solution" filler —
    training on reasoning that doesn't explain the answer is worse than not
    training on it at all."""
    result = classify_move(grid, box_width, box_height, r, c, value)
    if result is None:
        return None
    _technique, justification = result
    return justification


def build_sft_examples(strategy_id=None, seed=None, box_width=3, box_height=3, difficulty_mix=None):
    """Returns a list of {"messages": [...]} chat examples for trl SFTTrainer.
    Snapshots whose target move isn't justified by an implemented solving
    technique (see _reasoning_for) are skipped rather than trained on with a
    vacuous filler, so the resulting example count is smaller than
    len(puzzles) * SNAPSHOTS_PER_PUZZLE."""
    strategy_id = strategy_id or train_config.TRAIN_STRATEGY_ID
    seed = train_config.DATA_SEED if seed is None else seed
    rng = random.Random(seed)

    puzzles = generate_train_puzzles(seed=seed, box_width=box_width, box_height=box_height, difficulty_mix=difficulty_mix)

    examples = []
    skipped = 0
    for puzzle in puzzles:
        for grid, r, c, value in _snapshot_states(puzzle, train_config.SNAPSHOTS_PER_PUZZLE, rng):
            reasoning = _reasoning_for(grid, puzzle["box_width"], puzzle["box_height"], r, c, value)
            if reasoning is None:
                skipped += 1
                continue
            prompt = build_prompt(strategy_id, puzzle, grid)
            completion = f"REASONING: {reasoning}\nMOVE: R{r+1}C{c+1} = {value}"
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                # Qwen3's chat template defaults to inserting a <think>...</think>
                # block; Qwen2.5's template ignores unknown kwargs, so this is
                # safe to set unconditionally. Keeps train/eval prompts on the
                # same distribution regardless of which base model is used —
                # local_model.py passes the same kwarg at inference time.
                "chat_template_kwargs": {"enable_thinking": False},
            })
    print(f"  {skipped} snapshots skipped (no technique justifies the target move)")
    return examples


def build_grpo_examples(strategy_id=None, seed=None):
    """Returns a list of dicts with a chat "prompt" plus the extra columns
    reward.py's reward_func needs to score the model's own completion
    (grid_before/box dims/solution), one row per sampled partial-fill state."""
    strategy_id = strategy_id or train_config.TRAIN_STRATEGY_ID
    seed = train_config.DATA_SEED if seed is None else seed
    rng = random.Random(seed)

    puzzles = generate_train_puzzles(seed=seed)

    examples = []
    for puzzle in puzzles:
        for grid, r, c, value in _snapshot_states(puzzle, train_config.SNAPSHOTS_PER_PUZZLE, rng):
            prompt = build_prompt(strategy_id, puzzle, grid)
            examples.append({
                "prompt": [{"role": "user", "content": prompt}],
                "grid_before": json.dumps(grid),
                "box_width": puzzle["box_width"],
                "box_height": puzzle["box_height"],
                "size": puzzle["size"],
                "solution": json.dumps(puzzle["solution"]),
                "clues": json.dumps(puzzle["clues"]),
            })
    return examples


def save_jsonl(examples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path
