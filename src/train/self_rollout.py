"""Collects "recovery" SFT examples from a trained checkpoint's OWN live
rollouts, to close the exposure-bias gap between training and eval:
build_sft_examples (data.py) only ever shows the model grid states built by
revealing the TRUE solution cell-by-cell, so the model never once sees a
state resulting from its own error. Eval (loop.py) is a live rollout where
its own mistakes compound with no prior practice recovering from one.

Here we run the checkpoint through the real GameLoop on fresh puzzles,
capture every grid state it actually visits (including any legal-but-wrong-
for-this-solution placements it made along the way), and mine each state for
any move an implemented solving technique can justify from that state's
CURRENT constraints alone (validator.classify_move never references the
puzzle's original solution, so this works unmodified on self-corrupted
states, not just clean ones).
"""
import json
import random

from ..loop import GameLoop
from ..puzzles import generate_puzzle
from ..strategies import build_prompt
from ..validator import classify_move, find_hidden_singles, find_naked_singles
from . import config as train_config

SELF_ROLLOUT_SEED = 1  # disjoint from train_config.DATA_SEED's puzzle pool


def _generate_puzzles(n, box_width=3, box_height=3, difficulty=0.6, seed=SELF_ROLLOUT_SEED):
    rng = random.Random(seed)
    puzzles = []
    for i in range(n):
        puzzle_seed = rng.randrange(0xFFFFFFFF)
        puzzle = generate_puzzle(box_width, box_height, difficulty=difficulty, seed=puzzle_seed)
        puzzle["puzzle_id"] = f"selfrollout-{i:04d}"
        puzzles.append(puzzle)
    return puzzles


def _find_justified_move(grid, box_width, box_height, rng):
    """Cheapest-first search for one (row, col, value, justification) the
    validator can justify from this exact grid: bulk naked/hidden-single
    detectors first, falling back to a full per-cell classify_move scan
    (which additionally covers pair/pointing/box-line-derived singles)."""
    naked = find_naked_singles(grid, box_width, box_height)
    if naked:
        r, c, v = rng.choice(naked)
        return r, c, v, classify_move(grid, box_width, box_height, r, c, v)[1]

    hidden = find_hidden_singles(grid, box_width, box_height)
    if hidden:
        r, c, v = rng.choice(hidden)
        return r, c, v, classify_move(grid, box_width, box_height, r, c, v)[1]

    size = len(grid)
    cells = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == 0]
    rng.shuffle(cells)
    for r, c in cells:
        for v in range(1, size + 1):
            result = classify_move(grid, box_width, box_height, r, c, v)
            if result:
                return r, c, v, result[1]
    return None


def collect_recovery_examples(checkpoint_name, n_puzzles=25, strategy_id=None, seed=SELF_ROLLOUT_SEED):
    """Returns a list of {"messages": [...]} SFT examples mined from
    `checkpoint_name`'s own multi-turn rollouts on fresh puzzles."""
    strategy_id = strategy_id or train_config.TRAIN_STRATEGY_ID
    model = f"local:{checkpoint_name}"
    puzzles = _generate_puzzles(n_puzzles, seed=seed)
    rng = random.Random(seed)

    examples = []
    seen_states = set()
    for puzzle in puzzles:
        game = GameLoop(puzzle, model, "multi-step", strategy_id, run_number=1)
        result = game.run(log_prefix=f"[self-rollout {puzzle['puzzle_id']}]")
        bw, bh = puzzle["box_width"], puzzle["box_height"]

        for turn in result["turns"]:
            if turn["turn"] == 1:
                # Identical to puzzle["clues"] — already covered by the
                # normal oracle-snapshot training data, not a self-generated
                # state worth adding here.
                continue
            grid = turn["grid_before"]
            state_key = json.dumps(grid)
            if state_key in seen_states:
                continue

            found = _find_justified_move(grid, bw, bh, rng)
            if found is None:
                continue
            seen_states.add(state_key)

            r, c, value, justification = found
            prompt = build_prompt(strategy_id, puzzle, grid)
            completion = f"REASONING: {justification}\nMOVE: R{r+1}C{c+1} = {value}"
            examples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            })

    return examples


def save_jsonl(examples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-puzzles", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    examples = collect_recovery_examples(args.checkpoint, n_puzzles=args.n_puzzles)
    print(f"Collected {len(examples)} self-rollout recovery examples")
    out_path = save_jsonl(examples, Path(args.out))
    print(f"Saved -> {out_path}")
