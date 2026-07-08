"""Evaluates a trained checkpoint (or the raw base model, for tier0) against
the full real Nikoli-100 set through the exact same loop.py/swarm.py harness
frontier models run through — same protocol/strategy as training so numbers
are comparable, and every move's full trajectory (reasoning, parsed move,
validation errors, grid before/after) lands in data/parsed/ via the existing
save_parsed_run, keyed by this run's distinct "local:<checkpoint>" model
string so it never collides with other runs.
"""
import json
import time

from .. import config as base_config
from ..storage import model_slug
from ..swarm import run_swarm, print_summary
from . import config as train_config


def _load_all_parsed_runs(model_string, levels=None, parsed_dir=None):
    """run_swarm() only returns freshly-run jobs — puzzles already recorded in
    a prior run (via run_exists dedup) are silently skipped and excluded from
    its return value. Re-derive the full result set for this model from disk
    instead, so a summary is correct whether this is a fresh run, a resumed
    partial run, or a fully-cached re-run.

    `levels` filters to specific Nikoli difficulty levels (e.g. ["hard"]) —
    every parsed run JSON already carries its puzzle's "level" (see
    swarm.py's run_single), so a level-filtered summary can be recomputed
    from a prior full-100 run's cache with no new generation needed, as long
    as that full run already covered the target level (a superset)."""
    slug = model_slug(model_string)
    rows = []
    for path in (parsed_dir or base_config.PARSED_DIR).glob(f"*_{slug}_*.json"):
        with open(path) as f:
            row = json.load(f)
        if levels is None or row.get("level") in levels:
            rows.append(row)
    return rows


def register_tier0_checkpoint(model_key):
    """tier0 is the untrained base model — stub a meta.json so local_model.py's
    loader can serve it under the same "local:<name>" convention as trained
    checkpoints, with no adapter and no training step."""
    base_model_id = train_config.BASE_MODELS[model_key]
    out_name = f"{model_key}-tier0-zeroshot"
    out_dir = train_config.checkpoint_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"base_model": base_model_id, "tier": "tier0-zeroshot", "adapter": False}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_name


def evaluate_checkpoint(checkpoint_name, levels=None):
    """Runs the real Nikoli set (multi-step, TRAIN_STRATEGY_ID only, 1 run
    each since decoding is greedy/deterministic) against one checkpoint.
    Mutates then restores src.config's globals (the harness's own pattern).

    `levels` restricts to specific Nikoli difficulty levels (e.g. ["hard"]
    for the 51 hard puzzles instead of the full 100) — if this checkpoint
    was already evaluated on the full set, every puzzle in `levels` is
    already cached (same weights, greedy decoding) so this is a free
    re-aggregation with no new generation, not a fresh eval pass."""
    saved = {
        k: getattr(base_config, k)
        for k in ("PUZZLE_SOURCE", "NIKOLI_LEVELS", "NIKOLI_LIMIT", "PROTOCOLS",
                   "STRATEGIES", "RUNS_PER_STRATEGY", "MAX_WORKERS", "MODELS")
    }
    try:
        base_config.PUZZLE_SOURCE = "nikoli"
        base_config.NIKOLI_LEVELS = levels
        base_config.NIKOLI_LIMIT = None
        base_config.PROTOCOLS = ["multi-step"]
        base_config.STRATEGIES = [train_config.TRAIN_STRATEGY_ID]
        base_config.RUNS_PER_STRATEGY = 1
        base_config.MAX_WORKERS = 1
        base_config.MODELS = [f"local:{checkpoint_name}"]

        model_string = f"local:{checkpoint_name}"
        scope = f"Nikoli-{'+'.join(levels)}" if levels else "full Nikoli-100"
        print(f"\n{'#'*70}\n# EVAL: {checkpoint_name} vs. {scope}\n{'#'*70}")
        t0 = time.time()
        results = run_swarm()
        elapsed = time.time() - t0
        print_summary(results)

        # Use the full recorded set for this model (see _load_all_parsed_runs),
        # not just run_swarm()'s return value, which excludes puzzles already
        # cached from a prior/interrupted run.
        rows = _load_all_parsed_runs(model_string, levels=levels)
        total = len(rows)
        solved = sum(1 for r in rows if r["solved"])
        correct = sum(1 for r in rows if r["correct_against_solution"])
        summary = {
            "checkpoint": checkpoint_name,
            "levels": levels,
            "num_puzzles": total,
            "solved": solved,
            "solve_rate": solved / total if total else None,
            "correct_against_solution": correct,
            "correct_rate": correct / total if total else None,
            "avg_turns": sum(r["total_turns"] for r in rows) / total if total else None,
            "avg_tokens": sum(
                r["total_prompt_tokens"] + r["total_completion_tokens"] for r in rows
            ) / total if total else None,
            "eval_seconds": elapsed,
        }
    finally:
        for k, v in saved.items():
            setattr(base_config, k, v)

    out_dir = train_config.checkpoint_dir(checkpoint_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"eval_summary_{'-'.join(levels)}.json" if levels else "eval_summary.json"
    (out_dir / out_name).write_text(json.dumps(summary, indent=2))
    print(f"Eval summary -> {out_dir / out_name}")
    return summary
