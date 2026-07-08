"""CLI wrapper around evaluate.evaluate_checkpoint, so run_experiment.py can
run each eval in its own subprocess (GPU memory hygiene between stages).

Usage:
    python -m src.train.eval_cli --checkpoint qwen3-0.6b-tier1-lora-sft
"""
import argparse

from .evaluate import evaluate_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--levels", default=None,
                         help="comma-separated Nikoli levels to restrict to (e.g. 'hard'); omit for the full 100")
    args = parser.parse_args()
    levels = args.levels.split(",") if args.levels else None
    evaluate_checkpoint(args.checkpoint, levels=levels)
