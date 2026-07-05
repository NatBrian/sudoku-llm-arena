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
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)
