"""Registers the raw (untrained) base model under the "local:<name>" checkpoint
convention, so tier0 (zero-shot baseline) can be served and evaluated through
the exact same code path as trained tiers, with no training step.

Usage:
    python -m src.train.tier0_register --model qwen3-0.6b
"""
import argparse

from . import config as train_config
from .evaluate import register_tier0_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(train_config.BASE_MODELS))
    args = parser.parse_args()
    name = register_tier0_checkpoint(args.model)
    print(f"Registered tier0 checkpoint -> {name}")
