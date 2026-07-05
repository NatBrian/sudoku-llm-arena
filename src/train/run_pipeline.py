"""Orchestrates tiers 1-3 for one or more base models end to end:

    python -m src.train.run_pipeline --model qwen3-0.6b qwen2.5-1.5b

Runs, per model: tier1 LoRA SFT -> tier2 LoRA+GRPO (continues from tier1's
adapter) -> tier3 full fine-tune+GRPO (independent, from the base checkpoint).
Tier0 (zero-shot) needs no training — just add "<base-model-hf-id>" or the
litellm-style local id straight to src/config.py's MODELS list and run it
through run.py like any frontier model.

After training, evaluate all tiers against the real Nikoli-100 puzzles (never
used in training — see src/train/synth.py) by adding "local:<checkpoint-name>"
entries to src/config.py's MODELS and running run.py as usual — set
config.MAX_WORKERS = 1 first (single GPU serving all local checkpoints), or
just call src.train.evaluate.evaluate_checkpoint() directly.
"""
import argparse

from . import config as train_config
from .sft_train import train as train_sft
from .grpo_train import train as train_grpo


def run_all_tiers(model_key):
    print(f"\n=== {model_key}: tier1-lora-sft ===")
    tier1_name = train_sft(model_key)

    print(f"\n=== {model_key}: tier2-lora-grpo ===")
    tier2_name = train_grpo(model_key, "tier2-lora-grpo", init_adapter=tier1_name)

    print(f"\n=== {model_key}: tier3-full-grpo ===")
    tier3_name = train_grpo(model_key, "tier3-full-grpo")

    print(f"\n{model_key} done. Checkpoints: {tier1_name}, {tier2_name}, {tier3_name}")
    print("Add these to src/config.py MODELS as:")
    print(f'  "local:{tier1_name}"')
    print(f'  "local:{tier2_name}"')
    print(f'  "local:{tier3_name}"')
    return tier1_name, tier2_name, tier3_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", nargs="+", required=True, choices=list(train_config.BASE_MODELS)
    )
    args = parser.parse_args()
    for model_key in args.model:
        run_all_tiers(model_key)
