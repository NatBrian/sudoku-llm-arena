"""Merges a LoRA adapter into its base model's full weights.

Used to warm-start tier3 (full fine-tune + GRPO) from an SFT-tuned starting
point instead of the raw base checkpoint — tier3's GRPO_MAX_COMPLETION_LENGTH
is squeezed to 96 tokens for GPU memory, and an untrained base model rambles
well past that before ever emitting "MOVE:", so every rollout used to score
UNPARSEABLE_REWARD regardless of reward design. A warm-started model already
emits terse REASONING:/MOVE: completions comfortably under 96 tokens, at no
extra GPU memory cost (completion length is unchanged).

Usage:
    python -m src.train.merge_adapter --checkpoint qwen3-0.6b-tier1-recovery
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as train_config


def merge(checkpoint_name, out_name=None):
    ckpt_dir = train_config.checkpoint_dir(checkpoint_name)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    base_model_id = meta["base_model"]
    adapter_path = ckpt_dir / "adapter"

    print(f"Loading base model {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    out_name = out_name or f"{checkpoint_name}-merged"
    out_dir = train_config.checkpoint_dir(out_name) / "full"
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    meta_out = {
        "base_model": str(out_dir),
        "tier": "merged-full",
        "adapter": False,
        "merged_from": checkpoint_name,
    }
    (train_config.checkpoint_dir(out_name) / "meta.json").write_text(json.dumps(meta_out, indent=2))
    print(f"Saved merged full checkpoint -> {out_dir}")
    return out_name, str(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="LoRA adapter checkpoint name to merge")
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()
    name, path = merge(args.checkpoint, args.out_name)
    print(f"{name}\t{path}")
