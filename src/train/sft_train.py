"""Tier1: LoRA SFT. Teaches a base chat model the MOVE:/BACKTRACK output
format and correct-move imitation from synthetic (non-Nikoli) puzzles.

Usage:
    python -m src.train.sft_train --model qwen2.5-1.5b
    python -m src.train.sft_train --model qwen3-0.6b
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from . import config as train_config
from .data import build_sft_examples


def train(model_key, out_name=None):
    base_model_id = train_config.BASE_MODELS[model_key]
    out_name = out_name or f"{model_key}-tier1-lora-sft"
    out_dir = train_config.checkpoint_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    print("Building SFT examples from synthetic (non-Nikoli) puzzles...")
    examples = build_sft_examples()
    print(f"  {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    peft_config = LoraConfig(
        r=train_config.LORA_R,
        lora_alpha=train_config.LORA_ALPHA,
        lora_dropout=train_config.LORA_DROPOUT,
        target_modules=train_config.LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=train_config.SFT_EPOCHS,
        per_device_train_batch_size=train_config.SFT_BATCH_SIZE,
        gradient_accumulation_steps=train_config.SFT_GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=train_config.SFT_LR,
        max_length=train_config.SFT_MAX_LENGTH,
        assistant_only_loss=True,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    meta = {"base_model": base_model_id, "tier": "tier1-lora-sft", "adapter": True}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved LoRA adapter -> {adapter_dir}")
    return out_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(train_config.BASE_MODELS))
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()
    train(args.model, args.out_name)
