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
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from . import config as train_config
from .data import build_sft_examples


def _find_stage(stage_name):
    for stage in train_config.TRAIN_CURRICULUM_STAGES:
        if stage["name"] == stage_name:
            return stage
    raise ValueError(f"unknown curriculum stage {stage_name!r}, expected one of "
                      f"{[s['name'] for s in train_config.TRAIN_CURRICULUM_STAGES]}")


def train(model_key, out_name=None, init_adapter=None, stage=None, examples_jsonl=None, epochs_override=None):
    base_model_id = train_config.BASE_MODELS[model_key]
    out_name = out_name or f"{model_key}-tier1-lora-sft"
    out_dir = train_config.checkpoint_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    box_width, box_height, difficulty_mix, epochs = 3, 3, None, train_config.SFT_EPOCHS
    if stage is not None:
        stage_cfg = _find_stage(stage)
        box_width, box_height = stage_cfg["box_width"], stage_cfg["box_height"]
        difficulty_mix, epochs = stage_cfg["mix"], stage_cfg["epochs"]
    if epochs_override is not None:
        epochs = epochs_override

    print(f"Loading base model {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    if examples_jsonl:
        print(f"Loading SFT examples from {examples_jsonl}...")
        examples = [json.loads(line) for line in open(examples_jsonl) if line.strip()]
    else:
        print("Building SFT examples from synthetic (non-Nikoli) puzzles...")
        examples = build_sft_examples(box_width=box_width, box_height=box_height, difficulty_mix=difficulty_mix)
    print(f"  {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    if init_adapter:
        # Curriculum stage continuing a previous stage's adapter forward,
        # rather than starting a fresh LoRA (mirrors grpo_train.py's tier2
        # pattern of loading tier1's adapter).
        adapter_path = train_config.checkpoint_dir(init_adapter) / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(f"--init-adapter {init_adapter} has no adapter at {adapter_path}")
        print(f"Continuing from adapter {adapter_path}...")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=train_config.LORA_R,
            lora_alpha=train_config.LORA_ALPHA,
            lora_dropout=train_config.LORA_DROPOUT,
            target_modules=train_config.LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )

    sft_config = SFTConfig(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=epochs,
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
        peft_config=None if init_adapter else peft_config,
    )
    trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    if examples_jsonl:
        tier_label = "tier1-recovery"
    elif stage:
        tier_label = "tier1-curriculum"
    else:
        tier_label = "tier1-lora-sft"
    meta = {
        "base_model": base_model_id,
        "tier": tier_label,
        "adapter": True,
        "stage": stage,
        "init_adapter": init_adapter,
        "examples_jsonl": examples_jsonl,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved LoRA adapter -> {adapter_dir}")
    return out_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(train_config.BASE_MODELS))
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--init-adapter", default=None, help="checkpoint name of a prior LoRA adapter to continue from (curriculum stages)")
    parser.add_argument("--stage", default=None, choices=[s["name"] for s in train_config.TRAIN_CURRICULUM_STAGES],
                         help="curriculum stage to train on; omit for the canonical 9x9 tier1 pass")
    parser.add_argument("--examples-jsonl", default=None,
                         help="load SFT examples from this JSONL file instead of build_sft_examples() (e.g. self_rollout.py's recovery set)")
    parser.add_argument("--epochs", type=int, default=None, help="override epoch count (default: stage's or SFT_EPOCHS)")
    args = parser.parse_args()
    train(args.model, args.out_name, args.init_adapter, args.stage, args.examples_jsonl, args.epochs)
