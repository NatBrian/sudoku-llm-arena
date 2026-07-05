"""Tier2 (LoRA + GRPO, continuing from the tier1 adapter) and tier3 (full
fine-tune + GRPO from the base checkpoint). Reward comes from reward.py,
which reuses the real parser/validator so the trained policy is optimized
against the exact same rules loop.py grades it with at eval time.

Usage:
    python -m src.train.grpo_train --model qwen2.5-1.5b --tier tier2-lora-grpo
    python -m src.train.grpo_train --model qwen2.5-1.5b --tier tier3-full-grpo
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from . import config as train_config
from .data import build_grpo_examples
from .reward import reward_func


def train(model_key, tier, init_adapter=None, out_name=None, num_generations=None, max_completion_length=None):
    base_model_id = train_config.BASE_MODELS[model_key]
    out_name = out_name or f"{model_key}-{tier}"
    out_dir = train_config.checkpoint_dir(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )

    # CLI overrides (run_experiment.py uses these to try a headroom-dependent
    # relaxed rollout size before falling back to the known-safe squeeze
    # below) take priority over config/tier defaults, AND over tier3's
    # automatic squeeze — an explicit override means the caller already made
    # the headroom judgment call, so tier3 shouldn't silently re-clamp it.
    explicit_overrides = num_generations is not None or max_completion_length is not None
    num_generations = num_generations if num_generations is not None else train_config.GRPO_NUM_GENERATIONS
    max_completion_length = max_completion_length if max_completion_length is not None else train_config.GRPO_MAX_COMPLETION_LENGTH
    beta = train_config.GRPO_BETA

    if tier == "tier2-lora-grpo":
        init_adapter = init_adapter or f"{model_key}-tier1-lora-sft"
        adapter_path = train_config.checkpoint_dir(init_adapter) / "adapter"
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"tier2 continues from a tier1 LoRA adapter, but {adapter_path} "
                f"doesn't exist. Run `python -m src.train.sft_train --model {model_key}` first."
            )
        print(f"Loading tier1 adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        lr = train_config.GRPO_LR_LORA
        optim = "adamw_torch"  # LoRA-only optimizer state is small; no need for 8bit
    elif tier == "tier3-full-grpo":
        lr = train_config.GRPO_LR_FULL
        # adamw_bnb_8bit + shorter/fewer rollouts still OOM'd on the 1.5B
        # model at near-identical memory usage — turns out optimizer/rollout
        # size wasn't the dominant cost. trl's GRPOTrainer instantiates a
        # FULL SECOND COPY of the model as a frozen KL-reference whenever
        # beta != 0 and there's no LoRA adapter to fall back on (see
        # grpo_trainer.py: `if self.beta == 0.0: ref_model = None`, else
        # `create_model_from_path(...)` for full fine-tunes) — an extra
        # ~3GB+ for a 1.5B model, on top of the policy model itself. tier2
        # never pays this because PEFT can derive the reference by disabling
        # the adapter instead of loading a second copy. Dropping beta to 0
        # for tier3 only skips the reference model entirely.
        optim = "adafactor"
        # qwen3-1.7b's full-FT footprint left only ~900MB free at the first
        # optimizer step even with beta=0 (no ref model) — tighter than the
        # 1.5B model's margin, so trim rollout memory further for tier3 here,
        # unless the caller (run_experiment.py) already picked explicit
        # values based on a live headroom check.
        if not explicit_overrides:
            num_generations = min(num_generations, 2)
            max_completion_length = min(max_completion_length, 96)
        beta = 0.0
    else:
        raise ValueError(f"unknown tier {tier!r}, expected tier2-lora-grpo or tier3-full-grpo")

    print("Building GRPO rollout-start examples from synthetic (non-Nikoli) puzzles...")
    examples = build_grpo_examples()
    print(f"  {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    grpo_config = GRPOConfig(
        output_dir=str(out_dir / "trainer"),
        learning_rate=lr,
        optim=optim,
        per_device_train_batch_size=train_config.GRPO_BATCH_SIZE,
        gradient_accumulation_steps=train_config.GRPO_GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=beta,
        max_steps=train_config.GRPO_STEPS,
        # Qwen3's chat template defaults to a <think>...</think> block; force
        # it off so completions stay on the plain REASONING:/MOVE: format
        # tier1 SFT trained (see data.py's matching per-example kwarg for
        # SFTTrainer, and local_model.py for inference-time parity). Ignored
        # by templates (e.g. Qwen2.5's) that don't reference this variable.
        chat_template_kwargs={"enable_thinking": False},
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    (out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))

    if tier == "tier2-lora-grpo":
        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        meta = {
            "base_model": base_model_id,
            "tier": tier,
            "adapter": True,
            "init_adapter": init_adapter,
        }
    else:
        full_dir = out_dir / "full"
        trainer.save_model(str(full_dir))
        tokenizer.save_pretrained(str(full_dir))
        meta = {"base_model": str(full_dir), "tier": tier, "adapter": False}

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {tier} checkpoint -> {out_dir}")
    return out_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(train_config.BASE_MODELS))
    parser.add_argument("--tier", required=True, choices=["tier2-lora-grpo", "tier3-full-grpo"])
    parser.add_argument("--init-adapter", default=None, help="tier1 checkpoint name to continue from (tier2 only)")
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--num-generations", type=int, default=None, help="override GRPO_NUM_GENERATIONS (bypasses tier3's automatic squeeze)")
    parser.add_argument("--max-completion-length", type=int, default=None, help="override GRPO_MAX_COMPLETION_LENGTH (bypasses tier3's automatic squeeze)")
    args = parser.parse_args()
    train(args.model, args.tier, args.init_adapter, args.out_name, args.num_generations, args.max_completion_length)
