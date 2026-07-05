"""End-to-end experiment runner for one base model: registers+evaluates tier0
(zero-shot), then trains+evaluates tier1 (LoRA SFT), tier2 (LoRA+GRPO, from
tier1's adapter), and tier3 (full fine-tune+GRPO, from the base checkpoint).

Each stage runs in its own subprocess — no shared Python process across
stages, so one stage's loaded model/optimizer state can't linger and eat GPU
memory into the next. Every stage's full stdout/stderr is logged to
output/experiments/<model_key>/<stage>.log, and running status (pass/fail,
timing) is written to output/experiments/<model_key>/status.json after every
stage, so progress can be checked without waiting for the whole run to finish.
A failed training stage skips its dependent stages (tier2 needs tier1) but
independent stages still run (tier3 doesn't depend on tier1/tier2).

Usage:
    python -m src.train.run_experiment --model qwen3-0.6b
"""
import argparse
import json
import os
import subprocess
import sys
import time

from . import config as train_config

EXPERIMENTS_DIR = train_config.CHECKPOINTS_DIR.parent / "experiments"

# This host has 8 GPUs shared with other users. device_map={"": 0} in
# sft_train.py/grpo_train.py/local_model.py puts model weights on GPU0, but
# without restricting visibility, transformers.Trainer's automatic n_gpu>1
# DataParallel wrapping still sees all 8 and can reach onto GPUs this job has
# no business touching. Every stage subprocess is pinned to GPU0 only.
_STAGE_ENV = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
)


def _save_status(model_key, status):
    exp_dir = EXPERIMENTS_DIR / model_key
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "status.json").write_text(json.dumps(status, indent=2))


def _run_stage(model_key, stage_name, argv, status):
    exp_dir = EXPERIMENTS_DIR / model_key
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / f"{stage_name}.log"
    print(f"\n=== [{model_key}] {stage_name} starting -> {log_path} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT, env=_STAGE_ENV)
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status["stages"][stage_name] = {
        "ok": ok, "returncode": proc.returncode, "seconds": round(elapsed, 1), "log": str(log_path),
    }
    _save_status(model_key, status)
    icon = "OK" if ok else "FAILED"
    print(f"=== [{model_key}] {stage_name} {icon} ({elapsed:.0f}s) ===", flush=True)
    return ok


def run_experiment(model_key):
    status = {"model_key": model_key, "started": time.time(), "stages": {}, "done": False}
    py = sys.executable

    tier0_ok = _run_stage(
        model_key, "tier0-register", [py, "-m", "src.train.tier0_register", "--model", model_key], status,
    )
    if tier0_ok:
        _run_stage(model_key, "tier0-eval",
                   [py, "-m", "src.train.eval_cli", "--checkpoint", f"{model_key}-tier0-zeroshot"], status)

    tier1_ok = _run_stage(model_key, "tier1-train",
                           [py, "-m", "src.train.sft_train", "--model", model_key], status)
    if tier1_ok:
        _run_stage(model_key, "tier1-eval",
                   [py, "-m", "src.train.eval_cli", "--checkpoint", f"{model_key}-tier1-lora-sft"], status)

    tier2_ok = False
    if tier1_ok:
        tier2_ok = _run_stage(
            model_key, "tier2-train",
            [py, "-m", "src.train.grpo_train", "--model", model_key, "--tier", "tier2-lora-grpo"], status,
        )
    if tier2_ok:
        _run_stage(model_key, "tier2-eval",
                   [py, "-m", "src.train.eval_cli", "--checkpoint", f"{model_key}-tier2-lora-grpo"], status)

    tier3_ok = _run_stage(
        model_key, "tier3-train",
        [py, "-m", "src.train.grpo_train", "--model", model_key, "--tier", "tier3-full-grpo"], status,
    )
    if tier3_ok:
        _run_stage(model_key, "tier3-eval",
                   [py, "-m", "src.train.eval_cli", "--checkpoint", f"{model_key}-tier3-full-grpo"], status)

    status["done"] = True
    status["finished"] = time.time()
    _save_status(model_key, status)
    print(f"\nExperiment for {model_key} done. Status -> {EXPERIMENTS_DIR / model_key / 'status.json'}")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(train_config.BASE_MODELS))
    args = parser.parse_args()
    run_experiment(args.model)
