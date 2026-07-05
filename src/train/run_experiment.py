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

# tier3's known-safe rollout settings (see grpo_train.py's automatic squeeze)
# — used as the fallback if a relaxed attempt below OOMs.
TIER3_SAFE_NUM_GENERATIONS = 2
TIER3_SAFE_MAX_COMPLETION_LENGTH = 96
# Only attempt relaxed rollout settings if GPU0 has comfortably more free
# memory than what the safe squeeze needs — this host is shared and another
# process can hold >100GB of it at any time (see handoff's known pitfalls).
TIER3_RELAX_FREE_MIB_THRESHOLD = 20000
TIER3_RELAXED_NUM_GENERATIONS = 4
TIER3_RELAXED_MAX_COMPLETION_LENGTH = 160


def _gpu0_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception as e:
        print(f"nvidia-smi headroom check failed ({e}); assuming no extra headroom.")
        return 0


def _log_mentions_oom(log_path):
    try:
        text = log_path.read_text(errors="ignore").lower()
        return "out of memory" in text or "cuda oom" in text
    except OSError:
        return False


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

    prev_adapter = None
    curriculum_ok = True
    for stage in train_config.TRAIN_CURRICULUM_STAGES:
        stage_out = f"{model_key}-tier1-{stage['name']}"
        argv = [py, "-m", "src.train.sft_train", "--model", model_key,
                "--stage", stage["name"], "--out-name", stage_out]
        if prev_adapter:
            argv += ["--init-adapter", prev_adapter]
        curriculum_ok = _run_stage(model_key, f"tier1-{stage['name']}", argv, status)
        if not curriculum_ok:
            break
        prev_adapter = stage_out

    tier1_argv = [py, "-m", "src.train.sft_train", "--model", model_key]
    if curriculum_ok and prev_adapter:
        tier1_argv += ["--init-adapter", prev_adapter]
    tier1_ok = curriculum_ok and _run_stage(model_key, "tier1-train", tier1_argv, status)
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

    tier3_argv = [py, "-m", "src.train.grpo_train", "--model", model_key, "--tier", "tier3-full-grpo"]
    free_mib = _gpu0_free_mib()
    tried_relaxed = free_mib > TIER3_RELAX_FREE_MIB_THRESHOLD
    if tried_relaxed:
        print(f"GPU0 has {free_mib}MiB free (> {TIER3_RELAX_FREE_MIB_THRESHOLD}MiB threshold); "
              f"attempting relaxed tier3 rollout settings first.")
        tier3_ok = _run_stage(model_key, "tier3-train", tier3_argv + [
            "--num-generations", str(TIER3_RELAXED_NUM_GENERATIONS),
            "--max-completion-length", str(TIER3_RELAXED_MAX_COMPLETION_LENGTH),
        ], status)
        if not tier3_ok and _log_mentions_oom(EXPERIMENTS_DIR / model_key / "tier3-train.log"):
            print("Relaxed tier3 settings OOM'd; retrying with the known-safe squeezed settings.")
            tier3_ok = _run_stage(model_key, "tier3-train", tier3_argv + [
                "--num-generations", str(TIER3_SAFE_NUM_GENERATIONS),
                "--max-completion-length", str(TIER3_SAFE_MAX_COMPLETION_LENGTH),
            ], status)
    else:
        print(f"GPU0 has {free_mib}MiB free (<= {TIER3_RELAX_FREE_MIB_THRESHOLD}MiB threshold); "
              f"using tier3's default squeezed rollout settings.")
        tier3_ok = _run_stage(model_key, "tier3-train", tier3_argv, status)
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
