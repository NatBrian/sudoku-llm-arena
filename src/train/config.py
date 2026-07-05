"""Config for the small-model training pipeline (tiers 0-3, see README).

Kept separate from src/config.py since it's a distinct concern (training
hyperparams vs. eval/swarm settings) and most users of this repo will never
touch it.
"""
import torch

from .. import config as base_config

# This host's cuDNN build (9.19, picked up via LD_LIBRARY_PATH) doesn't match
# what this torch wheel expects (bundles 9.10.2) — the cuDNN backend of
# scaled_dot_product_attention throws CUDNN_STATUS_NOT_INITIALIZED on any
# bf16 forward pass otherwise. Disabling just that backend (not cuDNN
# globally) falls back to the flash/efficient/math SDPA kernels, which work
# fine. Runtime-only flag, doesn't touch shared system config.
torch.backends.cuda.enable_cudnn_sdp(False)

# --- Base models to fine-tune -----------------------------------------------
BASE_MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

# The strategy/prompt format the trained models are taught and evaluated on.
# Fixed to one strategy (not the full STRATEGIES list in src/config.py) so
# train and eval prompts match exactly.
TRAIN_STRATEGY_ID = "s1-direct"

# --- Tiers -------------------------------------------------------------------
# tier0-zeroshot: no training, just run the base model through the existing
#   swarm/loop harness like any other litellm/local model. Nothing to do here.
# tier1-lora-sft: LoRA adapter, supervised on (grid_state -> correct move).
# tier2-lora-grpo: start from the tier1 LoRA adapter, continue with GRPO
#   (reward = valid move / solved bonus) via reward.py.
# tier3-full-grpo: full fine-tune (no adapter) then GRPO on top, starting
#   from the base checkpoint (not tier1's adapter, since it's a separate
#   unfrozen-weights run).
TIERS = ["tier1-lora-sft", "tier2-lora-grpo", "tier3-full-grpo"]

CHECKPOINTS_DIR = base_config.OUTPUT_DIR / "checkpoints"
TRAIN_DATA_DIR = base_config.DATA_DIR / "train"

# --- Synthetic training data --------------------------------------------------
# Training puzzles are procedurally generated (py-sudoku), NOT derived from
# Sakana's real Nikoli-100 set — training on Nikoli-derived puzzles (even a
# held-out split) would still teach Nikoli-specific structure the eval
# doesn't test frontier models on. All 100 real Nikoli puzzles are reserved
# for eval only (see src/train/synth.py). label -> (py-sudoku difficulty
# float, puzzle count); mix approximates the real eval set's easy/medium/hard
# split (12/35/51) so training difficulty isn't skewed vs. what's tested.
TRAIN_DIFFICULTY_MIX = {
    "easy": (0.4, 40),
    "medium": (0.55, 110),
    "hard": (0.7, 150),
}
SNAPSHOTS_PER_PUZZLE = 12  # random partial-fill states sampled per puzzle
DATA_SEED = 0

# --- LoRA ----------------------------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# --- SFT (tier1) ---------------------------------------------------------------
# Per-device batch kept small + gradient accumulation used to reach the same
# effective batch size (8) at a much lower activation-memory footprint — this
# host's GPU0 has a large chunk reserved by hold_gpus.py (see run_experiment.py),
# leaving well under what the naive batch_size=8/no-accumulation setup needs.
SFT_EPOCHS = 3
SFT_LR = 1e-4
SFT_BATCH_SIZE = 2
SFT_GRAD_ACCUM_STEPS = 4  # effective batch size 8
SFT_MAX_LENGTH = 1024

# --- GRPO (tier2/tier3) ---------------------------------------------------------
GRPO_STEPS = 300
GRPO_LR_LORA = 5e-6
GRPO_LR_FULL = 1e-6
# 1 prompt/device-step x 8 accumulation = effective 8 prompts/update (same as
# before), but peak memory only ever holds 1 prompt's num_generations rollouts
# at a time instead of 8's worth simultaneously.
GRPO_BATCH_SIZE = 1
GRPO_GRAD_ACCUM_STEPS = 8
GRPO_NUM_GENERATIONS = 8
# No prompt-length knob in this trl version's GRPOConfig (older trl had
# max_prompt_length; this one doesn't) — only the completion side is capped.
GRPO_MAX_COMPLETION_LENGTH = 256
GRPO_BETA = 0.02


def checkpoint_dir(name):
    return CHECKPOINTS_DIR / name
