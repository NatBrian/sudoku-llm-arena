# Sudoku Arena

LLM Sudoku solving tournament — tests how well frontier models solve real Sudoku puzzles from Sakana AI's Sudoku-Bench, with animated visualizations of each solve attempt. Also trains small local models (Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct) to see how much post-training closes the gap to frontier — see [Training small models](#training-small-models-srctrain).

## Puzzle source: Sakana AI's Sudoku-Bench

Puzzles come from `SakanaAI/sudoku-bench-nikoli` on Hugging Face (100 real
hand-made 9x9 puzzles by Nikoli, CC-BY-4.0). Sakana's own eval code and the
larger CTC/Logic-Masters variant sets were pulled from GitHub in 2026-05; this
Nikoli subset is the only piece still hosted, so it's what this repo pulls
and caches locally (`data/nikoli_100.json`).

Difficulty isn't a dedicated dataset column — Nikoli's puzzle IDs encode it as
a trailing letter, which `src/nikoli.py` classifies into three **levels**:

| Level | Count |
|---|---|
| easy | 12 |
| medium | 35 |
| hard | 51 |
| other (seasonal/special) | 2 |

Pick which level(s) to run in `src/config.py`:

```python
NIKOLI_LEVELS = ["easy"]   # or ["easy", "medium", "hard"], or None for all
NIKOLI_LIMIT = 5           # cap puzzles per run; None for all matching puzzles
```

Set `PUZZLE_SOURCE = "generator"` to fall back to the original py-sudoku
procedural generator (unlimited synthetic 4x4/6x6/9x9/evil puzzles) instead.

## Eval protocols

Mirrors Sakana's own two Sudoku-Bench modes — pick one or both in `config.PROTOCOLS`:

- **multi-step** — model places one digit per turn given the updated board
  each time; the run halts on the first invalid placement. This repo layers
  5 prompting personas (strategies) on top for richer, more varied animations.
- **single-shot** — model gets the puzzle once and must return the complete
  solved grid in a single response, matching Sakana's headline solve-rate metric.

## Models

Model calls go through [litellm](https://docs.litellm.ai/docs/providers), so
any provider it supports works — just add model ids to `config.MODELS`:

```python
MODELS = [
    "gpt-5",
    "anthropic/claude-opus-4-8",
    "gemini/gemini-2.5-pro",
    "deepseek/deepseek-chat",
]
```

Each provider reads its API key from its usual env var (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, ...) — litellm
handles that lookup itself.

## Strategies (multi-step only)

| ID | Name | Approach |
|---|---|---|
| s1-direct | The Impulsive | No constraints — just fill any cell |
| s2-naked-singles | The Methodical | Only fill when exactly one candidate exists |
| s3-hidden-singles | The Scanner | Find numbers that fit only one cell |
| s4-full-logic | The Logician | Full logic chain (naked → hidden → cross-hatch) |
| s5-guess-verify | The Gambler | Logic + guess when stuck, backtrack on conflict |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # or whichever provider(s) you configured
```

## Usage

```bash
python run.py
```

Run individual steps:

```bash
python -c "from src.swarm import run_swarm; run_swarm()"
python -c "from src.visualize.gif import render_all_runs; render_all_runs()"
python -c "from src.visualize.web import generate_web_page; generate_web_page()"
python -c "from src.visualize.report import generate_report; generate_report()"
```

## Output

- `output/gifs/` — Animated GIF per run (one frame per move for multi-step, before/after for single-shot)
- `output/web/` — Interactive HTML viewer with puzzle/model/strategy filters and a playback timeline
- `output/report/` — Summary charts (including solve-rate-by-model) and markdown report
- `data/parsed/` — Run data in JSON format

## Training small models (`src/train/`)

Beyond evaluating frontier models, `src/train/` trains small local models
(Qwen3-0.6B, Qwen3-1.7B, Qwen2.5-1.5B-Instruct) to see how far cheap
post-training closes the gap on Sakana's own Nikoli-100 puzzles. Four tiers,
easiest to hardest:

| Tier | Method | What it does |
|---|---|---|
| tier0-zeroshot | none | Run the base model as-is through the normal harness — no training |
| tier1-lora-sft | LoRA supervised fine-tune | Teach `grid state -> correct move` on synthetic puzzles |
| tier2-lora-grpo | LoRA + GRPO (RL) | Continue tier1's adapter with reward-driven RL (reward = valid move / solve bonus) |
| tier3-full-grpo | Full fine-tune + GRPO | Same RL, but unfreezes all weights from the base checkpoint instead of a LoRA adapter |

**Data provenance:** training and eval data are from **disjoint sources**, not
just disjoint puzzle instances. All 100 real Nikoli puzzles are reserved for
eval only — none are ever used in training, not even in a held-out split.
Training puzzles are procedurally generated instead (`src/train/synth.py`,
via the same py-sudoku generator `PUZZLE_SOURCE="generator"` uses), across an
easy/medium/hard difficulty mix that approximates the real eval set's split
(12/35/51) so training difficulty isn't skewed vs. what's tested. This
matters because training on Nikoli-derived puzzles — even a held-out split —
would still teach Nikoli-specific structure that frontier models, evaluated
zero-shot, never get exposed to; that would make any tier1-3 vs. frontier
comparison an apples-to-oranges "trained on the test distribution" result.
Puzzle counts, difficulty mix, and tier hyperparameters all live in
`src/train/config.py`.

**Protocol:** training and inference both use the `s1-direct` single-move
strategy (the same multi-step protocol frontier models are graded on), so
trained checkpoints are evaluated through the identical `loop.py`/`parser.py`/
`validator.py` harness as every other model — no separate grading path.

### Results (first full tier0-3 pass, all 3 models)

None of the 12 checkpoints below (3 models x 4 tiers) solved a real Nikoli
puzzle (`solved=0/100` across the board) — consistent with frontier models
also solving only a small fraction of this set. `avg_turns` is a proxy for
"how long the model stays valid before an illegal move or backtrack
exhaustion," not solving competence:

| tier | qwen3-0.6b | qwen2.5-1.5b (Instruct) | qwen3-1.7b |
|---|---|---|---|
| tier0 zero-shot | 1.00 | 5.23 | 5.54 |
| tier1 LoRA SFT | 7.43 | 5.74 | 7.72 |
| tier2 LoRA+GRPO | 12.17 | 5.98 | 10.42 |
| tier3 full+GRPO | 1.00 (collapsed) | 5.16 | 5.55 (collapsed) |

Key finding: both raw/base Qwen3 models (0.6B and 1.7B) show real learning
tier0→tier2 (turns climb steadily as SFT then GRPO build on each other), but
tier3 — full fine-tune + GRPO from cold, with no SFT warm-start — collapses
back to near zero-shot behavior at both sizes. GRPO's reward signal alone
doesn't carry enough learning signal to bootstrap the output format from
scratch; it needs SFT to teach the format first. qwen2.5-1.5b (already
Instruct-tuned) never collapses this way but also never improves much —
it already knows chat format going in, so there's less new ground for
SFT/GRPO to cover. See `HANDOFF_TRAINING.md` (untracked, local-only) for the
detailed diagnosis and a prioritized plan for a follow-up training pass.

### Train

```bash
# one tier at a time
python -m src.train.sft_train --model qwen3-0.6b
python -m src.train.grpo_train --model qwen3-0.6b --tier tier2-lora-grpo
python -m src.train.grpo_train --model qwen3-0.6b --tier tier3-full-grpo

# or all three tiers back to back, for one or more models
python -m src.train.run_pipeline --model qwen3-0.6b qwen2.5-1.5b
```

Checkpoints land in `output/checkpoints/<name>/` (gitignored — large,
regenerable), each with a `meta.json` (base model/adapter info) and a
`train_log.json` dump of the trainer's full step-by-step metrics (loss/
accuracy for SFT; reward/KL/entropy for GRPO), for debugging training runs
after the fact.

### Full experiment: train + evaluate every tier, unattended

```bash
python -m src.train.run_experiment --model qwen3-0.6b
```

Registers tier0 (zero-shot base model), then trains and evaluates tier1,
tier2, tier3 in sequence — each stage in its own subprocess (clean GPU state
between stages), evaluated against the full real Nikoli-100 set right after
training. A failed stage skips only its dependents (tier2 needs tier1's
adapter; tier3 doesn't depend on anything and still runs). Progress and
timing land in `output/experiments/<model>/status.json`, full logs per stage
in `output/experiments/<model>/<stage>.log`, and each checkpoint's
`eval_summary.json` (solve rate, correct rate, avg turns/tokens) sits
alongside its `train_log.json` for a tier-by-tier comparison — full per-move
trajectories (reasoning, parsed move, validation errors) for every eval run
land in `data/parsed/` same as any other model, via the normal harness.

### Evaluate trained checkpoints against frontier models

Add each checkpoint to `src/config.py`'s `MODELS` list with a `local:` prefix,
alongside any litellm model ids, and run as usual:

```python
MODELS = [
    "gpt-5",
    "anthropic/claude-opus-4-8",
    "local:qwen3-0.6b-tier1-lora-sft",
    "local:qwen3-0.6b-tier2-lora-grpo",
    "local:qwen3-0.6b-tier3-full-grpo",
]
MAX_WORKERS = 1  # required — all local: checkpoints share one GPU
```

```bash
python run.py
```

### Setup

Training needs a separate, heavier dependency set than plain evaluation
(torch/transformers/trl/peft/accelerate/bitsandbytes — see `requirements.txt`).
A single CUDA GPU is required; hyperparameters assume one GPU serving one
checkpoint at a time.
