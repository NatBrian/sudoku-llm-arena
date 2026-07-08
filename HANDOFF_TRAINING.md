# Handoff: improve qwen3-0.6b / qwen3-1.7b sudoku training

## Goal

Re-run the tier0-3 training pipeline (`src/train/`) for **qwen3-0.6b** and
**qwen3-1.7b** with the changes below, aiming for actual improvement in
solve rate / puzzle-solving competence — not just format compliance. The
first full pass (already done, see baseline below) trained successfully on
all 4 tiers for both models but **0/100 puzzles solved in every single
checkpoint**. This handoff exists to fix that, not to just re-run the same
recipe for longer (diminishing returns already visible — see diagnosis).

`qwen2.5-1.5b` is a separate model family (Instruct-tuned Qwen2.5, not
Qwen3) already fully trained/evaluated in a prior pass — out of scope here
unless you want a 3-way comparison for free (its baseline results are also
in `output/checkpoints/qwen2.5-1.5b-tier*/eval_summary.json`).

## Baseline (already run, do not need to repeat as-is)

All 100 real Nikoli puzzles, `solved=0/100` for every row below. `avg_turns`
is a proxy for "how long the model stays valid before an illegal move or
backtrack-exhaustion" — NOT solving competence. It went up tier0→tier2 (real
learning of output format + short-horizon validity) then collapsed back to
~zero-shot at tier3 for both model sizes:

| tier | qwen3-0.6b avg_turns | qwen3-1.7b avg_turns |
|---|---|---|
| tier0 zero-shot | 1.00 | 5.54 |
| tier1 LoRA SFT | 7.43 | 7.72 |
| tier2 LoRA+GRPO | 12.17 | 10.42 |
| tier3 full+GRPO | 1.00 (collapsed) | 5.55 (collapsed) |

Checkpoints/logs from this baseline run are still on disk at
`output/checkpoints/qwen3-{0.6b,1.7b}-tier{0,1,2,3}*/` and
`output/experiments/qwen3-{0.6b,1.7b}/` — don't delete them, they're the
control group for judging whether your changes actually helped.

## Diagnosis (why baseline plateaued at 0 solves)

IMPORTANT: MUST NOT TRAIN USING SAKANA SUDOKU AI DATASET nor NIKOLI DATASET

1. **Scale gap.** 0.6-1.7B params vs. frontier-model scale. Real 9x9 sudoku
   via text requires tracking 81 cells + all constraints correctly across
   dozens of turns purely in-context. Even frontier LLMs solve only ~1% of
   real Nikoli 9x9 puzzles (see project memory / earlier sudoku-llm-arena
   benchmark runs) — this was never a low bar.
2. **Training data volume mismatch.** 300 synthetic puzzles, 3 SFT epochs,
   300 GRPO steps teaches output *format*, not a solving *algorithm*.
3. **GRPO reward is sparse** (`src/train/reward.py`): valid-move-or-not +
   solved bonus, all-or-nothing. Teaches "don't emit an illegal move," not
   "here's the deduction that proves this cell is 7." tier3's cold-start
   collapse (GRPO with no SFT warm-start) shows this reward alone carries
   ~zero learning signal for actual logic.
4. **Reasoning traces exist but are mostly vacuous — this is the biggest
   concrete finding, and the best lever to pull first.** `src/train/data.py`
   already generates a `REASONING: ...` line before every `MOVE:` in SFT
   training data (`_reasoning_for()`), and `src/validator.py` already has
   `find_naked_singles()` to justify the easy case. **But that's the only
   solving technique implemented.** Every cell that isn't a naked single
   (i.e. most cells in the 0.55/0.7-difficulty medium/hard puzzles that make
   up 260/300 of the training set — see `TRAIN_DIFFICULTY_MIX` in
   `src/train/config.py`) gets this generic, content-free filler instead:

   ```
   "Placing {value} at R{r}C{c} does not conflict with its row, column, or
   box, and matches the puzzle's unique solution."
   ```

   That sentence is true but teaches nothing — it doesn't say *why* {value}
   and not some other digit. The model is being SFT-trained to imitate
   reasoning-shaped text that isn't actually reasoning, for the majority of
   its training examples. This is very likely a major reason tier1/tier2
   learn "format" but not "logic."

## Recommended changes, in priority order

### 1. Extend `_reasoning_for()` with real solving techniques (highest leverage, cheapest)

`src/train/data.py:_reasoning_for()` and `src/validator.py` currently only
detect naked singles. Add detectors for at least:
- **Hidden singles** (a digit that can only go in one cell within a row/col/box,
  even if that cell has other candidates too)
- **Naked pairs / pointing pairs / box-line reduction** if time allows

Each new technique needs (a) a detector function in `validator.py` mirroring
`find_naked_singles()`'s signature, and (b) a branch in `_reasoning_for()`
producing a real justification (e.g. "7 can only go in R3C4 within this box
— every other cell in the box already has 7 excluded by its row/column").
Cells not covered by any implemented technique should probably be **excluded
from training** (or down-weighted) rather than given the vacuous filler —
training on "reasoning" that doesn't explain the answer is worse than not
training on it at all.

### 2. Curriculum: 4x4/6x6 → 9x9

`src/train/synth.py`'s `generate_puzzle` already goes through
`src/puzzles.py` (py-sudoku), which supports arbitrary `(box_width,
box_height)` — a 4x4 board is `(2,2)`, 6x6 is `(3,2)` or `(2,3)`. Add a
curriculum mix to `train_config.TRAIN_DIFFICULTY_MIX` (or a new config var)
that trains on a progression of board sizes, so the model can learn full
constraint-propagation logic on state spaces small enough to actually hold
in its 81-cell-analog working set, before being asked to transfer that to
9x9. Watch out: `src/strategies.py:build_prompt` and the eval harness
(`loop.py`) assume 9x9-shaped prompts in places — check that non-9x9 puzzles
render/parse correctly through the same prompt-building path before training
on them, or the model will learn a format eval never uses.

### 3. Denser GRPO reward (`src/train/reward.py`)

Current reward is `{-1, +1, +2}` (unparseable/invalid/stuck = -1, valid = +1,
valid+matches-solution = +2). Consider adding partial credit for: fewer
remaining candidates after the move (progress toward solved), or a bonus
tied to whether the move corresponds to a detected technique (ties in nicely
with #1 — reward the model more for placing digits it can actually justify).

### 4. If GPU memory allows, increase GRPO rollout diversity

`GRPO_NUM_GENERATIONS` had to be squeezed to 2 (tier3) / 8 (tier2) and
`GRPO_MAX_COMPLETION_LENGTH` to 96 (tier3, qwen3-1.7b) purely due to OOM on
this shared host, not because that's the right number for learning quality.
Small group sizes weaken the group-relative-advantage estimate GRPO's
gradient depends on. If you get a host with more free VRAM, raise these back
up in `src/train/config.py` (`GRPO_NUM_GENERATIONS`, `GRPO_MAX_COMPLETION_LENGTH`)
and the tier-specific overrides in `src/train/grpo_train.py`.

## Known operational pitfalls (read before running — all previously hit and fixed)

- **This host has 8 shared GPUs; another process holds ~126GB of GPU0's
  ~140GB.** Every training/eval subprocess MUST run with
  `CUDA_VISIBLE_DEVICES=0` restricted (already done via `_STAGE_ENV` in
  `src/train/run_experiment.py`) — otherwise `transformers.Trainer`
  auto-wraps in `DataParallel` across all 8 GPUs, touching other users' jobs.
- **tier3 (full fine-tune + GRPO) needs `beta=0.0`.** `trl`'s `GRPOTrainer`
  loads a full second copy of the model as a frozen KL-reference whenever
  `beta != 0` and there's no LoRA adapter to derive it from cheaply — an
  extra ~3GB+ that WILL OOM on this host's headroom. Already set in
  `grpo_train.py`'s `tier3-full-grpo` branch; don't remove it.
- **qwen3-1.7b's tier3 still needed extra headroom** beyond what worked for
  qwen3-0.6b/qwen2.5-1.5b: `num_generations` cut to 2, `max_completion_length`
  to 96, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (now in
  `_STAGE_ENV`). If you add a bigger model or restore larger rollout sizes,
  expect to retune these per-model — check `nvidia-smi` free memory on GPU0
  before launching, and watch the first ~5 GRPO steps for OOM (it fails fast,
  at the first `optimizer.step()`, if it's going to fail at all).
- **`evaluate_checkpoint()` rebuilds its summary from
  `data/parsed/*_{slug}_*.json` on disk**, not from `run_swarm()`'s return
  value — the latter excludes cached/deduped puzzles and will silently
  produce a hollow summary on a partial re-run. Don't "fix" this back.
- **Training data is 100% synthetic (`src/train/synth.py`, py-sudoku
  procedural generation) — never derived from the real Nikoli-100 set.**
  All 100 real Nikoli puzzles are reserved for eval only. This was a
  deliberate fix for a data-leakage bug in an earlier design (train/eval
  split of the *same* Nikoli puzzles) — do not reintroduce Nikoli puzzles
  into the training set, even a held-out split of them, or eval results
  become meaningless (the model would be tested on the same *kind* of
  puzzle distribution it trained on, unlike frontier models being compared
  against).
- **Run each stage as its own subprocess** (already how `run_experiment.py`
  works) — don't refactor to a single long-lived process across tiers, GPU
  memory from one stage lingers into the next otherwise.
- **Launch training via `Bash(..., run_in_background: true)`, not
  `nohup ... &`.** Only the former is harness-tracked and reliably triggers
  a completion notification; a detached process requires manual polling and
  was a mistake caught earlier in this project.

## How to run

```bash
python -m src.train.run_experiment --model qwen3-0.6b
python -m src.train.run_experiment --model qwen3-1.7b
```

Each does tier0-register → tier0-eval → tier1-train → tier1-eval → (if ok)
tier2-train → tier2-eval → tier3-train → tier3-eval, writing
`output/experiments/<model>/status.json` (stage-by-stage pass/fail + timing)
and per-stage `.log` files as it goes — check those instead of waiting for
the whole run. Each eval writes `output/checkpoints/<checkpoint>/eval_summary.json`
with `solved`/`solve_rate`/`avg_turns`/`avg_tokens`.

**Monitor autonomously, don't wait for the user to nudge you.** Use
`ScheduleWakeup` at ~20 min intervals to check `status.json` + latest stage
log, and always end a wakeup-triggered turn with a visible status line even
if nothing changed — a silent turn reads as "monitoring is broken" even when
it isn't (this was flagged twice in the prior session).

## Success criteria

Primary: any checkpoint with `solved > 0` would already be a meaningful win
(baseline is a clean 0 across 8 checkpoints/2 models). Secondary, more
likely achievable: `avg_turns` improving further past the tier2 baseline
(12.17 / 10.42) without tier3 collapsing back to zero-shot — i.e. showing
GRPO can build on genuinely-taught logic rather than just format. Report
final tier0-3 comparison tables for both models against the baseline table
above, same as the format used for the original 3-model comparison earlier
in this project.
