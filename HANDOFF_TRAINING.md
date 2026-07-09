# Handoff: qwen3-0.6b / qwen3-1.7b sudoku training — round 2

You are picking up a project that has already had one full round of fixes
applied and re-run. This doc gives you everything you need without assuming
you've read any prior conversation. Read it fully before changing anything.

## Project overview

Repo: `sudoku-llm-arena` (this directory). Two purposes:
1. Evaluate frontier LLMs (GPT/Claude/etc, via litellm) solving real Sudoku
   puzzles from Sakana AI's `sudoku-bench-nikoli` (100 real hand-made 9x9
   puzzles by Nikoli, on Hugging Face). Not the focus of this handoff.
2. Train small local models (`Qwen3-0.6B`, `Qwen3-1.7B`, and a separately
   already-finished `Qwen2.5-1.5B-Instruct` run) via `src/train/` to see how
   much cheap post-training closes the gap to frontier models on the same
   100 puzzles. **This is what you're working on.**

Training pipeline has 4 tiers per model, run in sequence by
`src/train/run_experiment.py`:
- `tier0-zeroshot` — base model, no training, run through normal eval harness
- `tier1-lora-sft` — LoRA adapter, supervised on `grid_state -> correct move`
  (plus two curriculum pre-stages on 4x4 and 6x6 boards before the final 9x9
  SFT pass — see `TRAIN_CURRICULUM_STAGES` in `src/train/config.py`)
- `tier2-lora-grpo` — continues tier1's LoRA adapter with GRPO (RL), reward
  from `src/train/reward.py`
- `tier3-full-grpo` — full fine-tune (unfreezes all weights, starts from base
  checkpoint, not tier1's adapter) + GRPO on top

Eval happens through the *same* harness real frontier models are graded
through (`loop.py` / `parser.py` / `validator.py`) — no separate grading path
for trained checkpoints. Protocol used for both training and eval is
`s1-direct`, a single-move-per-turn strategy under the **multi-step**
protocol: model sees current board, places one digit, gets told
valid/invalid, repeats. This is one of Sakana's two benchmark modes.

**Important nuance you need to know:** Sakana's own *headline* solve-rate
metric uses the OTHER mode — **single-shot** (model gets the puzzle once,
must return the entire solved grid in one response). This repo supports
single-shot too (`src/config.py: PROTOCOLS = ["multi-step"]` — it's a list,
single-shot is implemented but currently disabled/unused). **Nothing in this
project has ever been trained or evaluated in single-shot mode.** All results
below, and all of round 1's results, are multi-step only. Keep this in mind —
see "Open question: protocol mismatch" below.

`qwen2.5-1.5b` is a separate, already-fully-trained model family, out of
scope here unless you want a 3-way comparison for free (results already in
`output/checkpoints/qwen2.5-1.5b-tier*/eval_summary.json`).

## What round 1 found, and what was done about it (all already applied)

Round 1 trained both qwen3 models through all 4 tiers and got 0/100 solved
on every checkpoint. Diagnosis at the time, and current status of each fix:

1. **Vacuous reasoning traces.** SFT training data included a `REASONING:`
   line before every `MOVE:`, but for any cell that wasn't a trivial "naked
   single" it fell back to a content-free filler sentence ("does not conflict
   ... and matches the puzzle's unique solution") that didn't explain *why*.
   **Fixed.** `src/validator.py` now has real detectors: `find_naked_singles`,
   `find_hidden_singles`, `eliminate_naked_pairs`, `eliminate_pointing_pairs`,
   `eliminate_box_line_reduction`, unified through `classify_move()`.
   `src/train/data.py:_reasoning_for()` now **skips** (doesn't train on) any
   cell whose target move isn't justified by one of these real techniques,
   instead of falling back to filler. Confirmed via SFT data-gen logs
   ("N snapshots skipped (no technique justifies the target move)").
2. **No curriculum.** Model went straight to 9x9, no smaller-board warm-up.
   **Fixed.** `TRAIN_CURRICULUM_STAGES` in `src/train/config.py` now runs a
   4x4 (`box_width=2, box_height=2`) then 6x6 (`box_width=3, box_height=2`)
   SFT pass before the canonical 9x9 tier1 pass, each continuing the previous
   stage's LoRA adapter. Confirmed via
   `output/experiments/<model>/tier1-curriculum-{4x4,6x6}.log` existing and
   showing `mean_token_accuracy` climbing to 0.97-0.98 by end of each stage.
   **Caveat: this token-level SFT accuracy was never converted into an actual
   solve-rate eval on 4x4/6x6 puzzles** — see open questions below.
3. **Sparse GRPO reward.** Was `{-1, +1, +2}` only (invalid / valid / valid +
   matches solution). **Fixed.** `src/train/reward.py:reward_func()` now
   adds: `TECHNIQUE_BONUS` (+0.5, if the move is justified by a known
   technique), and a `PROGRESS_BONUS` (up to +0.5, scaled by how many
   candidates the move eliminates elsewhere on the board — isolates genuine
   constraint propagation from a merely-legal-but-uninformative placement).
   Also fixed a backtrack-handling bug where the model was previously
   punished for backtracking a genuine self-made mistake (`BACKTRACK_CORRECT_REWARD
   = 1.0` now rewards this — it's the model's only tool to escape a
   self-created dead end per the eval harness).
4. **Rollout diversity squeezed too low by OOM workarounds.** Round 1 had to
   cut `GRPO_NUM_GENERATIONS`/`GRPO_MAX_COMPLETION_LENGTH` for memory.
   **Partially fixed.** Base config (`src/train/config.py`) now defaults to
   `GRPO_NUM_GENERATIONS = 8`, `GRPO_MAX_COMPLETION_LENGTH = 256` — this
   applies to tier2 (LoRA+GRPO) unmodified. **tier3 (full fine-tune + GRPO)
   still hard-caps these down** to `num_generations = min(8, 2) = 2` and
   `max_completion_length = min(256, 96) = 96` in `grpo_train.py` (search
   `beta = 0.0` branch) — full fine-tune without a LoRA adapter to derive a
   cheap KL-reference from still needs a full second frozen model copy for
   `beta != 0`, which OOMs on this host, so `beta` is forced to 0 and rollout
   size stays squeezed for tier3 specifically. **This is very likely still a
   live problem — see diagnosis below.**

Also fixed since round 1 (per prior session, not itemized in the original
diagnosis but relevant context): a retry-loop determinism bug, a
tier3-truncation bug, and an exposure-bias bug in self-rollout recovery data
collection. All 4 tiers for both models completed a **second full run** after
all of the above; that second run is the "current baseline" below.

## Current baseline (already run — this is the control group, do not repeat as-is)

All 100 real Nikoli puzzles, multi-step protocol. **Still 0/100 solved on
every single checkpoint, both models, every tier**, despite every round-1 fix
above being genuinely implemented and confirmed working (curriculum ran,
techniques exist, denser reward is active, tier3 no longer collapses back to
near-zero-shot the way it did in round 1 — see avg_turns trend now being
mildly monotonic instead of collapsing).

| tier | qwen3-0.6b avg_turns | qwen3-1.7b avg_turns |
|---|---|---|
| tier1 LoRA SFT | 56.4 | 52.3 |
| tier2 LoRA+GRPO | 51.9 | 52.8 |
| tier3 full+GRPO | 50.9 | 53.8 |

(`avg_turns` = how many turns elapse before the harness gives up on a puzzle,
NOT solving competence — max possible is 110. Puzzles have 81 cells; turns
also include backtracks.) Note 0.6b improves slightly tier-over-tier, 1.7b
gets marginally *worse* — bigger model, more training, same failure mode.

Logs/checkpoints for this run are on disk at
`output/checkpoints/qwen3-{0.6b,1.7b}-tier{0,1,2,3}*/` and
`output/experiments/qwen3-{0.6b,1.7b}/*.log` + `status.json` — **don't
delete, this is the control group**.

## New diagnosis (this session, from re-reading the current-run logs directly)

This goes deeper than round 1's diagnosis, which was mostly hypothesis-driven
(built before this run's logs existed). These findings are all directly
measured from `output/experiments/qwen3-{0.6b,1.7b}/tier3-eval.log` and the
`tier1/2/3-train.log` files, both models:

1. **Every single puzzle (200/200 across both models) ends via the harness's
   "too many consecutive errors (5) — stopping" rule**, not by running out
   of turns (110 cap) or by solving. Average stop point ~turn 51-54, with
   ~26-32 of 81 cells still empty at that point.
2. **Raw invalid-move rate is 45.6-45.8%** across all moves attempted in
   tier3 eval, both models — the model is wrong on very close to half its
   guesses even after all 3 training tiers.
3. **This is NOT a "stuck repeating the same wrong move" loop.** Only ~2-3%
   of moves are an exact repeat of the immediately preceding invalid move
   (120/5094 for 0.6b, 166/5382 for 1.7b). The other ~43% are *distinct*
   wrong guesses — the model isn't looping, it's genuinely failing to track
   row/column/box constraints as the board fills in, move after move.
4. **GRPO training visibly plateaus from step 1 and never improves** for
   both models, at both tier2 and tier3. Grep any `tier{2,3}-train.log` for
   the `reward`/`entropy`/`frac_reward_zero_std` fields (trl's GRPOTrainer
   logs these every step): `reward` mean hovers ~1.9-2.2 (of a max ~3.0)
   flat from the first logged step to the last, `entropy` stays pinned in a
   narrow 0.07-0.09 band the entire run (never climbs, never really drops
   either — just noise around a fixed point), and `frac_reward_zero_std`
   (fraction of GRPO groups where every sampled completion got the *exact
   same* reward, i.e. zero advantage signal for that group) sits at
   0.2-0.5 throughout. **A GRPO run this flat is not learning anything
   during that stage** — whatever solving competence exists is coming
   entirely from tier1 SFT, and tier2/tier3's RL stages are spinning wheels.
5. **`completions/mean_length`, `min_length`, and `max_length` are reported
   as numerically identical (all exactly `38`) at every single logged step**
   in both tier2 and tier3 training, both models — i.e., across an entire
   300-step run, batch after batch, every sampled completion is reported as
   exactly the same token length. This needs verification (dump a handful of
   raw completions during a training run and read them — don't just trust
   the aggregate metric) but is consistent with finding #4: if completions
   are this length-rigid, they're likely also content-rigid, which would
   directly explain why GRPO groups keep landing on identical rewards (no
   diversity to reward-differentiate) and why entropy never moves.

**Working theory, not yet confirmed:** tier1 SFT (target completion format is
literally `"REASONING: {one short sentence}\nMOVE: R{r}C{c} = {value}"`, with
Qwen3's `<think>` block explicitly disabled via `enable_thinking: False` in
`src/train/data.py` for train/eval consistency across model families) trains
the model into an extremely narrow, low-entropy output distribution before
GRPO ever starts. GRPO's on-policy sampling then can't produce meaningfully
different rollouts within a group to compute an advantage from, so the RL
stages (tier2, tier3) never get a real learning signal — they just do
short-caption imitation of whatever the SFT stage already baked in, which
tops out at ~54% valid-move accuracy and no actual constraint-tracking
competence. **This would mean tier1 SFT is effectively the ceiling of the
whole pipeline right now, and tier2/tier3 are not adding value** — worth
directly testing by comparing tier1-only eval numbers against tier2/tier3 (a
comparison round 1 already collected — `output/experiments/*/tier1-eval.log`
— but wasn't the focus at the time).

## Open questions to resolve before spending a lot of compute on more training

- **Does the 4x4/6x6 curriculum actually work?** It was added and clearly
  changes the SFT loss curve (mean_token_accuracy 0.97-0.98 by end of each
  stage), but that's *teacher-forced next-token accuracy*, not solve rate —
  it doesn't tell you the model can actually solve a 4x4 or 6x6 board
  end-to-end. Nobody has run the eval harness against a 4x4/6x6 held-out
  puzzle set. If the model can't reliably solve *4x4* end-to-end, the 9x9
  failure isn't surprising and no amount of 9x9-specific tuning will fix the
  root cause. If it CAN solve 4x4/6x6 well, that's strong evidence the
  problem is specifically about scale-transfer (constraint tracking that
  works for 16-36 cells breaking down at 81), which points at different
  fixes (e.g. explicit board-state re-statement in the prompt every turn,
  rather than relying on the model to hold state in its own context).
- **Protocol mismatch.** Everything above is multi-step. Sakana's headline
  metric is single-shot (whole grid in one response), and it's implemented
  but unused (`src/config.py: PROTOCOLS`). Before sinking more time into
  multi-step training, run a cheap **zero-shot, no-training** single-shot
  eval on both base models (just flip `PROTOCOLS` to include `"single-shot"`
  and run tier0 again) to see whether that mode is meaningfully different —
  it might not be worth training for if zero-shot single-shot is also ~0%,
  but it's a very cheap check (no GPU training, just inference) and it's the
  metric actually being compared to frontier models' headline numbers.
- **Verify finding #5 isn't a metric-logging artifact.** Before concluding
  anything about output rigidity, actually print 10-20 raw completions
  sampled mid-training (add a debug print in `grpo_train.py` or hook into
  trl's completion callback) and read them by eye. If they're genuinely all
  ~38 tokens of near-identical structure with different digits, that
  confirms the "too rigid to explore" theory. If they vary more than the
  aggregate metric suggests, the metric itself is misleading and the flat
  entropy needs a different explanation.

## Recommended next steps, in priority order

### 1. Isolate whether tier2/tier3 RL is adding anything (cheap, do first)

Compare `output/experiments/qwen3-{0.6b,1.7b}/tier1-eval.log` numbers
directly against tier2 and tier3 (already have all 3 on disk from the
current run — no new training needed, just read the logs). If tier1 alone is
statistically indistinguishable from tier2/tier3 on invalid-move-rate and
avg_turns, that confirms GRPO isn't contributing, and effort should go into
fixing tier1 SFT (better curriculum, longer reasoning traces, more data) or
the GRPO exploration problem (below) — not into tuning GRPO reward shaping
further, which round 1 already did without effect.

### 2. Fix GRPO exploration (if #1 confirms RL stages are inert)

Investigate and likely raise generation temperature/sampling diversity for
GRPO rollouts specifically — there is currently NO explicit
temperature/top_p/top_k set anywhere in `src/train/grpo_train.py` or
`config.py`, meaning trl's GRPOTrainer defaults are in effect. Check what
those defaults are for the installed trl version and consider explicitly
raising temperature (e.g. 1.0-1.3) or adding `min_p` sampling specifically
for the GRPO rollout phase (NOT for eval, which should stay at whatever
matches how frontier models are graded) to break the near-zero group
variance. Also worth trying: restore tier3's `num_generations` above 2 (find
a memory workaround other than dropping group size — e.g. gradient
checkpointing, smaller `GRPO_BATCH_SIZE`, or accept a smaller model instead
of full fine-tune) since 2 rollouts per group is a very weak base for
GRPO's group-relative advantage regardless of temperature.

### 3. Get a real solve-rate number on 4x4/6x6 (cheap diagnostic, do early)

After a curriculum stage finishes, run the existing eval harness against a
held-out set of procedurally generated 4x4 and 6x6 puzzles (same generator
`src/train/synth.py` already uses, just don't reuse training seeds) instead
of only looking at SFT token accuracy. This is a config/script change, not a
retrain — reuses the already-trained curriculum checkpoints if you're
willing to eval mid-pipeline (check `run_experiment.py` for how tier1-eval
is invoked; adapt to point at a 4x4/6x6 puzzle set and box dimensions
instead of the default 9x9 Nikoli set).

### 4. Test single-shot protocol, zero-shot first (cheap, resolves the metric question)

Flip `PROTOCOLS` in `src/config.py` to include `"single-shot"`, run tier0
(no training) for both base models through it. This tells you, cheaply,
whether the metric you actually care about (Sakana's headline number) looks
any different from multi-step before deciding whether to invest in training
for it. **Do not jump straight to training a single-shot-optimized model
without doing this check first** — single-shot removes the harness's
per-move validator feedback entirely (model gets zero correction until the
very end), which is plausibly *harder* for a model this small to succeed at
than multi-step, not easier — go in with that expectation, not an assumption
that switching protocols is a fix.

### 5. Only if 1-4 above don't explain it: revisit reasoning-trace richness

`_reasoning_for()` currently emits one short sentence per move and Qwen3's
`<think>` block is explicitly suppressed. If the exploration/RL-inertness
problem turns out NOT to be the bottleneck, the next lever is testing
whether *actual* multi-sentence step-by-step deduction (enabling `<think>`,
accepting the train/eval prompt-format inconsistency across Qwen3 vs
Qwen2.5 that `enable_thinking: False` was added to avoid) gives the model
enough "space" to do real constraint propagation instead of a one-shot
pattern-matched answer. This is a bigger, slower experiment than 1-4 — do it
last.

## Known operational pitfalls (read before running — all previously hit and fixed; re-verify still true on this host before assuming so)

- **This host has 8 shared GPUs; another process may hold most of GPU0's
  memory.** Every training/eval subprocess MUST run with
  `CUDA_VISIBLE_DEVICES=0` restricted (already done via `_STAGE_ENV` in
  `src/train/run_experiment.py`) — otherwise `transformers.Trainer`
  auto-wraps in `DataParallel` across all 8 GPUs, touching other users' jobs.
  Check `nvidia-smi` free memory on GPU0 before launching anything.
- **tier3 (full fine-tune + GRPO) needs `beta=0.0`.** `trl`'s `GRPOTrainer`
  loads a full second copy of the model as a frozen KL-reference whenever
  `beta != 0` and there's no LoRA adapter to derive it from cheaply — extra
  memory that WILL OOM on this host's headroom. Already set in
  `grpo_train.py`'s tier3 branch; don't remove it without a memory plan.
- **`evaluate_checkpoint()` rebuilds its summary from
  `data/parsed/*_{slug}_*.json` on disk**, not from `run_swarm()`'s return
  value — the latter excludes cached/deduped puzzles and will silently
  produce a hollow summary on a partial re-run. Don't "fix" this back.
- **Training data is 100% synthetic (`src/train/synth.py`, py-sudoku
  procedural generation) — never derived from the real Nikoli-100 set.** All
  100 real Nikoli puzzles are reserved for eval only. This was a deliberate
  fix for an earlier data-leakage bug (train/eval split of the *same* Nikoli
  puzzles). **Do not reintroduce Nikoli puzzles into the training set, even a
  held-out split** — eval results become meaningless if the model trains on
  the same *kind* of puzzle distribution it's tested on, unlike the frontier
  models being compared against.
- **Run each stage as its own subprocess** (already how `run_experiment.py`
  works) — don't refactor to a single long-lived process across tiers, GPU
  memory from one stage lingers into the next otherwise.
- **Launch training via `Bash(..., run_in_background: true)`, not
  `nohup ... &`.** Only the former is harness-tracked and reliably triggers
  a completion notification; a detached process requires manual polling.

## How to run

```bash
python -m src.train.run_experiment --model qwen3-0.6b
python -m src.train.run_experiment --model qwen3-1.7b
```

Each does tier0-register → tier0-eval → curriculum-4x4 → curriculum-6x6 →
tier1-train → tier1-eval → (if ok) tier2-train → tier2-eval → tier3-train →
tier3-eval, writing `output/experiments/<model>/status.json` (stage-by-stage
pass/fail + timing) and per-stage `.log` files as it goes — check those
instead of waiting for the whole run to finish. Each eval writes
`output/checkpoints/<checkpoint>/eval_summary.json` with
`solved`/`solve_rate`/`avg_turns`/`avg_tokens`.

A full 2-model run takes on the order of 6-8 hours wall-clock (tier1-eval and
tier2-eval alone are ~2 hours each per model — 100 puzzles x up to 110 turns
x per-turn local inference). **Monitor autonomously, don't wait for the user
to nudge you.** Use `ScheduleWakeup` at ~20 min intervals to check
`status.json` + latest stage log, and always end a wakeup-triggered turn with
a visible status line even if nothing changed — a silent turn reads as
"monitoring is broken" even when it isn't.

## Success criteria

Primary: any checkpoint with `solved > 0` on the real Nikoli-100 eval would
already be a meaningful win (current baseline is a clean 0 across every
checkpoint of two full training rounds). Secondary, more diagnostic than a
final win: a clear answer to "is tier2/tier3 GRPO adding anything over
tier1 SFT" and "does the model actually solve 4x4/6x6," since both are
currently unknown and would redirect effort more efficiently than another
blind full retrain. Report a tier0-3 comparison table for both models in the
same format as above, plus explicit answers to the two open questions.
