# Sudoku Arena

LLM Sudoku solving tournament — tests how well frontier models solve real Sudoku puzzles from Sakana AI's Sudoku-Bench, with animated visualizations of each solve attempt.

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
