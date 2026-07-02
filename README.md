# Sudoku Arena

LLM Sudoku solving tournament — tests how well different prompting strategies help language models solve Sudoku puzzles of varying sizes.

## How it works

1. **Generate** puzzles (4x4, 6x6, 9x9, 9x9 Evil)
2. **Run** each puzzle through 5 prompting strategies (each with 3 runs)
3. **Visualize** results as GIFs, interactive web page, and summary report

## Strategies

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
```

Set your DeepSeek API key:

```bash
# PowerShell
$env:DEEPSEEK_API_KEY = "sk-..."
# bash
export DEEPSEEK_API_KEY="sk-..."
```

Optionally set a custom API base:

```bash
$env:DEEPSEEK_BASE_URL = "https://your-proxy.com/v1"
```

## Usage

```bash
python run.py
```

Run individual steps:

```bash
python -c "from src.puzzles import generate_all; generate_all()"
python -c "from src.swarm import run_swarm; run_swarm()"
python -c "from src.visualize.gif import render_all_runs; render_all_runs()"
python -c "from src.visualize.web import generate_web_page; generate_web_page()"
python -c "from src.visualize.report import generate_report; generate_report()"
```

## Output

- `output/gifs/` — Animated GIFs per run
- `output/web/` — Interactive HTML viewer
- `output/report/` — Summary charts and markdown report
- `data/parsed/` — Run data in JSON format
