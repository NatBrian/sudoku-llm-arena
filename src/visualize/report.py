"""Generate static summary report with comparison charts."""
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..strategies import STRATEGIES as STRAT_META


def generate_report(output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "report"
    os.makedirs(output_dir, exist_ok=True)

    runs = []
    parsed_dir = config.PARSED_DIR
    if not parsed_dir.exists():
        print("No parsed data found.")
        return
    for path in sorted(parsed_dir.glob("*.json")):
        with open(path) as f:
            runs.append(json.load(f))

    if not runs:
        print("No runs found.")
        return

    charts = {}

    charts["success_matrix"] = _chart_success_matrix(runs, output_dir)

    charts["turns_comparison"] = _chart_turns_comparison(runs, output_dir)

    charts["token_usage"] = _chart_token_usage(runs, output_dir)

    charts["errors"] = _chart_errors(runs, output_dir)

    _write_summary_md(runs, output_dir)

    print(f"  Report saved to {output_dir}/")
    return charts


def _chart_success_matrix(runs, output_dir):
    by_puzzle = defaultdict(lambda: defaultdict(list))
    for r in runs:
        pid = r.get("puzzle_id", "unknown").rsplit("_", 1)[0]
        sid = r.get("strategy_id", "unknown")
        by_puzzle[pid][sid].append(r.get("solved", False))

    puzzles = sorted(by_puzzle.keys())
    strats = list(STRAT_META.keys())

    data = np.zeros((len(puzzles), len(strats)))
    for i, p in enumerate(puzzles):
        for j, s in enumerate(strats):
            results = by_puzzle[p].get(s, [])
            data[i, j] = sum(results) / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels([STRAT_META[s]["label"][:12] for s in strats], rotation=30, ha="right")
    ax.set_yticks(range(len(puzzles)))
    ax.set_yticklabels(puzzles)

    for i in range(len(puzzles)):
        for j in range(len(strats)):
            val = data[i, j]
            if val >= 0:
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Success Rate by Puzzle and Strategy", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "success_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _chart_turns_comparison(runs, output_dir):
    by_strat = defaultdict(list)
    for r in runs:
        sid = r.get("strategy_id", "unknown")
        turns = r.get("total_turns", 0)
        if turns > 0:
            by_strat[sid].append(turns)

    strats = [s for s in STRAT_META.keys() if s in by_strat]
    labels = [STRAT_META[s]["label"] for s in strats]
    colors = [STRAT_META[s]["color"] for s in strats]
    means = [np.mean(by_strat[s]) for s in strats]
    stds = [np.std(by_strat[s]) if len(by_strat[s]) > 1 else 0 for s in strats]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(strats)), means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average Steps Taken")
    ax.set_title("Average Steps by Strategy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "turns_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _chart_token_usage(runs, output_dir):
    by_puzzle_label = defaultdict(list)
    for r in runs:
        pid = r.get("puzzle_id", "unknown").rsplit("_", 1)[0]
        tokens = (r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0))
        by_puzzle_label[pid].append(tokens)

    labels = sorted(by_puzzle_label.keys())
    data = [by_puzzle_label[l] for l in labels]
    means = [np.mean(d) for d in data]
    stds = [np.std(d) if len(d) > 1 else 0 for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=colors[:len(labels)], capsize=5, alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Tokens Used")
    ax.set_title("Token Usage by Puzzle Size", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "token_usage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _chart_errors(runs, output_dir):
    by_strat = defaultdict(list)
    for r in runs:
        sid = r.get("strategy_id", "unknown")
        errs = r.get("total_errors", 0)
        by_strat[sid].append(errs)

    strats = [s for s in STRAT_META.keys() if s in by_strat]
    labels = [STRAT_META[s]["label"] for s in strats]
    colors = [STRAT_META[s]["color"] for s in strats]
    means = [np.mean(by_strat[s]) for s in strats]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(strats)), means, color=colors, alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average Errors")
    ax.set_title("Average Errors by Strategy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "errors.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_summary_md(runs, output_dir):
    total = len(runs)
    solved = sum(1 for r in runs if r.get("solved"))
    correct = sum(1 for r in runs if r.get("correct_against_solution"))
    avg_turns = np.mean([r.get("total_turns", 0) for r in runs]) if runs else 0
    avg_errors = np.mean([r.get("total_errors", 0) for r in runs]) if runs else 0
    total_tokens = sum(r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) for r in runs)

    lines = [
        "# Sudoku Arena — Summary Report",
        "",
        f"**Total runs:** {total}",
        f"**Solved:** {solved} ({solved/total*100:.1f}%)",
        f"**Correct solutions:** {correct} ({correct/total*100:.1f}%)",
        f"**Average turns:** {avg_turns:.1f}",
        f"**Average errors:** {avg_errors:.2f}",
        f"**Total tokens consumed:** {total_tokens:,}",
        "",
        "## By Strategy",
        "| Strategy | Solved/Total | Avg Turns | Avg Errors |",
        "|---|---|---|---|",
    ]
    for sid in config.STRATEGIES:
        group = [r for r in runs if r.get("strategy_id") == sid]
        if not group:
            continue
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        g_turns = np.mean([r.get("total_turns", 0) for r in group])
        g_errs = np.mean([r.get("total_errors", 0) for r in group])
        label = STRAT_META[sid]["label"]
        lines.append(
            f"| {label} ({sid}) | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) | "
            f"{g_turns:.1f} | {g_errs:.2f} |"
        )

    lines.extend([
        "",
        "## By Puzzle Size",
        "| Size | Solved/Total |",
        "|---|---|",
    ])

    by_label = defaultdict(list)
    for r in runs:
        by_label[r.get("puzzle_id", "unknown").rsplit("_", 1)[0]].append(r)

    for label in sorted(by_label):
        group = by_label[label]
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        lines.append(f"| {label} | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) |")

    lines.append("")
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
