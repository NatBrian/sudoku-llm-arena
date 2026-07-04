"""Generate static summary report with comparison charts."""
import os
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .. import config
from ..strategies import STRATEGIES as STRAT_META

_MODEL_PALETTE = [
    "#3F51B5", "#E91E63", "#009688", "#FF9800",
    "#795548", "#607D8B", "#9C27B0", "#4CAF50",
]


def _strat_label(sid):
    return STRAT_META.get(sid, {}).get("label", sid)


def _known_and_extra_strategies(runs):
    present = {r.get("strategy_id", "unknown") for r in runs}
    known = [s for s in STRAT_META.keys() if s in present]
    extra = sorted(present - set(known))
    return known + extra


def _models(runs):
    return sorted({r.get("model", "unknown") for r in runs})


def _model_color_map(runs):
    models = _models(runs)
    return {m: _MODEL_PALETTE[i % len(_MODEL_PALETTE)] for i, m in enumerate(models)}


def _model_strategy_combos(runs):
    """(model, strategy_id) pairs actually present in the data, grouped by
    model (so per-model blocks stay visually contiguous on an axis) then
    ordered by strategy within each model."""
    strat_order = _known_and_extra_strategies(runs)
    present = {(r.get("model", "unknown"), r.get("strategy_id", "unknown")) for r in runs}
    combos = []
    for m in _models(runs):
        for s in strat_order:
            if (m, s) in present:
                combos.append((m, s))
    return combos


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

    charts["model_comparison"] = _chart_model_comparison(runs, output_dir)

    _write_summary_md(runs, output_dir)

    print(f"  Report saved to {output_dir}/")
    return charts


def _chart_success_matrix(runs, output_dir):
    by_puzzle = defaultdict(lambda: defaultdict(list))
    for r in runs:
        pid = r.get("puzzle_id", "unknown").rsplit("_", 1)[0]
        key = (r.get("model", "unknown"), r.get("strategy_id", "unknown"))
        by_puzzle[pid][key].append(r.get("solved", False))

    puzzles = sorted(by_puzzle.keys())
    combos = _model_strategy_combos(runs)

    data = np.zeros((len(puzzles), len(combos)))
    for i, p in enumerate(puzzles):
        for j, key in enumerate(combos):
            results = by_puzzle[p].get(key, [])
            data[i, j] = sum(results) / max(len(results), 1)

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.1), 5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(
        [f"{m}\n{_strat_label(s)[:12]}" for m, s in combos], rotation=30, ha="right"
    )
    ax.set_yticks(range(len(puzzles)))
    ax.set_yticklabels(puzzles)

    # Vertical separators between each model's block of strategy columns, so
    # multi-model results read as distinct groups instead of one blurred row.
    boundary = 0
    for m in _models(runs):
        count = sum(1 for combo_m, _ in combos if combo_m == m)
        boundary += count
        if boundary < len(combos):
            ax.axvline(boundary - 0.5, color="black", linewidth=1.5)

    for i in range(len(puzzles)):
        for j in range(len(combos)):
            val = data[i, j]
            if val >= 0:
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, color=color)

    ax.set_title("Success Rate by Puzzle, Model, and Strategy", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, "success_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _chart_turns_comparison(runs, output_dir):
    by_combo = defaultdict(list)
    for r in runs:
        key = (r.get("model", "unknown"), r.get("strategy_id", "unknown"))
        turns = r.get("total_turns", 0)
        if turns > 0:
            by_combo[key].append(turns)

    combos = [c for c in _model_strategy_combos(runs) if c in by_combo]
    color_map = _model_color_map(runs)
    labels = [f"{m}\n{_strat_label(s)}" for m, s in combos]
    colors = [color_map[m] for m, s in combos]
    means = [np.mean(by_combo[c]) for c in combos]
    stds = [np.std(by_combo[c]) if len(by_combo[c]) > 1 else 0 for c in combos]

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.3), 5))
    bars = ax.bar(range(len(combos)), means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average Turns Taken (incl. invalid attempts)")
    ax.set_title("Average Turns by Model and Strategy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _add_model_legend(ax, color_map)
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
    by_combo = defaultdict(list)
    for r in runs:
        key = (r.get("model", "unknown"), r.get("strategy_id", "unknown"))
        by_combo[key].append(r.get("total_errors", 0))

    combos = [c for c in _model_strategy_combos(runs) if c in by_combo]
    color_map = _model_color_map(runs)
    labels = [f"{m}\n{_strat_label(s)}" for m, s in combos]
    colors = [color_map[m] for m, s in combos]
    means = [np.mean(by_combo[c]) for c in combos]

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.3), 5))
    bars = ax.bar(range(len(combos)), means, color=colors, alpha=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average Errors")
    ax.set_title("Average Errors by Model and Strategy", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _add_model_legend(ax, color_map)
    fig.tight_layout()
    path = os.path.join(output_dir, "errors.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _add_model_legend(ax, color_map):
    handles = [mpatches.Patch(color=c, label=m) for m, c in color_map.items()]
    if len(handles) > 1:
        ax.legend(handles=handles, fontsize=8, loc="upper right")


def _chart_model_comparison(runs, output_dir):
    by_model = defaultdict(list)
    for r in runs:
        by_model[r.get("model", "unknown")].append(r.get("solved", False))

    models = sorted(by_model.keys())
    if not models:
        return None
    rates = [sum(by_model[m]) / len(by_model[m]) for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    bars = ax.bar(range(len(models)), rates, color="#3F51B5", alpha=0.85)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate by Model", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
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
        "## By Model",
        "| Model | Solved/Total | Avg Turns | Avg Valid Moves | Avg Tokens |",
        "|---|---|---|---|---|",
    ]
    models = sorted({r.get("model", "unknown") for r in runs})
    for model in models:
        group = [r for r in runs if r.get("model") == model]
        if not group:
            continue
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        g_turns = np.mean([r.get("total_turns", 0) for r in group])
        g_valid = np.mean([r.get("valid_moves", 0) for r in group])
        g_tokens = np.mean([
            r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) for r in group
        ])
        lines.append(
            f"| {model} | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) | "
            f"{g_turns:.1f} | {g_valid:.1f} | {g_tokens:.0f} |"
        )

    lines.extend([
        "",
        "## By Strategy",
        "| Strategy | Solved/Total | Avg Turns | Avg Valid Moves | Avg Errors |",
        "|---|---|---|---|---|",
    ])
    for sid in _known_and_extra_strategies(runs):
        group = [r for r in runs if r.get("strategy_id") == sid]
        if not group:
            continue
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        g_turns = np.mean([r.get("total_turns", 0) for r in group])
        g_valid = np.mean([r.get("valid_moves", 0) for r in group])
        g_errs = np.mean([r.get("total_errors", 0) for r in group])
        label = _strat_label(sid)
        lines.append(
            f"| {label} ({sid}) | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) | "
            f"{g_turns:.1f} | {g_valid:.1f} | {g_errs:.2f} |"
        )

    lines.extend([
        "",
        "## By Model × Strategy",
        "| Model | Strategy | Solved/Total | Avg Turns | Avg Errors |",
        "|---|---|---|---|---|",
    ])
    for model, sid in _model_strategy_combos(runs):
        group = [r for r in runs if r.get("model") == model and r.get("strategy_id") == sid]
        if not group:
            continue
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        g_turns = np.mean([r.get("total_turns", 0) for r in group])
        g_errs = np.mean([r.get("total_errors", 0) for r in group])
        lines.append(
            f"| {model} | {_strat_label(sid)} | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) | "
            f"{g_turns:.1f} | {g_errs:.2f} |"
        )

    lines.extend([
        "",
        "## By Puzzle",
        "| Puzzle | Level | Solved/Total |",
        "|---|---|---|",
    ])

    by_label = defaultdict(list)
    for r in runs:
        by_label[r.get("puzzle_id", "unknown")].append(r)

    for label in sorted(by_label):
        group = by_label[label]
        g_solved = sum(1 for r in group if r.get("solved"))
        g_total = len(group)
        level = group[0].get("level") or "-"
        lines.append(f"| {label} | {level} | {g_solved}/{g_total} ({g_solved/g_total*100:.0f}%) |")

    lines.append("")
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
