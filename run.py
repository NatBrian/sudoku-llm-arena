#!/usr/bin/env python3
"""Sudoku Arena — LLM Sudoku Solving Tournament."""

import sys


def main():
    print("=" * 60)
    print("  SUDOKU ARENA — LLM Sudoku Solving Tournament")
    print("=" * 60)
    print()

    args = sys.argv[1:] if len(sys.argv) > 1 else []

    if "generate" in args or not args:
        _generate()
    if "run" in args or not args:
        _run_swarm()
    if "visualize" in args or "viz" in args or not args:
        _visualize()


def _generate():
    from src import config
    if config.PUZZLE_SOURCE == "nikoli":
        print("[1/3] Fetching Sakana Sudoku-Bench (Nikoli) puzzles...")
        from src.nikoli import load_nikoli, level_counts
        puzzles = load_nikoli(levels=config.NIKOLI_LEVELS, limit=config.NIKOLI_LIMIT)
        print(f"  levels available: {level_counts()}")
        print(f"  {len(puzzles)} puzzles selected (levels={config.NIKOLI_LEVELS}, limit={config.NIKOLI_LIMIT})")
    else:
        print("[1/3] Generating puzzles...")
        from src.puzzles import generate_all
        puzzles = generate_all()
        print(f"  {len(puzzles)} puzzles ready")
    print()


def _run_swarm():
    print("[2/3] Running swarm...")
    from src.swarm import run_swarm, print_summary
    results = run_swarm()
    if results:
        print_summary(results)
    print()


def _visualize():
    print("[3/3] Generating visualizations...")

    print("  Generating GIFs...")
    from src.visualize.gif import render_all_runs
    render_all_runs()

    print("  Generating web page...")
    from src.visualize.web import generate_web_page
    generate_web_page()

    print("  Generating report...")
    from src.visualize.report import generate_report
    generate_report()

    print()
    print(f"  Output files:")
    print(f"    output/gifs/    — Animated GIFs per run")
    print(f"    output/web/     — Interactive HTML page")
    print(f"    output/report/  — Summary charts and report")
    print()


if __name__ == "__main__":
    main()
