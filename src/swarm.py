from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .puzzles import load_all
from .loop import GameLoop
from .storage import run_exists, save_parsed_run


def run_single(strategy_id, puzzle_id, puzzle_data, run_number):
    label = f"[{puzzle_id} {strategy_id} r{run_number}]"
    print(f"{label} Starting...", flush=True)
    game = GameLoop(puzzle_data, strategy_id, run_number)
    result = game.run(log_prefix=label)

    turns = result["turns"]
    parsed = {
        "run_id": result["meta"]["run_id"],
        "puzzle_id": puzzle_id,
        "strategy_id": strategy_id,
        "strategy_label": result["meta"]["strategy_label"],
        "run_number": run_number,
        "solved": result["result"]["solved"],
        "correct_against_solution": result["result"][
            "correct_against_solution"
        ],
        "total_turns": result["result"]["total_turns"],
        "total_errors": result["result"]["total_errors"],
        "total_prompt_tokens": result["result"]["total_prompt_tokens"],
        "total_completion_tokens": result["result"][
            "total_completion_tokens"
        ],
        "duration_ms": result["meta"]["duration_ms"],
        "final_grid": result["result"]["final_grid"],
        "steps": [
            {
                "turn": t["turn"],
                "reasoning": t["reasoning_raw"],
                "parsed_move": t["parsed_move"],
                "valid": t["valid"],
                "validation_errors": t["validation_errors"],
                "value": (
                    t["parsed_move"]["value"] if t["parsed_move"] else None
                ),
                "grid_before": t["grid_before"],
                "grid_after": t["grid_after"],
                "candidates_before": t["candidates_before"],
            }
            for t in turns
        ],
        "error_steps": [
            {
                "turn": t["turn"],
                "reasoning": t["reasoning_raw"],
                "parsed_move": t["parsed_move"],
                "validation_errors": t["validation_errors"],
            }
            for t in turns
            if not t["valid"]
        ],
    }
    save_parsed_run(parsed)
    return result


def run_swarm():
    puzzles = load_all()
    if not puzzles:
        print("No puzzles found. Run puzzle generation first.")
        return []

    total_jobs = 0
    skipped = 0
    for puzzle_id in puzzles:
        for strategy_id in config.STRATEGIES:
            for run_number in range(1, config.RUNS_PER_STRATEGY + 1):
                total_jobs += 1
                if run_exists(puzzle_id, strategy_id, run_number):
                    skipped += 1

    print(f"Total jobs: {total_jobs}  Skipped (already done): {skipped}")

    jobs = []
    for puzzle_id, puzzle_data in puzzles.items():
        for strategy_id in config.STRATEGIES:
            for run_number in range(1, config.RUNS_PER_STRATEGY + 1):
                if run_exists(puzzle_id, strategy_id, run_number):
                    continue
                jobs.append((strategy_id, puzzle_id, puzzle_data, run_number))

    if not jobs:
        print("All jobs already completed.")
        return []

    print(f"Running {len(jobs)} new jobs ({config.MAX_WORKERS} parallel)...")
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single, s, p, d, r): (s, p, r)
            for s, p, d, r in jobs
        }
        done = 0
        for future in as_completed(futures):
            s, p, r = futures[future]
            done += 1
            try:
                result = future.result()
                results.append(result)
                solved = result["result"]["solved"]
                correct = result["result"]["correct_against_solution"]
                tok = (
                    result["result"]["total_prompt_tokens"]
                    + result["result"]["total_completion_tokens"]
                )
                turns = result["result"]["total_turns"]
                icon = "+" if solved else "-"
                print(f"  [{icon}] [{done}/{len(jobs)}] {p} {s} r{r}: {turns}turns {tok}tokens solved={solved} correct={correct}", flush=True)
            except Exception as e:
                print(f"  [!] [{done}/{len(jobs)}] {p} {s} r{r}: FAILED {e}", flush=True)

    return results


def print_summary(results):
    if not results:
        print("No results to summarize.")
        return

    solved = sum(1 for r in results if r["result"]["solved"])
    correct = sum(1 for r in results if r["result"]["correct_against_solution"])
    total = len(results)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {solved}/{total} solved ({solved/total*100:.1f}%)")
    print(f"         {correct}/{total} correct solutions ({correct/total*100:.1f}%)")
    print(f"{'='*60}")

    for sid in config.STRATEGIES:
        strat_results = [r for r in results if r["meta"]["strategy_id"] == sid]
        if not strat_results:
            continue
        s_solved = sum(1 for r in strat_results if r["result"]["solved"])
        s_correct = sum(
            1 for r in strat_results if r["result"]["correct_against_solution"]
        )
        s_total = len(strat_results)
        avg_turns = (
            sum(r["result"]["total_turns"] for r in strat_results) / s_total
        )
        avg_errors = (
            sum(r["result"]["total_errors"] for r in strat_results) / s_total
        )
        label = strat_results[0]["meta"]["strategy_label"]
        print(
            f"  {label:20s} ({sid:20s})"
            f"  {s_solved}/{s_total} solved"
            f"  {s_correct} correct"
            f"  avg {avg_turns:.1f} turns"
            f"  avg {avg_errors:.1f} errors"
        )

    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r["meta"]["puzzle_id"].rsplit("_", 1)[0]].append(r)

    print()
    for label in sorted(by_label):
        group = by_label[label]
        g_solved = sum(1 for r in group if r["result"]["solved"])
        g_total = len(group)
        g_turns = sum(r["result"]["total_turns"] for r in group) / g_total
        print(
            f"  {label:15s}  {g_solved}/{g_total} solved"
            f"  avg {g_turns:.1f} turns"
        )
