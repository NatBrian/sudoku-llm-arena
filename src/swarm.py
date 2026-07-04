from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .loop import GameLoop
from .storage import run_exists, save_parsed_run


def load_puzzles():
    if config.PUZZLE_SOURCE == "nikoli":
        from .nikoli import load_nikoli
        return load_nikoli(levels=config.NIKOLI_LEVELS, limit=config.NIKOLI_LIMIT)
    from .puzzles import load_all
    return load_all()


def build_jobs(puzzles):
    """Each job: (model, protocol, strategy_id_or_None, puzzle_id, puzzle_data, run_number)."""
    jobs = []
    for puzzle_id, puzzle_data in puzzles.items():
        for model in config.MODELS:
            for protocol in config.PROTOCOLS:
                if protocol == "single-shot":
                    for run_number in range(1, config.RUNS_PER_STRATEGY + 1):
                        jobs.append((model, protocol, None, puzzle_id, puzzle_data, run_number))
                else:
                    for strategy_id in config.STRATEGIES:
                        for run_number in range(1, config.RUNS_PER_STRATEGY + 1):
                            jobs.append((model, protocol, strategy_id, puzzle_id, puzzle_data, run_number))
    return jobs


def estimate_api_calls(jobs):
    """Worst-case call count. Multi-step jobs make up to empty_cells *
    MAX_TURNS_MULTIPLIER calls each (loop.py's max_turns) — actual usage is
    usually far lower since a run stops early on solve/stuck/error-cap, but
    this is the number worth knowing before launching against paid APIs."""
    total = 0
    for model, protocol, strategy_id, puzzle_id, puzzle_data, run_number in jobs:
        if protocol == "single-shot":
            total += 1
        else:
            empty_cells = sum(1 for row in puzzle_data["clues"] for c in row if c == 0)
            total += empty_cells * config.MAX_TURNS_MULTIPLIER
    return total


def run_single(model, protocol, strategy_id, puzzle_id, puzzle_data, run_number):
    label = f"[{puzzle_id} {model} {strategy_id or protocol} r{run_number}]"
    print(f"{label} Starting...", flush=True)
    game = GameLoop(puzzle_data, model, protocol, strategy_id, run_number)
    result = game.run(log_prefix=label)

    turns = result["turns"]
    parsed = {
        "run_id": result["meta"]["run_id"],
        "puzzle_id": puzzle_id,
        "level": puzzle_data.get("level"),
        "model": model,
        "protocol": protocol,
        "strategy_id": result["meta"]["strategy_id"],
        "strategy_label": result["meta"]["strategy_label"],
        "run_number": run_number,
        "size": puzzle_data["size"],
        "box_width": puzzle_data["box_width"],
        "box_height": puzzle_data["box_height"],
        "solved": result["result"]["solved"],
        "correct_against_solution": result["result"]["correct_against_solution"],
        "total_turns": result["result"]["total_turns"],
        "valid_moves": sum(1 for t in turns if t["valid"]),
        "total_errors": result["result"]["total_errors"],
        "total_prompt_tokens": result["result"]["total_prompt_tokens"],
        "total_completion_tokens": result["result"]["total_completion_tokens"],
        "duration_ms": result["meta"]["duration_ms"],
        "final_grid": result["result"]["final_grid"],
        "steps": [
            {
                "turn": t["turn"],
                "reasoning": t["reasoning_raw"],
                "parsed_move": t["parsed_move"],
                "backtrack": t.get("backtrack"),
                "valid": t["valid"],
                "validation_errors": t["validation_errors"],
                "value": (t["parsed_move"]["value"] if t["parsed_move"] else None),
                "grid_before": t["grid_before"],
                "grid_after": t["grid_after"],
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
    puzzles = load_puzzles()
    if not puzzles:
        print("No puzzles found. Check PUZZLE_SOURCE / NIKOLI_LEVELS in config, or run puzzle generation first.")
        return []

    if not config.MODELS:
        print("config.MODELS is empty — add at least one litellm model id before running.")
        return []

    all_jobs = build_jobs(puzzles)
    jobs = [j for j in all_jobs if not run_exists(j[3], j[0], j[2] or j[1], j[5])]
    skipped = len(all_jobs) - len(jobs)

    print(f"Total jobs: {len(all_jobs)}  Skipped (already done): {skipped}")
    if not jobs:
        print("All jobs already completed.")
        return []

    est_calls = estimate_api_calls(jobs)
    print(
        f"Estimated API calls for these {len(jobs)} jobs: up to {est_calls:,} "
        f"(worst case — a job's own multi-step loop can end earlier on solve/stuck/error-cap)"
    )
    print(f"Running {len(jobs)} new jobs ({config.MAX_WORKERS} parallel)...")
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single, model, protocol, strategy_id, puzzle_id, puzzle_data, run_number): (
                model, protocol, strategy_id, puzzle_id, run_number
            )
            for model, protocol, strategy_id, puzzle_id, puzzle_data, run_number in jobs
        }
        done = 0
        for future in as_completed(futures):
            model, protocol, strategy_id, puzzle_id, run_number = futures[future]
            done += 1
            try:
                result = future.result()
                results.append(result)
                solved = result["result"]["solved"]
                correct = result["result"]["correct_against_solution"]
                tok = result["result"]["total_prompt_tokens"] + result["result"]["total_completion_tokens"]
                turns = result["result"]["total_turns"]
                icon = "+" if solved else "-"
                print(f"  [{icon}] [{done}/{len(jobs)}] {puzzle_id} {model} {strategy_id or protocol} r{run_number}: "
                      f"{turns}turns {tok}tokens solved={solved} correct={correct}", flush=True)
            except Exception as e:
                print(f"  [!] [{done}/{len(jobs)}] {puzzle_id} {model} {strategy_id or protocol} r{run_number}: FAILED {e}", flush=True)

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

    for model in config.MODELS:
        model_results = [r for r in results if r["meta"]["model"] == model]
        if not model_results:
            continue
        m_solved = sum(1 for r in model_results if r["result"]["solved"])
        m_correct = sum(1 for r in model_results if r["result"]["correct_against_solution"])
        m_total = len(model_results)
        avg_turns = sum(r["result"]["total_turns"] for r in model_results) / m_total
        avg_tokens = sum(
            r["result"]["total_prompt_tokens"] + r["result"]["total_completion_tokens"]
            for r in model_results
        ) / m_total
        print(
            f"  {model:35s}"
            f"  {m_solved}/{m_total} solved"
            f"  {m_correct} correct"
            f"  avg {avg_turns:.1f} turns"
            f"  avg {avg_tokens:.0f} tokens"
        )

    print()
    for protocol in config.PROTOCOLS:
        proto_results = [r for r in results if r["meta"]["protocol"] == protocol]
        if not proto_results:
            continue
        p_solved = sum(1 for r in proto_results if r["result"]["solved"])
        p_total = len(proto_results)
        print(f"  protocol={protocol:12s}  {p_solved}/{p_total} solved")
