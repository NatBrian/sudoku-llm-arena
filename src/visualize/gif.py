"""Generate animated GIF from parsed run data."""
import os
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .frames import draw_grid
from .. import config


def render_run_to_gif(puzzle_id, strategy_id, run_number, output_path=None):
    parsed_path = config.PARSED_DIR / f"{puzzle_id}_{strategy_id}_run{run_number}.json"
    if not parsed_path.exists():
        print(f"  No parsed data for {puzzle_id} {strategy_id} run {run_number}")
        return None

    import json
    with open(parsed_path) as f:
        data = json.load(f)

    size = 0
    box_width = 0
    box_height = 0

    puzzle_path = config.PUZZLE_DIR / f"{puzzle_id}.json"
    if puzzle_path.exists():
        with open(puzzle_path) as f:
            puzzle = json.load(f)
        size = puzzle["size"]
        box_width = puzzle["box_width"]
        box_height = puzzle["box_height"]
        clues_grid = puzzle["clues"]
    else:
        steps = data["steps"]
        if steps and steps[0].get("grid_before"):
            g = steps[0]["grid_before"]
            size = len(g)
            box_width = 2 if size == 4 else (3 if size == 6 else 3)
            box_height = 2 if size == 4 else (2 if size == 6 else 3)
            clues_grid = g

    if output_path is None:
        output_path = config.OUTPUT_DIR / "gifs" / f"{puzzle_id}_{strategy_id}_run{run_number}.gif"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = []
    durations = []
    steps = data["steps"]

    total_steps = len(steps)
    solved = data.get("solved", False)
    label = data.get("strategy_label", strategy_id)

    initial_grid = None
    clue_mask = None
    if steps and steps[0].get("grid_before") is not None:
        initial_grid = steps[0]["grid_before"]
        clue_mask = [
            [c != 0 for c in row]
            for row in initial_grid
        ]

    if initial_grid:
        fig, _ = draw_grid(
            initial_grid, box_width, box_height,
            clues=clue_mask,
            title=f"{label} — {puzzle_id}",
            reasoning="Initial puzzle state",
        )
        frames.append(_fig_to_image(fig))
        durations.append(2000)
        plt.close(fig)

    for i, step in enumerate(steps):
        grid = step.get("grid_before") or step.get("grid_after")
        if grid is None:
            continue

        reasoning = step.get("reasoning", "")[:200]
        move = step.get("parsed_move")
        valid = step.get("valid", False)
        error_cells = set()
        highlight_cell = None

        if move and not valid:
            error_cells.add((move["row"], move["col"]))
        elif move and valid:
            highlight_cell = (move["row"], move["col"])

        fig, _ = draw_grid(
            grid, box_width, box_height,
            clues=clue_mask,
            highlight_cell=highlight_cell,
            error_cells=error_cells,
            step_number=i + 1,
            total_steps=total_steps,
            reasoning=reasoning,
            title=f"{label} — Step {i + 1}/{total_steps}",
        )
        frames.append(_fig_to_image(fig))
        is_key_step = not valid or "stuck" in reasoning.lower() or "backtrack" in reasoning.lower()
        durations.append(3000 if is_key_step else 1500)
        plt.close(fig)

    final_grid = data.get("final_grid") or (steps[-1]["grid_after"] if steps else None)
    if final_grid:
        status = "SOLVED!" if solved else "UNSOLVED"
        fig, _ = draw_grid(
            final_grid, box_width, box_height,
            clues=clue_mask,
            title=f"{label} — {status}",
            reasoning=f"Solved: {solved} | Total steps: {total_steps}",
        )
        frames.append(_fig_to_image(fig))
        durations.append(3000)
        plt.close(fig)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=False,
        )
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  GIF saved: {output_path} ({size_kb:.0f} KB, {len(frames)} frames)")
        return output_path

    return None


def render_all_runs():
    parsed_dir = config.PARSED_DIR
    if not parsed_dir.exists():
        print("No parsed data found. Run the swarm first.")
        return

    count = 0
    for path in sorted(parsed_dir.glob("*.json")):
        parts = path.stem.split("_")
        if len(parts) >= 3:
            puzzle_id = f"{parts[0]}_{parts[1]}"
            strategy_id = parts[2]
            run_number = int(parts[3].replace("run", ""))
            result = render_run_to_gif(puzzle_id, strategy_id, run_number)
            if result:
                count += 1

    print(f"\nGenerated {count} GIFs in {config.OUTPUT_DIR / 'gifs/'}")


def _fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
