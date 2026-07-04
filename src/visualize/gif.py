"""Generate animated GIF from parsed run data."""
import json
import os
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .frames import draw_grid
from .. import config
from ..storage import model_slug


def render_parsed_run_to_gif(data, output_path=None):
    """data: a parsed-run dict as saved by src.storage.save_parsed_run."""
    puzzle_id = data["puzzle_id"]
    model = data["model"]
    strategy_id = data["strategy_id"]
    run_number = data["run_number"]
    box_width = data["box_width"]
    box_height = data["box_height"]

    if output_path is None:
        output_path = (
            config.OUTPUT_DIR / "gifs"
            / f"{puzzle_id}_{model_slug(model)}_{strategy_id}_run{run_number}.gif"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames = []
    durations = []
    steps = data["steps"]
    total_steps = len(steps)
    solved = data.get("solved", False)
    label = f"{data.get('strategy_label', strategy_id)} [{model}]"

    initial_grid = None
    clue_mask = None
    if steps and steps[0].get("grid_before") is not None:
        initial_grid = steps[0]["grid_before"]
        clue_mask = [[c != 0 for c in row] for row in initial_grid]

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
        grid = step.get("grid_after") or step.get("grid_before")
        if grid is None:
            continue

        reasoning = (step.get("reasoning") or "")[:200]
        move = step.get("parsed_move")
        backtrack = step.get("backtrack")
        valid = step.get("valid", False)
        error_cells = set()
        highlight_cell = None

        if move and not valid:
            error_cells.add((move["row"], move["col"]))
        elif move and valid:
            highlight_cell = (move["row"], move["col"])
        elif backtrack and valid:
            highlight_cell = (backtrack["row"], backtrack["col"])

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
        with open(path) as f:
            data = json.load(f)
        if render_parsed_run_to_gif(data):
            count += 1

    print(f"\nGenerated {count} GIFs in {config.OUTPUT_DIR / 'gifs/'}")


def _fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=150)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
