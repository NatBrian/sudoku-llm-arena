"""Render individual Sudoku grid frames using matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba, to_hex
import numpy as np
from .. import config

CLUE_COLOR = "#F0F0F0"
CLUE_TEXT_COLOR = "#333333"
EMPTY_COLOR = "#FFFFFF"
BORDER_COLOR = "#333333"
BOX_BORDER_COLOR = "#111111"
HIGHLIGHT_COLOR = "#4CAF50"
ERROR_COLOR = "#FF5252"
STEP_COLORMAP = plt.cm.Blues


def draw_grid(
    grid,
    box_width,
    box_height,
    clues=None,
    highlight_cell=None,
    error_cells=None,
    step_number=None,
    total_steps=None,
    reasoning=None,
    title=None,
    size_inches=6,
    dpi=150,
):
    size = len(grid)
    fig, ax = plt.subplots(1, 1, figsize=(size_inches, size_inches))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")

    if clues is None:
        clues = [[c != 0 for c in row] for row in grid]

    base_color = _step_color(step_number, total_steps) if step_number else "#E3F2FD"

    for r in range(size):
        for c in range(size):
            val = grid[r][c]
            is_clue = clues[r][c] if isinstance(clues[0], list) else False
            is_error = error_cells and (r, c) in error_cells
            is_highlight = highlight_cell and highlight_cell == (r, c)

            if is_error:
                cell_color = ERROR_COLOR
            elif is_clue:
                cell_color = CLUE_COLOR
            elif val != 0:
                cell_color = base_color
            else:
                cell_color = EMPTY_COLOR

            rect = mpatches.Rectangle(
                (c, size - 1 - r),
                1, 1,
                facecolor=cell_color,
                edgecolor=HIGHLIGHT_COLOR if is_highlight else BORDER_COLOR,
                linewidth=3 if is_highlight else 1,
            )
            ax.add_patch(rect)

            if val != 0:
                weight = "bold" if is_clue else "normal"
                color = CLUE_TEXT_COLOR if is_clue else "#333333"
                ax.text(
                    c + 0.5, size - 1 - r + 0.5,
                    str(val),
                    ha="center", va="center",
                    fontsize=size * 6,
                    fontweight=weight,
                    color=color,
                )

    for r in range(size + 1):
        lw = 3 if r % box_height == 0 else 1
        ax.plot([0, size], [r, r], color=BOX_BORDER_COLOR if r % box_height == 0 else BORDER_COLOR, linewidth=lw)

    for c in range(size + 1):
        lw = 3 if c % box_width == 0 else 1
        ax.plot([c, c], [0, size], color=BOX_BORDER_COLOR if c % box_width == 0 else BORDER_COLOR, linewidth=lw)

    if reasoning:
        fig.text(
            0.5, -0.02,
            reasoning,
            ha="center", va="top",
            fontsize=10,
            wrap=True,
            color="#555555",
        )
        fig.subplots_adjust(bottom=0.12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig, ax


def draw_summary_grid(grid, box_width, box_height, title=None, size_inches=6, dpi=150):
    size = len(grid)
    fig, ax = plt.subplots(1, 1, figsize=(size_inches, size_inches))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")

    GREEN = "#C8E6C9"
    GREY = "#F0F0F0"

    for r in range(size):
        for c in range(size):
            val = grid[r][c]
            cell_color = EMPTY_COLOR if val == 0 else GREEN
            rect = mpatches.Rectangle(
                (c, size - 1 - r), 1, 1,
                facecolor=cell_color,
                edgecolor=BORDER_COLOR,
                linewidth=1,
            )
            ax.add_patch(rect)
            if val != 0:
                ax.text(
                    c + 0.5, size - 1 - r + 0.5,
                    str(val),
                    ha="center", va="center",
                    fontsize=size * 6,
                    color="#333333",
                )

    for r in range(size + 1):
        lw = 3 if r % box_height == 0 else 1
        ax.plot([0, size], [r, r], color=BOX_BORDER_COLOR if r % box_height == 0 else BORDER_COLOR, linewidth=lw)

    for c in range(size + 1):
        lw = 3 if c % box_width == 0 else 1
        ax.plot([c, c], [0, size], color=BOX_BORDER_COLOR if c % box_width == 0 else BORDER_COLOR, linewidth=lw)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig, ax


def _step_color(step, total):
    if step is None or total is None or total <= 1:
        return "#E3F2FD"
    frac = step / total
    r = int(227 + (255 - 227) * frac)
    g = int(242 - (242 - 224) * frac)
    b = int(253 - (253 - 157) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"
