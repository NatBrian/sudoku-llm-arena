BASE_PROMPT = """You are solving a {size}x{size} Sudoku puzzle.

RULES:
- The grid is {size} rows x {size} columns.
- It is divided into {box_width}x{box_height} boxes.
- Each row, column, and {box_width}x{box_height} box must contain digits 1-{size} exactly once.
- Cells pre-filled with clues cannot be changed.

CURRENT GRID:
{grid_text}

INSTRUCTIONS:
{strategy_instruction}

Keep your reasoning brief (1-3 sentences).
Then output EXACTLY ONE move in this format:
MOVE: R<row>C<col> = <value>

Example:
REASONING: Row 3 already has 1, column 2 has 4, box 1 has 3.
MOVE: R1C2 = 2

If you truly cannot make any move, output:
I'M STUCK"""

STRATEGIES = {
    "s1-direct": {
        "label": "The Impulsive",
        "color": "#FF4444",
        "instruction": (
            "Analyze the grid and choose ONE empty cell to fill.\n"
            "Use any logical reasoning you like."
        ),
    },
    "s2-naked-singles": {
        "label": "The Methodical",
        "color": "#4488FF",
        "instruction": (
            "For each empty cell, list all possible values that fit its row, column, and box.\n"
            "Only fill a cell if there is exactly ONE possible value (a naked single).\n"
            "State which candidates you considered.\n"
            "If no naked singles exist, say I'M STUCK."
        ),
    },
    "s3-hidden-singles": {
        "label": "The Scanner",
        "color": "#44CC44",
        "instruction": (
            "Scan each row, column, and box for a number that can ONLY go in ONE specific cell.\n"
            "When you find such a hidden single, fill it and explain your reasoning.\n"
            "If no hidden singles exist, say I'M STUCK."
        ),
    },
    "s4-full-logic": {
        "label": "The Logician",
        "color": "#CCAA00",
        "instruction": (
            "Use LOGICAL techniques in order:\n"
            "1. Naked single: only one candidate fits a cell.\n"
            "2. Hidden single: a number fits only one cell in a row/col/box.\n"
            "3. Cross-hatch: eliminate candidates across rows and columns.\n"
            "Pick the cell with fewest candidates. Keep reasoning brief."
        ),
    },
    "s5-guess-verify": {
        "label": "The Gambler",
        "color": "#CC44CC",
        "instruction": (
            "First try naked singles and hidden singles.\n"
            "If stuck, pick a cell with only 2 candidate values and try one.\n"
            "If you realize a previous choice leads to a dead end, say: BACKTRACK at R<row>C<col>"
        ),
    },
}


def format_grid(grid):
    lines = []
    for r, row in enumerate(grid):
        formatted = "  ".join(str(c) if c != 0 else "." for c in row)
        lines.append(f"Row {r+1}: {formatted}")
    return "\n".join(lines)


def build_prompt(strategy_id, puzzle, grid):
    strat = STRATEGIES[strategy_id]
    size = puzzle["size"]
    box_width = puzzle["box_width"]
    box_height = puzzle["box_height"]
    grid_text = format_grid(grid)

    return BASE_PROMPT.format(
        size=size,
        box_width=box_width,
        box_height=box_height,
        grid_text=grid_text,
        strategy_instruction=strat["instruction"],
    )
