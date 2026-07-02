from sudoku import Sudoku
from . import config
import json


def _board_to_list(board):
    return [[c if c is not None else 0 for c in row] for row in board]


def generate_puzzle(width, height, difficulty=0.5, seed=None):
    seed = seed or config.PUZZLE_DIR.stat().st_ctime_ns if hasattr(config.PUZZLE_DIR.stat(), 'st_ctime_ns') else None
    puzzle = Sudoku(width, height, seed=seed)
    puzzle = puzzle.difficulty(difficulty)
    solution = puzzle.solve()

    clues = _board_to_list(puzzle.board)
    sol = _board_to_list(solution.board)
    clue_count = sum(1 for row in clues for c in row if c != 0)

    return {
        "size": width * height,
        "box_width": width,
        "box_height": height,
        "clues": clues,
        "solution": sol,
        "clue_count": clue_count,
    }


def generate_all():
    config.PUZZLE_DIR.mkdir(parents=True, exist_ok=True)
    puzzles = {}
    for label, params in config.PUZZLE_SIZES.items():
        for i in range(params["count"]):
            puzzle_id = f"{label}_{i+1:03d}"
            data = generate_puzzle(
                params["width"],
                params["height"],
                params["difficulty"],
                seed=hash(puzzle_id) & 0xFFFFFFFF,
            )
            data["puzzle_id"] = puzzle_id
            data["label"] = label
            filepath = config.PUZZLE_DIR / f"{puzzle_id}.json"
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            puzzles[puzzle_id] = data
            print(f"  Puzzle: {puzzle_id} | {data['clue_count']} clues | {data['size']}x{data['size']}")
    return puzzles


def load_puzzle(puzzle_id):
    with open(config.PUZZLE_DIR / f"{puzzle_id}.json") as f:
        return json.load(f)


def load_all():
    puzzles = {}
    if not config.PUZZLE_DIR.exists():
        return puzzles
    for path in sorted(config.PUZZLE_DIR.glob("*.json")):
        with open(path) as f:
            puzzles[path.stem] = json.load(f)
    return puzzles
