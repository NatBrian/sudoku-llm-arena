import json
from . import config


def save_puzzle(puzzle_data):
    config.PUZZLE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PUZZLE_DIR / f"{puzzle_data['puzzle_id']}.json"
    with open(path, "w") as f:
        json.dump(puzzle_data, f, indent=2, default=str)
    return path


def save_raw_run(run_data):
    puzzle_id = run_data["meta"]["puzzle_id"]
    strategy_id = run_data["meta"]["strategy_id"]
    run_number = run_data["meta"]["run_number"]

    dirpath = config.RAW_DIR / puzzle_id
    dirpath.mkdir(parents=True, exist_ok=True)

    filepath = dirpath / f"{strategy_id}_run{run_number}.json"
    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    return filepath


def save_parsed_run(parsed_data):
    config.PARSED_DIR.mkdir(parents=True, exist_ok=True)
    filepath = (
        config.PARSED_DIR
        / f"{parsed_data['puzzle_id']}_{parsed_data['strategy_id']}_run{parsed_data['run_number']}.json"
    )
    with open(filepath, "w") as f:
        json.dump(parsed_data, f, indent=2, default=str)
    return filepath


def run_exists(puzzle_id, strategy_id, run_number):
    path = config.RAW_DIR / puzzle_id / f"{strategy_id}_run{run_number}.json"
    return path.exists()
