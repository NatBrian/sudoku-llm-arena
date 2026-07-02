import json
from pathlib import Path

# Delete parsed + raw files for zero-turn failures
parsed_dir = Path("data/parsed")
raw_dir = Path("data/raw")

for f in sorted(parsed_dir.glob("*.json")):
    d = json.load(open(f))
    if d.get("total_turns", 0) == 0 and not d.get("solved", False):
        # Delete parsed
        f.unlink()
        # Delete raw file
        parts = f.stem.split("_")
        puzzle_id = f"{parts[0]}_{parts[1]}"
        strategy_id = parts[2]
        run_number = parts[3].replace("run", "")
        raw_path = raw_dir / puzzle_id / f"{strategy_id}_run{run_number}.raw.json"
        if raw_path.exists():
            raw_path.unlink()
        print(f"Cleared {f.stem}")

# Also clean up empty raw dirs
for d in sorted(raw_dir.iterdir()):
    if d.is_dir() and not list(d.iterdir()):
        d.rmdir()
        print(f"Removed empty dir {d.name}")

print("Done")
