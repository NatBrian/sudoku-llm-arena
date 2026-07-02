from pathlib import Path

parsed_dir = Path("data/parsed")
raw_dir = Path("data/raw")

# Build set of valid parsed file stems
valid_stems = set()
for f in parsed_dir.glob("*.json"):
    valid_stems.add(f.stem)

# Delete raw files that have no matching parsed file
deleted = 0
for puzzle_dir in sorted(raw_dir.iterdir()):
    if not puzzle_dir.is_dir():
        continue
    for raw_file in sorted(puzzle_dir.glob("*.json")):
        # parsed filename is: {puzzle_id}_{strategy}_run{N}.json  
        # raw filename is: {strategy}_run{N}.json
        # So parsed_stem = {puzzle_id}_{raw_stem}
        puzzle_id = puzzle_dir.name
        parsed_stem = f"{puzzle_id}_{raw_file.stem}"
        if parsed_stem not in valid_stems:
            raw_file.unlink()
            deleted += 1
            print(f"Deleted raw: {puzzle_id}/{raw_file.name}")

print(f"Deleted {deleted} raw files")

# Clean empty dirs
for d in sorted(raw_dir.iterdir()):
    if d.is_dir() and not list(d.iterdir()):
        d.rmdir()
        print(f"Removed empty dir {d.name}")

print("Done")
