"""Generate only missing GIFs."""
import os, glob, json
from src.visualize.gif import render_run_to_gif
from src import config

parsed = glob.glob(str(config.PARSED_DIR / "*.json"))
existing = {os.path.basename(f).replace(".gif","") for f in glob.glob(str(config.OUTPUT_DIR / "gifs" / "*.gif"))}

missing = []
for path in sorted(parsed):
    stem = os.path.basename(path).replace(".json","")
    if stem not in existing:
        parts = stem.split("_")
        if len(parts) >= 4:
            puzzle_id = f"{parts[0]}_{parts[1]}"
            strategy_id = parts[2]
            run_number = int(parts[3].replace("run",""))
            missing.append((puzzle_id, strategy_id, run_number))

print(f"Missing GIFs: {len(missing)}")
for i, (pid, sid, rn) in enumerate(missing):
    result = render_run_to_gif(pid, sid, rn)
    if (i+1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(missing)}")
