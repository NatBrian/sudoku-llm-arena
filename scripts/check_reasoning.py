import glob, json, os

files = sorted(glob.glob("data/parsed/*.json"))
solved = [json.load(open(f)) for f in files if json.load(open(f)).get("solved")]
for pfx in ["4x4","6x9","9x9"]:
    sz = [f for f in solved if f.get("puzzle_file","").startswith("data/puzzles/"+pfx)]
    tot = [f for f in [json.load(open(f)) for f in files] if f.get("puzzle_file","").startswith("data/puzzles/"+pfx)]
    print(f'{pfx}: {len(sz)}/{len(tot)} solved')

# Check raw files for reasoning_content
raw_dirs = sorted(glob.glob("data/raw/*"), key=os.path.getmtime, reverse=True)
for rd in raw_dirs[:3]:
    steps = sorted(glob.glob(rd+"/*.json"))
    if not steps:
        continue
    raw = json.load(open(steps[0]))
    print(f"\n{rd}: has reasoning_content={'reasoning_content' in raw}")
    print(f"  has content={'content' in raw}")
    content = raw.get("content","")
    rc = raw.get("reasoning_content","")
    print(f"  content len={len(content)}, reasoning_content len={len(rc)}")
