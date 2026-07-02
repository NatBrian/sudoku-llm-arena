import glob, json

files = sorted(glob.glob("data/parsed/*.json"))
print(f"Total parsed files: {len(files)}")

solved = []
for f in files:
    data = json.load(open(f))
    if data.get("solved"):
        solved.append(data)

print(f"Solved: {len(solved)}")

# Break down by puzzle name prefix
from collections import Counter
cnt = Counter()
sz_solved = Counter()
for data in [json.load(open(f)) for f in files]:
    pf = data.get("puzzle_file", "")
    for p in ["4x4", "6x6", "9x9"]:
        if p in pf:
            cnt[p] += 1
            if data.get("solved"):
                sz_solved[p] += 1
            break

for p in ["4x4", "6x6", "9x9"]:
    print(f"  {p}: {sz_solved.get(p,0)}/{cnt.get(p,0)} solved")

# Check raw files for reasoning structure
import os
raw_dirs = sorted(glob.glob("data/raw/*"), key=os.path.getmtime, reverse=True)
for rd in raw_dirs[:3]:
    steps = sorted(glob.glob(rd + "/*.json"))
    if not steps:
        continue
    raw = json.load(open(steps[0]))
    has_rc = "reasoning_content" in raw
    has_c = "content" in raw
    content = raw.get("content", "")
    rc = raw.get("reasoning_content", "")
    print(f"\n{rd}:")
    print(f"  has reasoning_content: {has_rc}")
    print(f"  has content: {has_c}")
    print(f"  content length: {len(content)}")
    print(f"  reasoning_content length: {len(rc)}")

# Check a parsed step
if files:
    data = json.load(open(files[0]))
    steps = data.get("steps", [])
    if steps:
        print(f"\nFirst parsed step keys: {list(steps[0].keys())}")
        print(f"Has reasoning: {'reasoning' in steps[0]}")
        r = steps[0].get("reasoning", "")
        print(f"Reasoning (first 200): {str(r)[:200]}")
