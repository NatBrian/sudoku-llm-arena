import glob, json, os
from collections import Counter

files = glob.glob("data/parsed/*.json")
cnt = Counter()
solved = Counter()
for f in files:
    name = os.path.basename(f)
    parts = name.split("_")
    pfx = parts[0]
    cnt[pfx] += 1
    data = json.load(open(f))
    if data.get("solved"):
        solved[pfx] += 1
for p in ["4x4", "6x6", "9x9", "9x9-evil"]:
    print("{}: {}/{} solved".format(p, solved.get(p,0), cnt.get(p,0)))

# Check raw dirs
raw_dirs = sorted([d for d in glob.glob("data/raw/*") if os.path.isdir(d)])
for rd in raw_dirs[:5]:
    steps = sorted(glob.glob(os.path.join(rd, "*.json")))
    if not steps:
        continue
    raw = json.load(open(steps[0]))
    has_rc = "reasoning_content" in raw
    has_c = "content" in raw
    keys = list(raw.keys())
    rc_len = len(str(raw.get("reasoning_content", "")))
    c_len = len(str(raw.get("content", "")))
    dname = os.path.basename(rd)
    print("{}: reasoning_content={}, content={}, keys={}, rc_len={}, c_len={}".format(
        dname, has_rc, has_c, keys[:5], rc_len, c_len))

# Check a successful run for solved count by strategy
print("\nSolved by strategy:")
strat_solved = Counter()
for f in files:
    data = json.load(open(f))
    if data.get("solved"):
        strat_solved[data.get("strategy", "?")] += 1
for s, n in strat_solved.most_common():
    total = sum(1 for f in files if json.load(open(f)).get("strategy") == s)
    print("  {}: {}/{}".format(s.ljust(20), n, total))
