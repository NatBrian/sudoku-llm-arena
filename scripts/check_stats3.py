import glob, json, os

# Check new runs specifically
raw_dirs = sorted(glob.glob("data/raw/*"))
print("All raw dirs:")
for rd in raw_dirs:
    dname = os.path.basename(rd)
    steps = sorted(glob.glob(os.path.join(rd, "*.json")))
    if steps:
        raw = json.load(open(steps[0]))
        keys = list(raw.keys())
        has_c = "content" in raw
        has_rc = "reasoning_content" in raw
        print("  {}: {} steps, keys={}, has_content={}, has_reasoning_content={}".format(
            dname, len(steps), keys[:5], has_c, has_rc))

# Check parsed data fields too
print("\nParsed data fields check:")
files = sorted(glob.glob("data/parsed/*.json"))
for f in files[-5:]:
    data = json.load(open(f))
    name = os.path.basename(f)
    print("{}: solved={}, total_turns={}, total_errors={}".format(
        name, data.get("solved"), data.get("total_turns"), data.get("total_errors")))
