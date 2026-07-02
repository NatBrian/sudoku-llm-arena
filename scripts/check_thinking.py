import glob, json, os

raw_dirs = sorted(glob.glob("data/raw/*"))
for rd in raw_dirs[:3]:
    steps = sorted(glob.glob(os.path.join(rd, "*.json")))
    if not steps:
        continue
    raw = json.load(open(steps[0]))
    turns = raw.get("turns", [])
    dname = os.path.basename(rd)
    if turns:
        t0 = turns[0]
        rr = t0.get("raw_response", {})
        content = rr.get("content", "")
        rc = rr.get("reasoning_content", "")
        print("{}:".format(dname))
        print("  content len={}, first 100: {}".format(len(content), content[:100]))
        print("  reasoning_content len={}, first 100: {}".format(len(rc), rc[:100]))
        print("  reasoning_content empty? {}".format(len(rc) == 0))
        print("  reasoning_raw len={}".format(len(t0.get("reasoning_raw", ""))))
