import glob, json, os

raw_dirs = sorted(glob.glob("data/raw/*"))
for rd in raw_dirs:
    dname = os.path.basename(rd)
    steps = sorted(glob.glob(os.path.join(rd, "*.json")))
    if not steps:
        continue
    # Check step_000
    raw = json.load(open(steps[0]))
    keys = list(raw.keys())
    print("{}: keys={}".format(dname, keys))
    # Check result key
    if "result" in raw:
        res = raw["result"]
        if isinstance(res, dict):
            print("  result keys: {}".format(list(res.keys())))
        else:
            print("  result type: {}".format(type(res).__name__))
    # Check meta
    if "meta" in raw:
        meta = raw["meta"]
        print("  meta: {}".format(meta))
    break  # just first one
