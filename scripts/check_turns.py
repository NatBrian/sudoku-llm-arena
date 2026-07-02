import glob, json, os

raw_dirs = sorted(glob.glob("data/raw/*"))
for rd in raw_dirs[:1]:
    steps = sorted(glob.glob(os.path.join(rd, "*.json")))
    if not steps:
        continue
    raw = json.load(open(steps[0]))
    turns = raw.get("turns", [])
    print("Step files: {}".format(len(steps)))
    print("turns count: {}".format(len(turns)))
    if turns:
        t0 = turns[0]
        print("First turn keys: {}".format(list(t0.keys())))
        # Check raw_response
        if "raw_response" in t0:
            rr = t0["raw_response"]
            if isinstance(rr, dict):
                print("raw_response keys: {}".format(list(rr.keys())))
                if "choices" in rr:
                    if isinstance(rr["choices"], list) and len(rr["choices"]) > 0:
                        print("choices[0] keys: {}".format(list(rr["choices"][0].keys())))
                        msg = rr["choices"][0].get("message", {})
                        print("message keys: {}".format(list(msg.keys())))
            else:
                print("raw_response type: {}".format(type(rr).__name__))
        if "reasoning" in t0:
            r = t0["reasoning"]
            print("First turn reasoning (first 200): {}".format(str(r)[:200]))
