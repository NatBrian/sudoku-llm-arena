from pathlib import Path
p = len(list(Path("data/parsed").glob("*.json")))
r = len(list(Path("data/raw").iterdir()))
print(f"Remaining: {p} parsed, {r} raw dirs")
