from pathlib import Path
import json

paths = list(Path(".").glob("**/bug.json"))
for path in paths:
    with open(path) as f:
        d = json.load(f)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
