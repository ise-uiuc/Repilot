import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = list(root.glob("**/bug.json"))
for path in paths:
    with open(path) as f:
        d = json.load(f)
        text = "/JawTitan/yuxiang-data/Developer/d4j-checkout/"
        assert d["file"]["path"].startswith(text)
        d["file"]["path"] = d["file"]["path"][len(text) :]
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

f1 = root / "repair_transformed.json"
assert f1.exists()
f1.unlink()
