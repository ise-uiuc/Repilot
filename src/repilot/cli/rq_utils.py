import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


D4J1_CONSIDERED_BUGS = set[str](
    load_json("data/considered-bugs/d4j1.2-single-hunk.json")
)
D4J2_CONSIDERED_BUGS = set[str](
    load_json("data/considered-bugs/d4j2.0-single-line.json")
)


def get_correct_fixes(dir: Path) -> set[str]:
    result = set[str]()
    for file in dir.iterdir():
        assert file.suffix == ".md"
        result.add(file.stem)
    return result


TOOL_NAME = "Repilot"

PLOT_DIR = Path("plots")
