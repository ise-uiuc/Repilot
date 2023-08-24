from pathlib import Path

from repilot.evaluation import META_CONFIG, evaluate_runners
from repilot.runner import Runner

from . import rq_utils

CORRECT_FIXES_ROOT = Path("data/correct-patches/rq4")
RUNNER_ROOT = Path("data/large/generated-and-validated-patches/rq4")
NAMES = [
    "d4j1-codet5-vanilla",
    "d4j1-codet5-repilot",
    "d4j2-codet5-vanilla",
    "d4j2-codet5-repilot",
    "d4j1-incoder-vanilla",
    "d4j1-incoder-repilot",
    "d4j2-incoder-vanilla",
    "d4j2-incoder-repilot",
]


runners: list[Runner] = []
n_correct_fixes: list[int] = []
for name in NAMES:
    correct_fixes = rq_utils.get_correct_fixes(CORRECT_FIXES_ROOT / name)
    n_correct_fixes.append(len(correct_fixes))
    path = RUNNER_ROOT / name
    assert path.exists()
    runner = Runner.load(path, META_CONFIG)
    runners.append(runner)


evaluate_runners(
    runners,
    n_correct_fixes,
    title="Table 4: Generalizability of Repilot across both subjects of bugs and models",
)
