from pathlib import Path

from repilot.evaluation import META_CONFIG, evaluate_runners
from repilot.runner import Runner

from . import rq_utils


class Names:
    VANILLA = "d4j1-codet5-vanilla"
    NOMEM = "d4j1-codet5-nomem"
    MEM = "d4j1-codet5-mem"
    REPILOT = "d4j1-codet5-repilot"


class CorrectFixes:
    CORRECT_FIXES_ROOT = Path("data/correct-patches/rq3")
    VANILLA = rq_utils.get_correct_fixes(CORRECT_FIXES_ROOT / Names.VANILLA)
    NOMEM = rq_utils.get_correct_fixes(CORRECT_FIXES_ROOT / Names.NOMEM)
    MEM = rq_utils.get_correct_fixes(CORRECT_FIXES_ROOT / Names.MEM)
    REPILOT = rq_utils.get_correct_fixes(CORRECT_FIXES_ROOT / Names.REPILOT)


class Runners:
    RUNNER_ROOT = Path("data/large/generated-and-validated-patches/rq3")
    VANILLA = RUNNER_ROOT / Names.VANILLA
    NOMEM = RUNNER_ROOT / Names.NOMEM
    MEM = RUNNER_ROOT / Names.MEM
    REPILOT = RUNNER_ROOT / Names.REPILOT


DATA = [
    (Runners.VANILLA, CorrectFixes.VANILLA),
    (Runners.NOMEM, CorrectFixes.NOMEM),
    (Runners.MEM, CorrectFixes.MEM),
    (Runners.REPILOT, CorrectFixes.REPILOT),
]

RUNNERS = [Runner.load(Path(dir), META_CONFIG) for dir, _ in DATA]
CORRECT_FIXES = [len(correct_fixes) for _, correct_fixes in DATA]

evaluate_runners(
    RUNNERS, CORRECT_FIXES, title="Table 3: Component contribution of Repilot"
)
