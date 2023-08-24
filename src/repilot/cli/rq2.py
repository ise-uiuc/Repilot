import statistics as st
from pathlib import Path

from rich.console import Console
from rich.table import Table

from repilot.evaluation import META_CONFIG
from repilot.runner import Runner

from . import rq_utils

CompilationRateResult = tuple[str, str, str, str]


def alpha_repair_compilation_rates() -> CompilationRateResult:
    compile_rates_1000: list[float] = []
    compile_rates_100: list[float] = []
    compile_rates_30: list[float] = []
    compile_rates_5000: list[float] = []

    for file in Path("data/large/uniapr_pfl").glob("*result.txt"):
        assert any(
            file.name.startswith(bug_id) for bug_id in rq_utils.D4J1_CONSIDERED_BUGS
        )
        t = file.read_text()
        # logging_info = t.split(
        #     "----------------- FINISH VALIDATION OF PATCHES -----------------"
        # )[-1]
        patches = t.split("----------------------------------------")
        top_100_compile = 0
        top_30_compile = 0
        top_1000_compile = 0
        compiled = 0
        total = 0
        for patch in patches:
            patch_number = 0
            if "Patch Number :" in patch:
                for line in patch.splitlines():
                    if "Patch Number :" in line:
                        patch_number = int(line.split("Patch Number :")[-1].strip())
                if "Compiled Error" not in patch and "Syntax Error" not in patch:
                    assert patch_number != 0
                    if patch_number <= 100:
                        top_100_compile += 1
                    if patch_number <= 30:
                        top_30_compile += 1
                    if patch_number <= 1000:
                        top_1000_compile += 1
                    compiled += 1
                if patch_number > 5000:
                    break
                total += 1

        if total == 0:
            continue

        compile_rates_5000.append(compiled / total)
        compile_rates_1000.append(top_1000_compile / 1000)
        compile_rates_30.append(top_30_compile / 30)
        compile_rates_100.append(top_100_compile / 100)

    return (
        f"{st.mean(compile_rates_30) * 100:.0f}%",
        f"{st.mean(compile_rates_100) * 100:.0f}%",
        f"{st.mean(compile_rates_1000) * 100:.0f}%",
        f"{st.mean(compile_rates_5000) * 100:.0f}%",
    )


ALPHAREPAIR_RATES = alpha_repair_compilation_rates()
SEQUENCER_RATES = ("33%", "-", "-", "-")
COCONUT_RATES = ("24%", "15%", "6%", "3%")
CURE_RATES = ("39%", "28%", "14%", "9%")
# the third element of the tuple is actually the top-200 compilation rate for RewardRepair
REWARDREPAIR_RATES = ("45%", "38%", "33%", "-")

path = "data/large/generated-and-validated-patches/rq2/d4j1-codet5-template-repilot"
runner = Runner.load(Path(path), META_CONFIG)
repilot_rates = [
    f"{runner.top_k_comp_point(top_k).unique_compilation_rate() * 100:.0f}%"
    for top_k in [30, 100, 1000, 5000]
]

TOOLS_AND_RATES = [
    ("Sequencer", SEQUENCER_RATES),
    ("CoCoNuT", COCONUT_RATES),
    ("CURE", CURE_RATES),
    ("AlphaRepair", ALPHAREPAIR_RATES),
    ("RewardRepair", REWARDREPAIR_RATES),
    (rq_utils.TOOL_NAME, repilot_rates),
]

console = Console()
table = Table(title="Table 2: Comparison with existing APR tools on compilation rate")
table.add_column("Tool")
table.add_column("Top-30")
table.add_column("Top-100")
table.add_column("Top-1000")
table.add_column("Top-5000")

for tool, (top_30, top_100, top_1000, top_5000) in TOOLS_AND_RATES:
    table.add_row(tool, top_30, top_100, top_1000, top_5000)
console.print(table)
