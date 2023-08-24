import itertools
import json
from functools import cmp_to_key
from pathlib import Path

import pandas
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from venn import venn

from . import rq_utils


def get_baseline_fixes(csv_path: Path, filter: set[str]) -> dict[str, set[str]]:
    dl_methods = [
        "AlphaRepair",
        "CURE",
        "CoCoNuT",
        "CURE",
        "Recoder",
        "SEQUENCER",
        "DLFix",
        "RewardRepair",
    ]
    nondl_methods = [
        "TBar",
        "Tbar-68",
        "Prapr2",
        "AVATAR",
        "SimFix",
        "FixMiner",
        "capgen",
        "jaid",
        "sketchfix",
        "Nopol",
        "GenProg-A",
        "jGenProg",
        "jKali",
        "jMutRepair",
        "Kali-A",
    ]

    def parse_bug_id(text: str) -> str:
        return text.split("(")[0].replace("_", "-").replace(" ", "-")

    data = pandas.read_csv(csv_path, header=0)
    tool_fixed_bugs: dict[str, set[str]] = {}
    for col in data.columns:
        fixed_bugs = set(parse_bug_id(x) for x in data[col].dropna().tolist())
        if col in dl_methods:
            tool_fixed_bugs[col] = fixed_bugs & filter
        elif col in nondl_methods:
            tool_fixed_bugs[col] = fixed_bugs & filter
    return tool_fixed_bugs


def _load_repilot_fixes():
    data = pandas.read_csv("csv/repilot-correct-dfj12-single-hunk.csv", header=0)
    correct_fixes = set(data["Correct"].dropna().tolist())

    data = pandas.read_csv("csv/repilot-correct-dfj2-single-line.csv", header=0)
    correct_fixes_dfj2 = set(data["Correct"].dropna().tolist())

    return correct_fixes, correct_fixes_dfj2


REPILOT_D4J1_CORRECT_FIXES = rq_utils.get_correct_fixes(
    Path("data/correct-patches/rq1/d4j1-codet5-template-repilot")
)
REPILOT_D4J2_CORRECT_FIXES = rq_utils.get_correct_fixes(
    Path("data/correct-patches/rq1/d4j2-codet5-template-repilot")
)


D4J1_BASELINE_FIXES = get_baseline_fixes(
    Path("data/baseline-patch-correctness-results/d4j1.2.csv"),
    rq_utils.D4J1_CONSIDERED_BUGS,
)
D4J2_BASELINE_FIXES = get_baseline_fixes(
    Path("data/baseline-patch-correctness-results/d4j2.0.csv"),
    rq_utils.D4J2_CONSIDERED_BUGS,
)

D4J1_OVERLAPPING_BUGS = set[str](
    rq_utils.load_json("data/codet5-data-overlap/d4j1.2.json")
)
D4J2_OVERLAPPING_BUGS = set[str](
    rq_utils.load_json("data/codet5-data-overlap/d4j2.0.json")
)


# Tool -> #Fixes on D4J1.2 and D4J2.0
def get_all_fixes() -> dict[str, tuple[set[str], set[str]]]:
    all_fixes: dict[str, tuple[set[str], set[str]]] = {}
    for tool in itertools.chain(D4J1_BASELINE_FIXES.keys(), D4J2_BASELINE_FIXES.keys()):
        if tool == "TBar":
            continue
        d4j1_fixes = D4J1_BASELINE_FIXES.get(tool, set())
        if tool == "Tbar-68":
            tool = "TBar"
        d4j2_fixes = D4J2_BASELINE_FIXES.get(tool, set())
        all_fixes[tool] = (d4j1_fixes, d4j2_fixes)
    all_fixes[rq_utils.TOOL_NAME] = (
        REPILOT_D4J1_CORRECT_FIXES,
        REPILOT_D4J2_CORRECT_FIXES,
    )
    return {
        tool: result for tool, result in all_fixes.items() if tool in CONSIDERED_TOOLS
    }


CONSIDERED_TOOLS = {
    "AlphaRepair",
    "RewardRepair",
    "Recoder",
    "TBar",
    # "Tbar-68",
    "CURE",
    "CoCoNuT",
    "Prapr2",
    "DLFix",
    rq_utils.TOOL_NAME,
}
ALL_FIXES = get_all_fixes()


def generate_fig_6_venn():
    venn_fixes = dict[str, set[str]]()
    venn_fixes_all: dict[str, set[str]] = {"Others": set()}
    for tool, (d4j1_fixes, _) in ALL_FIXES.items():
        if tool in [
            "AlphaRepair",
            "Recoder",
            "CURE",
            "RewardRepair",
            rq_utils.TOOL_NAME,
        ]:
            venn_fixes[tool] = d4j1_fixes
        if tool in ["AlphaRepair", "Recoder", "RewardRepair", rq_utils.TOOL_NAME]:
            venn_fixes_all[tool] = d4j1_fixes
        else:
            venn_fixes_all["Others"] |= d4j1_fixes

    # venn_fixes[rq_utils.TOOL_NAME] = correct_fixes
    venn(venn_fixes, fontsize=30, figsize=(20, 20), cmap="viridis")
    print("Creating venn diagrams..")
    rq_utils.PLOT_DIR.mkdir(exist_ok=True)
    path_a = rq_utils.PLOT_DIR / "venn-with-learning-based-d4j1.pdf"
    plt.savefig(path_a)

    # venn_fixes_all[rq_utils.TOOL_NAME] = correct_fixes
    venn(venn_fixes_all, fontsize=30, figsize=(20, 20), cmap="viridis")
    path_b = rq_utils.PLOT_DIR / "venn-with-all.pdf"
    plt.savefig(path_b)

    prompt = f"Venn diagrams in Figure 6 (a) and (b) saved to {path_a} and {path_b}."
    print(prompt)


def generate_fixes_table(
    title: str, data: dict[str, tuple[set[str], set[str]]], filtered_bugs: set[str]
):
    def compare_baseline(
        lhs: tuple[set[str], set[str]], rhs: tuple[set[str], set[str]]
    ) -> int:
        lhs_d4j1, lhs_d4j2 = lhs
        rhs_d4j1, rhs_d4j2 = rhs
        if len(lhs_d4j1) < len(rhs_d4j1):
            return -1
        if len(lhs_d4j1) > len(rhs_d4j1):
            return 1
        assert len(lhs_d4j1) == len(rhs_d4j1)
        if len(lhs_d4j2) < len(rhs_d4j2):
            return -1
        if len(lhs_d4j2) > len(rhs_d4j2):
            return 1
        assert len(lhs_d4j2) == len(rhs_d4j2)
        return 0

    all_fixes = list(data.items())
    all_fixes.sort(key=cmp_to_key(lambda x, y: compare_baseline(x[1], y[1])))  # type: ignore
    all_fixes = [
        (
            tool,
            (
                d4j1_fixes - (d4j1_fixes & filtered_bugs),
                d4j2_fixes - (d4j2_fixes & filtered_bugs),
            ),
        )
        for tool, (d4j1_fixes, d4j2_fixes) in all_fixes
    ]

    table = Table(title=title)
    table.add_column("Tool")
    table.add_column("D4J 1.2")
    table.add_column("D4J 2.0")
    table.add_column("Total")
    for tool, (d4j1_fixes, d4j2_fixes) in all_fixes:
        table.add_row(
            tool,
            str("-" if len(d4j1_fixes) == 0 else len(d4j1_fixes)),
            str("-" if len(d4j2_fixes) == 0 else len(d4j2_fixes)),
            str(
                "-"
                if len(d4j1_fixes) == 0 or len(d4j2_fixes) == 0
                else len(d4j1_fixes | d4j2_fixes)
            ),
        )
    console = Console()
    console.print(table)


def get_in_dataset_patches():
    ret_set = set()
    data = pandas.read_csv("csv/d4j-overlap.csv", header=0).fillna(0)
    for index, type in enumerate(data["Result"]):
        if type == "Fixed function":
            ret_set.add(data["Bug"][index].replace(" ", "-"))
    return ret_set


def get_in_dataset_patches_2():
    ret_set = set()
    data = pandas.read_csv("csv/d4j-overlap.csv", header=0).fillna(0)
    for index, type in enumerate(data["Result-2"]):
        if type == "Fixed function":
            ret_set.add(data["Bug-2"][index].replace(" ", "-"))
    return ret_set


def check_overlap():
    correct_fixes, correct_fixes_dfj2 = _load_repilot_fixes()
    seed_fix_1 = get_in_dataset_patches()
    seed_fix_2 = get_in_dataset_patches_2()

    print(len(correct_fixes - seed_fix_1))
    print(len(correct_fixes_dfj2 - seed_fix_2))

    print(correct_fixes_dfj2 & seed_fix_2)


def generate_table_1():
    generate_fixes_table(
        "Table 1: Number of correct fixes on Defects4j 1.2 single-hunk and Defects4j 2.0 single-line bugs",
        ALL_FIXES,
        set(),
    )


def generate_threats():
    generate_fixes_table(
        "Number of correct fixes by removing bugs with overlapping developer fixes in the CodeT5 training data",
        ALL_FIXES,
        D4J1_OVERLAPPING_BUGS | D4J2_OVERLAPPING_BUGS,
    )


generate_table_1()
generate_fig_6_venn()
generate_threats()
