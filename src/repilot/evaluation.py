from dataclasses import dataclass
from itertools import groupby, zip_longest
from pathlib import Path
from typing import TypeVar, cast

from rich.console import Console
from rich.table import Table

from .config import MetaConfig
from .d4j import Defects4J
from .results import concat_hunks
from .runner import Runner

Datapoint = TypeVar("Datapoint")

Project = str
RunnerId = str

META_CONFIG = MetaConfig.from_json_file(Path("meta_config.json"))
D4J = Defects4J(
    META_CONFIG.d4j_home, META_CONFIG.d4j_checkout_root, META_CONFIG.java8_home
)


def transpose(
    data_grouped_by_project: list[
        tuple[RunnerId, list[tuple[Project, dict[str, Datapoint]]]]
    ]
) -> list[tuple[Project, list[tuple[RunnerId, dict[str, Datapoint]]]]]:
    data = [
        (project_id, runner_id, data)
        for runner_id, project_data in data_grouped_by_project
        for (project_id, data) in project_data
    ]
    data.sort(key=lambda tp: tp[0])
    results: list[tuple[Project, list[tuple[RunnerId, dict[str, Datapoint]]]]] = []
    for project, project_data in groupby(data, lambda tp: tp[0]):
        results.append(
            (
                project,
                [
                    (runner_id, result_dict)
                    for _, runner_id, result_dict in project_data
                ],
            )
        )
    return results


@dataclass(frozen=True)
class ValidationResult:
    compilation_rate: float
    plausible_rate: float
    n_plausible_fixes: int


@dataclass(frozen=True)
class EvaluationResult:
    generation_time: float
    validation_result: ValidationResult | None
    n_correct_fixes: int | None

    @staticmethod
    def format_number(f: float | int | None) -> str:
        if isinstance(f, int):
            return str(f)
        if f is None:
            return "-"
        if f >= 1.0:
            return f"{f:.2f}"
        else:
            return f"{f:.3f}"

    @staticmethod
    def format_validation_result(
        result: ValidationResult | None,
    ) -> tuple[str, str, str]:
        format_number = EvaluationResult.format_number
        if result is None:
            return ("-", "-", "-")
        return (
            f"{result.compilation_rate * 100:.1f}%",
            f"{format_number(result.plausible_rate * 100)}%",
            format_number(result.n_plausible_fixes),
        )

    def formatted(self) -> tuple[str, str, str, str, str]:
        format_number = EvaluationResult.format_number
        format_validation_result = EvaluationResult.format_validation_result
        return (
            format_number(self.generation_time) + "s",
            *format_validation_result(self.validation_result),
            format_number(self.n_correct_fixes),
        )


def get_runner_evaluation(
    runner: Runner, n_correct_fixes: int | None
) -> EvaluationResult:
    generation_summary = runner.evaluate_generation_summary()
    generation_time = generation_summary.average_time
    if runner.report.validation_result is not None:
        _proj_summary, validation_summary = runner.evaluate_validation_summary()
        compilation_rate = validation_summary.unique_compilation_rate()
        plausible_rate = validation_summary.unique_plausible_rate()
        proj_plausible_fixes = {
            proj: {
                bug_id: [concat_hunks(patch) for patch in patches]
                for bug_id, patches in proj_values.items()
            }
            for proj, proj_values in runner.get_plausible_patches_grouped().items()
        }
        n_plausible_fixes = sum(
            1 for fixes in proj_plausible_fixes.values() for _ in fixes
        )
        validation_result = ValidationResult(
            compilation_rate, plausible_rate, n_plausible_fixes
        )
    else:
        validation_result = None
    return EvaluationResult(generation_time, validation_result, n_correct_fixes)


def evaluate_runners(
    runners: list[Runner],
    n_correct_fixes_list: list[int],
    title: str = "Repilot Evaluation Results",
) -> None:
    runner_evaluations = (
        get_runner_evaluation(cast(Runner, runner), n_correct_fixes)
        for runner, n_correct_fixes in zip_longest(
            runners, n_correct_fixes_list, fillvalue=None
        )
    )
    table = Table(title=title)
    table.add_column("Tag")
    table.add_column("Average Gen Time")
    table.add_column("%Compilable Patches")
    table.add_column("%Plausible Patches")
    table.add_column("#Plausible Fixes")
    table.add_column("#Correct Fixes")

    for runner, result in zip(runners, runner_evaluations):
        tag = runner.report.root.name
        table.add_row(tag, *result.formatted())
    console = Console()
    console.print(table)
