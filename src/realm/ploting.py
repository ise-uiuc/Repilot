import json
import os
from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable, TypeVar, cast

import torch
from matplotlib import pyplot as plt

from . import utils
from .config import MetaConfig
from .d4j import Defects4J
from .results import GenerationDatapoint, ValidationDatapoint, concat_hunks
from .runner import Runner

Datapoint = TypeVar("Datapoint")

Project = str
RunnerId = str

COLORS = ["#1D1F5F", "#FF99FF", "#6EAF46", "#FAC748"]
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


def plot_runners(
    tags: list[str],
    runners: list[Runner],
) -> None:
    assert len(tags) == len(runners)
    generation_results = transpose(
        [
            (tag, runner.evaluate_generation_grouped())
            for tag, runner in zip(tags, runners)
        ]
    )

    def generation_datapoint_getter(datapoint: GenerationDatapoint) -> list[int]:
        return [
            datapoint.n_unique - datapoint.n_unfinished - datapoint.n_pruned,
            # datapoint.n_unfinished,
            # datapoint.n_pruned,
            datapoint.n_total - datapoint.n_unique,
        ]

    # generation_names = ["Unique", "Unfinished", "Pruned", "Duplicate"]
    generation_names = ["Unique", "Duplicate"]

    validated_runners = [
        runner for runner in runners if runner.report.validation_result is not None
    ]
    raw_validation_results = [
        (tag, runner.evaluate_validation_grouped())
        for tag, runner in zip(tags, validated_runners)
    ]
    validation_results = transpose(raw_validation_results)
    # breakpoint()

    target = Path(os.getenv("PLOT_DIR", "plots"))
    target.mkdir(exist_ok=True)

    print("Generation Summary")
    for tag, runner in zip(tags, runners):
        print(f"{tag} Unique: {runner.evaluate_unique_generation_summary()}")
        print(f"{tag} Total: {runner.evaluate_generation_summary()}")

    # Print averaged summary
    # TODO: now the points' total and unique is the same. To change this, modify
    # `map_validaton_datapoint`
    print("Validation Summary")
    for tag, runner, (_, raw_results) in zip(tags, runners, raw_validation_results):
        if runner.report.validation_result is None:
            continue
        proj_summary, summary = runner.evaluate_validation_summary()

        plausible_fixes = {
            proj: {
                bug_id: [concat_hunks(patch) for patch in patches]
                for bug_id, patches in proj_values.items()
            }
            for proj, proj_values in runner.get_plausible_patches_grouped().items()
        }

        # plausible_root = Path("plausible_patches")
        # assert runner.report.transformed_result is not None
        # transformed = runner.report.transformed_result.result_dict
        # for proj, proj_values in runner.get_plausible_patches_grouped().items():
        #     plausible_root.mkdir(exist_ok=True)
        #     for bug_id, patches in proj_values.items():
        #         bug_id_dir = plausible_root / bug_id
        #         bug_id_dir.mkdir()
        #         patch_strs: list[str] = []
        #         bugs, _ = transformed[bug_id]
        #         assert len(bugs) == 1
        #         diffs: list[str] = []
        #         for patch_id, patch in enumerate(patches):
        #             assert len(patch) == 1
        #             patch_content = concat_hunks(patch)
        #             patch_strs.append(patch_content)
        #             patch_file = (bug_id_dir / str(patch_id)).with_suffix(".txt")
        #             patch_file.write_text(patch_content)
        #             patch_text_file = patch[0].compute_patch(bugs[0])
        #             assert patch_text_file is not None
        #             diff = utils.diff(
        #                 bugs[0].content,
        #                 patch_text_file.content,
        #                 lhs_msg=f"bug/{bugs[0]._path}",
        #                 rhs_msg=f"fix/{bugs[0]._path}",
        #             )
        #             # diff_file = (bug_id_dir / str(patch_id)).with_suffix(".diff")
        #             # diff_file.write_text(diff)
        #             # diffs.append(diff)
        #         integrated_file = bug_id_dir / "integrated.txt"
        #         integrated_file.write_text(utils.RULE.join(patch_strs))
        #         (bug_id_dir / f"reference.patch").write_text(D4J.get_patch(bug_id))
        #         # integrated_diff_file = bug_id_dir / "integrated.diff"
        #         # integrated_diff_file.write_text(utils.RULE.join(diffs))

        # {
        #     proj: [
        #         bug_id
        #         for bug_id, bug_id_result in bug_id_results.items()
        #         if bug_id_result.n_test_success > 0
        #     ]
        #     for proj, bug_id_results in raw_results
        # }
        # n_plausible = {proj: v.n_test_success for proj, v in proj_summary.items()}
        print(tag, "Metadata", summary)
        print(
            tag,
            f"Compilation rate: {summary.unique_compilation_rate()}",
            f"Plausible rate: {summary.unique_plausible_rate()}",
            f"Plausible fixes: {sum(1 for fixes in plausible_fixes.values() for fix in fixes)}",
        )
        print()
        print(
            {
                proj: (
                    proj_sum.unique_compilation_rate(),
                    proj_sum.gen_datapoint.n_unique,
                )
                for proj, proj_sum in proj_summary.items()
            }
        )
        print()
        print("Plausible fixes (project)")
        print({k: len(v) for k, v in plausible_fixes.items()})
        with open(target / f"{tag}_plausible_details.json", "w") as f:
            json.dump(plausible_fixes, f, indent=2)
        print()
        print()
    if os.getenv("PLOT") is None:
        return

    def validation_datapoint_getter(datapoint: ValidationDatapoint) -> list[int]:
        return [
            datapoint.n_test_success,
            # datapoint.n_comp_success - datapoint.n_test_success,
            # datapoint.n_parse_success - datapoint.n_comp_success,
            # datapoint.gen_datapoint.n_total - datapoint.n_parse_success,
            # - datapoint.gen_datapoint.n_unfinished
            # - datapoint.gen_datapoint.n_pruned
            # - datapoint.n_test_success
            # - datapoint.n_comp_success
            # - datapoint.n_parse_success,
        ]

    def validation_avg_plausible(datapoint: ValidationDatapoint) -> list[int]:
        # TODO: fix the type
        return [
            datapoint.n_test_success / datapoint.gen_datapoint.n_total,  # type: ignore # fmt: skip
            # datapoint.n_comp_success - datapoint.n_test_success,
            # datapoint.n_parse_success - datapoint.n_comp_success,
            # datapoint.gen_datapoint.n_total - datapoint.n_parse_success,
            # - datapoint.gen_datapoint.n_unfinished
            # - datapoint.gen_datapoint.n_pruned
            # - datapoint.n_test_success
            # - datapoint.n_comp_success
            # - datapoint.n_parse_success,
        ]

    # validation_names = ["Test Suc.", "Comp. Suc.", "Parse. Suc", "Total"]
    validation_names = ["Test Suc."]  # , "Comp. Suc.", "Total"]

    def validation_avg_compilable(datapoint: ValidationDatapoint) -> list[int]:
        # TODO: fix the type
        return [
            datapoint.n_comp_success / datapoint.gen_datapoint.n_unique,  # type: ignore # fmt: skip
            # datapoint.n_comp_success - datapoint.n_test_success,
            # datapoint.n_parse_success - datapoint.n_comp_success,
            # datapoint.gen_datapoint.n_total - datapoint.n_parse_success,
            # - datapoint.gen_datapoint.n_unfinished
            # - datapoint.gen_datapoint.n_pruned
            # - datapoint.n_test_success
            # - datapoint.n_comp_success
            # - datapoint.n_parse_success,
        ]

    validation_avg_compilable_names = ["Compilation Rate"]

    # validation_names = ["Test Suc.", "Comp. Suc.", "Parse. Suc", "Total"]
    validation_names = ["Test Suc."]  # , "Comp. Suc.", "Total"]
    # validation_first_plausible_results = transpose(
    #     [
    #         (
    #             tag,
    #             runner.evaluate_validation_first_one_grouped(
    #                 lambda p: p.n_test_success
    #             ),
    #         )
    #         for tag, runner in zip(tags, runners)
    #         if runner.report.validation_result is not None
    #     ]
    # )

    # def validation_plausible_getter(datapoint: ValidationDatapoint) -> list[int]:
    #     assert datapoint.n_test_success <= 1
    #     return [datapoint.gen_datapoint.n_total if datapoint.n_test_success == 1 else 0]

    validation_plausible_names = ["Plausible rate"]

    for project, generation_result in generation_results:
        plt.clf()
        plt.title(f"Generation results for {project}")
        plot_bars(
            generation_datapoint_getter,
            generation_names,
            generation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-gen-{project}.png")
    plt.close()
    for project, validation_result in validation_results:
        plt.clf()
        plt.title(f"Validation results for {project}")
        plot_bars(
            validation_datapoint_getter,
            validation_names,
            validation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-val-{project}.png")

    plt.close()
    for (
        project,
        validation_result,
    ) in validation_results:
        plt.clf()
        plt.title(f"Plausible rate for {project}")
        plot_bars(
            validation_avg_compilable,
            validation_avg_compilable_names,
            validation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-compilation_rate-{project}.png")

    plt.close()
    for (
        project,
        validation_result,
    ) in validation_results:
        plt.clf()
        plt.title(f"Plausible rate for {project}")
        plot_bars(
            validation_avg_plausible,
            validation_plausible_names,
            validation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-plausible_rate-{project}.png")


def plot_bars(
    points_getter: Callable[[Datapoint], list[int]],
    point_names: list[str],
    # 1st `str` is for the cluster name, and 2nd for the entire clusters name
    data_dicts: list[tuple[str, dict[str, Datapoint]]],
    width: float,
    cluster_in_between_gap: float,
    cluster_gap: float,
):
    """Clustered AND stacked bar chart for comparing multiple repair runs"""
    # Group the same ids
    # groups, data = cast(
    #     tuple[list[str], list[dict[str, Datapoint]]], tuple(zip(*data_dicts))
    # )
    # assert len(groups) == len(data)cess,

    # list[tuple[str, list[tuple[str, Datapoint]]]]
    # transformed: dict[str, list[tuple[str, Datapoint]]] = {}
    n_clusters = len(data_dicts)
    offsets = get_offsets(n_clusters, width, cluster_in_between_gap)
    all_labels = set(key for _, data_dict in data_dicts for key in data_dict)
    plt.figure(figsize=(16, len(all_labels)))
    label_offset = {label: idx for idx, label in enumerate(all_labels)}
    gap = cluster_gap + width * n_clusters + (n_clusters - 1) * cluster_in_between_gap
    ticks = list(utils.stride(0.0, gap, len(all_labels)))
    assert len(ticks) == len(all_labels)
    for offset, (cluster_name, cluster_dict), color in zip(offsets, data_dicts, COLORS):
        data_dict = {
            cluster_key: points_getter(cluster_datapoint)
            for cluster_key, cluster_datapoint in cluster_dict.items()
        }
        plot_bar(
            width,
            [ticks[label_offset[label]] + offset for label in data_dict.keys()],
            data_dict,
            [f"{cluster_name}-{name}" for name in point_names],
            color,
        )
    plt.xlabel("Bug IDs")
    plt.xlabel("Number of patches")
    plt.yticks(ticks, all_labels)
    plt.legend()


def plot_bar(
    width: float,
    offsets: list[float],
    data_dict: dict[str, list[int]],
    names: list[str],
    color: str,
) -> None:
    plt.style.use("ggplot")
    # labels = list(data_dict.keys())
    # offsets = [label_offset[label] + offset for label in labels]
    stacked_data = torch.tensor(list(data_dict.values()), dtype=torch.float64)
    assert stacked_data.shape[1] == len(names)
    assert stacked_data.shape[0] == len(offsets)
    starting_values = torch.zeros(len(offsets), dtype=torch.float64)
    for idx in range(len(names)):
        datapoints = stacked_data[:, idx]
        bar = plt.barh(
            offsets,
            datapoints,
            left=starting_values,
            height=width,
            color=color,
            alpha=(1.0 - idx * 0.32),
            label=names[idx],
        )
        plt.bar_label(bar)
        # for rect, offset, datapoint in zip(bar, offsets, datapoints):
        #     width = rect.get_width()
        #     y = rect.get_y()
        #     plt.text(width, y + rect.get_width() / 2, str(datapoint.item()), ha='center', va='bottom')
        # starting_values += datapoints
    # plt.tight_layout()
    # assert len(xs) == len(labels)


def get_offsets(
    n_clusters: int, width: float, in_between_gap: float
) -> Iterable[float]:
    current = -(n_clusters - 1) * (width + in_between_gap) / 2
    for _ in range(n_clusters):
        yield current
        current += width + in_between_gap


# def plot_validation(datapoints_dict: dict[str, list[ValidationDatapoint]], label: str):
#     all_xs_ys = [
#         transform_points(
#             datapoints,
#             lambda datapoint: (
#                 datapoint.gen_datapoint.n_unique,
#                 datapoint.n_comp_success,
#             ),
#         )
#         for bug_id, datapoints in datapoints_dict.items()
#     ]
#     t_all_xs_ys = torch.tensor(all_xs_ys)
#     mean = t_all_xs_ys.mean(dim=0, dtype=torch.float64)
#     plt.plot(mean[:, 0], mean[:, 1], label=label)


# all_datapoints: list[list[GenerationDatapoint]] = []
# for bug_id, datapoints in datapoints_dict.items():
#     all_datapoints.append(datapoints)
# times = []
# total = []
# for x in zip(*all_datapoints):
#     times.append(sum(dp.gen_time for dp in x) / len(x))
#     total.append(sum(dp.n_total for dp in x) / len(x))
# plt.plot(
#     times,
#     total,
#     label=bug_id,
# )
# plt.savefig("plot.png")
