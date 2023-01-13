from itertools import groupby
from pathlib import Path
from typing import Callable, Iterable, TypeVar, cast

import torch
from matplotlib import pyplot as plt

from . import utils
from .results import GenerationDatapoint, ValidationDatapoint
from .runner import Runner

Datapoint = TypeVar("Datapoint")

Project = str
RunnerId = str

COLORS = ["#1D2F6F", "#F390FA", "#6EAF46", "#FAC748"]


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
    runners: list[Runner],
) -> None:
    plt.figure(figsize=(16, 30))
    generation_results = transpose(
        [
            (str(runner.report.root), runner.evaluate_generation_grouped())
            for runner in runners
        ]
    )

    def generation_datapoint_getter(datapoint: GenerationDatapoint) -> list[int]:
        return [
            datapoint.n_unique - datapoint.n_unfinished - datapoint.n_pruned,
            datapoint.n_unfinished,
            datapoint.n_pruned,
            datapoint.n_total - datapoint.n_unique,
        ]

    generation_names = ["Unique", "Unfinished", "Pruned", "Duplicate"]

    validation_results = transpose(
        [
            (str(runner.report.root), runner.evaluate_validation_grouped())
            for runner in runners
            if runner.report.validation_result is not None
        ]
    )

    def validation_datapoint_getter(datapoint: ValidationDatapoint) -> list[int]:
        return [
            datapoint.n_test_success,
            datapoint.n_comp_success - datapoint.n_test_success,
            datapoint.n_parse_success - datapoint.n_comp_success,
            datapoint.gen_datapoint.n_total - datapoint.n_parse_success,
            # - datapoint.gen_datapoint.n_unfinished
            # - datapoint.gen_datapoint.n_pruned
            # - datapoint.n_test_success
            # - datapoint.n_comp_success
            # - datapoint.n_parse_success,
        ]

    validation_names = ["Test Suc.", "Comp. Suc.", "Parse. Suc", "Total"]

    target = Path("plots")
    target.mkdir(exist_ok=True)

    for project, generation_result in generation_results:
        plt.clf()
        plt.autoscale(True)
        plot_bars(
            generation_datapoint_getter,
            generation_names,
            generation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-gen-{project}.png")
    for project, validation_result in validation_results:
        plt.clf()
        plt.autoscale(True)
        plot_bars(
            validation_datapoint_getter,
            validation_names,
            validation_result,
            5 * 1.65,
            5 * 0.4,
            5 * 1.5,
        )
        plt.savefig(target / f"plot-val-{project}.png")


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
    # assert len(groups) == len(data)
    # list[tuple[str, list[tuple[str, Datapoint]]]]
    # transformed: dict[str, list[tuple[str, Datapoint]]] = {}
    n_clusters = len(data_dicts)
    offsets = get_offsets(n_clusters, width, cluster_in_between_gap)
    all_labels = set(key for _, data_dict in data_dicts for key in data_dict)
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
    stacked_data = torch.tensor(list(data_dict.values()))
    assert stacked_data.shape[1] == len(names)
    assert stacked_data.shape[0] == len(offsets)
    starting_values = torch.zeros(len(offsets))
    for idx in range(len(names)):
        datapoints = stacked_data[:, idx]
        plt.barh(
            offsets,
            datapoints,
            left=starting_values,
            height=width,
            color=color,
            alpha=(1.0 - idx * 0.2),
            label=names[idx],
        )
        starting_values += datapoints
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
