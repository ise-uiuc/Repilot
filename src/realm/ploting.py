from typing import Callable, Iterable, TypeVar, cast

import torch
from matplotlib import pyplot as plt

from .results import GenerationDatapoint, ValidationDatapoint

Datapoint = TypeVar("Datapoint")


# def transform_points(
#     datapoints: Iterable[Datapoint],
#     get_xy: Callable[[Datapoint], tuple[int | float, int | float]],
# ) -> list[tuple[int | float, int | float]]:
#     return list(map(get_xy, datapoints))


def plot_bar(data_dict: dict[str, list[int]], names: list[str]) -> None:
    labels = list(data_dict.keys())
    xs = list(range(len(labels)))
    stacked_data = torch.tensor(list(data_dict.values()))
    assert stacked_data.shape[1] == len(names)
    starting_values = torch.zeros(len(labels))
    for idx in range(len(names)):
        datapoints = stacked_data[:, idx]
        plt.barh(xs, datapoints, left=starting_values)
        starting_values += datapoints
    # plt.tight_layout()
    assert len(xs) == len(labels)
    plt.yticks(xs, labels)
    plt.legend(names)


def plot_bars(data_dicts: list[dict[str, list[int]]], names: list[str]) -> None:
    ...


def plot_generation(labels_and_datapoints: list[tuple[str, GenerationDatapoint]]):
    plot_bar(
        {
            k: [
                v.n_unique - v.n_unfinished - v.n_pruned,
                v.n_unfinished,
                v.n_pruned,
                v.n_total - v.n_unique,
            ]
            for k, v in labels_and_datapoints
        },
        ["Unique", "Unfinished", "Pruned", "Duplicate"],
    )


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
