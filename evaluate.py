import json
import sys
from pathlib import Path
from typing import TypedDict, List
from argparse import ArgumentParser


class DataPoint(TypedDict):
    n_total: int
    n_parse_success: int
    n_comp_success: int
    n_test_success: int
    gen_time: float
    total_time_consumed: float


def get_data_for_bug(bug_meta: dict, avg_gen_time: float) -> List[DataPoint]:
    test_succeeded = bug_meta['succeeded']
    comp_succeeded = test_succeeded + bug_meta['test_failed']
    parse_succeeded = comp_succeeded + bug_meta['comp_failed']
    all_unique_patches = parse_succeeded + (bug_meta['parse_failed'] if 'parse_failed' in bug_meta else [])
    times = {int(key): value for key, value in bug_meta['times'].items()}
    assert len(times) == len(all_unique_patches)
    result: List[DataPoint] = []
    datapoint: DataPoint = {
        'n_total': 0,
        'n_parse_success': 0,
        'n_comp_success': 0,
        'n_test_success': 0,
        'gen_time': 0,
        'total_time_consumed': 0,
    }
    for patch_id in sorted(all_unique_patches):
        # TODO: not use avg gen time
        total_time = times[patch_id] + avg_gen_time
        if patch_id in test_succeeded:
            increase = 1, 1, 1, 1
        elif patch_id in comp_succeeded:
            increase = 1, 1, 1, 0
        elif patch_id in parse_succeeded:
            increase = 1, 1, 0, 0
        else:
            assert patch_id in all_unique_patches
            increase = 1, 0, 0, 0
        datapoint = {
            'n_total': datapoint['n_total'] + increase[0],
            'n_parse_success': datapoint['n_parse_success'] + increase[1],
            'n_comp_success': datapoint['n_comp_success'] + increase[2],
            'n_test_success': datapoint['n_test_success'] + increase[3],
            'gen_time': datapoint['gen_time'] + avg_gen_time,
            'total_time_consumed': datapoint['total_time_consumed'] + total_time,
        }
        result.append(datapoint)
    return result


def get_data(folder: Path) -> dict:
    result: dict = {'detailed': {}, 'general': {}}
    for proj_dir in filter(Path.is_dir, folder.iterdir()):
        proj_name = proj_dir.name
        f_val_meta = proj_dir / (proj_name + '.json')
        result['detailed'][proj_name] = {}
        with open(f_val_meta) as f:
            val_meta = json.load(f)
        all_bugs = list(filter(Path.is_dir, proj_dir.iterdir()))
        all_bugs.sort(key=lambda bug: int(bug.name))
        accum_datapoints = [{
            'n_total': 0,
            'n_parse_success': 0,
            'n_comp_success': 0,
            'n_test_success': 0,
            'gen_time': 0,
            'total_time_consumed': 0.,
        }]
        for bug in all_bugs:
            with open(bug / 'time.json') as f:
                gen_time_meta = json.load(f)
            bug_id = proj_name + '-' + bug.name
            if bug_id not in val_meta:
                continue
            bug_meta = val_meta[bug_id]
            # TODO: do not use average (use dict instead of list to store generation times)
            avg_gen_time = sum(
                gen_time_meta['times']) / len(gen_time_meta['times'])
            datapoints = get_data_for_bug(bug_meta, avg_gen_time)
            result['detailed'][proj_name][bug.name] = datapoints
            last = accum_datapoints[-1]
            accum_datapoints.extend({
                'n_total': last['n_total'] + datapoint['n_total'],
                'n_parse_success': last['n_parse_success'] + datapoint['n_parse_success'],
                'n_comp_success': last['n_comp_success'] + datapoint['n_comp_success'],
                'n_test_success': last['n_test_success'] + datapoint['n_test_success'],
                'gen_time': last['gen_time'] + datapoint['gen_time'],
                'total_time_consumed': last['total_time_consumed'] + datapoint['total_time_consumed'],
            } for datapoint in datapoints)
        result['general'][proj_name] = accum_datapoints
    return result


# analysis: dict = {
#     'general': {
#         'n_fixed_bug (plausible)': 0,
#         'n_total_bug': 0,
#         'n_plausible': 0,
#         'n_parse_success': 0,
#         'n_comp_success': 0,
#         'val_time': 0,
#         'n_total': 0,
#     },
#     'details': {},
# }


# general = analysis['general']
# details = analysis['details']
# for bug_id, result in data.items():
#     # if bug_id == 'Chart-11':
#     #     continue
#     general['n_total_bug'] += 1
#     details[bug_id] = {}

#     succeeded = len(result['succeeded'])
#     parse_failed = len(result['parse_failed']) if 'parse_failed' in result else 0
#     comp_failed = len(result['comp_failed'])
#     test_failed = len(result['test_failed'])
#     details[bug_id]['n_plausible'] = succeeded
#     general['n_plausible'] += succeeded
#     general['n_fixed_bug (plausible)'] += 1 if succeeded > 0 else 0

#     if 'times' in result:
#         details[bug_id]['val_time'] = sum(result['times'].values())
#         general['val_time'] += details[bug_id]['val_time']

#     details[bug_id]['n_parse_success'] = comp_failed + test_failed + succeeded
#     details[bug_id]['n_comp_success'] = test_failed + succeeded

#     general['n_parse_success'] += comp_failed + test_failed + succeeded
#     general['n_comp_success'] += test_failed + succeeded

#     details[bug_id]['n_total'] = succeeded + parse_failed + comp_failed + test_failed
#     general['n_total'] += succeeded + parse_failed + comp_failed + test_failed
# general['avg_val_time'] = general['val_time'] / general['n_total']

# print(json.dumps(analysis, indent=2))

from matplotlib import pyplot as plt
import sys
from pathlib import Path
import json

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE, family='sans-serif')  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE - 1)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

MIN_FAC_TWO = None
MIN_FAC = None

plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{xfrac}")

def make_label(text: str) -> str:
    return f'\\textbf{{{text}}}'

if __name__ == '__main__':
    parser = ArgumentParser('Evaluate the experimental results')
    parser.add_argument(
        "-f", "--folders", type=str, nargs="+", help="bug report folder"
    )
    # parser.add_argument("--tags", type=str, nargs="+", help="tags")
    parser.add_argument(
        "-o", "--output", type=str, default="results", help="results folder"
    )
    parser.add_argument("--pdf", action="store_true", help="use pdf as well")
    args = parser.parse_args()

    result_folder = Path(args.output)
    result_folder.mkdir(exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for folder in map(Path, args.folders):
        data = get_data(folder)

        general = data['general']

        proj, datapoints = next(iter(general.items()))
        # datapoints = data['detailed']['Chart']['10']
        n_total = [v['n_total'] for v in datapoints]
        times = [v['total_time_consumed'] for v in datapoints]
        gen_times = [v['gen_time'] for v in datapoints]
        n_comp_success = [v['n_comp_success'] for v in datapoints]

        y_label = make_label('Number of compilable patches')

        axs[0].set_xlabel(make_label('Number of unique patches'))
        axs[0].set_ylabel(y_label)
        axs[0].plot(n_total, n_comp_success, label=folder.name)
        axs[0].legend()

        axs[1].set_xlabel(make_label('Total time consumed'))
        axs[1].set_ylabel(y_label)
        axs[1].plot(times, n_comp_success, label=folder.name)
        axs[1].legend()

        axs[2].set_xlabel(make_label('Generation time'))
        axs[2].set_ylabel(y_label)
        axs[2].plot(gen_times, n_comp_success, label=folder.name)
        axs[2].legend()

    fig.savefig(result_folder / 'plot.png')
