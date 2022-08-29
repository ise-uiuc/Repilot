import os
import json
from typing import Dict, Type
from bears import Bears
from bugsinpy import BugsInPy
from bugs_dot_jar import BugsDotJar
import bugsinpy
import bugs_dot_jar
import bears
from dataset import FilterBenchmark, RepairBenchmark, ReproduceBenchmark, SingleFuncRepairBenchmark
import argparse
from utils import filter_single_line

ALL_DATASETS: Dict[str, Type[RepairBenchmark]] = {
    x.__name__: x for x in [BugsInPy, BugsDotJar, Bears]}  # type: ignore # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process a program repair benchmark')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True, choices=list(ALL_DATASETS.keys()), help='The path that stores the repair results')
    parser.add_argument(
        '-t', '--log-folder-tag', type=str, default='', help='The tag of the logging folder')

    subparsers = parser.add_subparsers(dest='command')
    parser_checkout = subparsers.add_parser('checkout')

    parser_filter = subparsers.add_parser('filter')

    parser_validate = subparsers.add_parser('validate')
    parser_validate.add_argument(
        '-o', '--option', type=str, required=True, choices=['hunk', 'func', 'line'], help='Select how to filter the dataset')
    parser_validate.add_argument(
        '-p', '--path', type=str, required=False, help='The path that stores the repair results')
    parser_validate.add_argument(
        '--from-metadata', action='store_true', help='Load dataset from metadata')
    parser_validate.add_argument(
        '--reproduce', action='store_true', help='Just reproduce the bugs')
    args = parser.parse_args()

    command: str = args.command

    dataset_cls = ALL_DATASETS[args.dataset]
    checkout_meta = f'{args.dataset}-checkout.json'
    filter_func_meta = f'{args.dataset}-filter-func.json'
    filter_hunk_meta = f'{args.dataset}-filter-hunk.json'
    single_func_meta = os.path.join(
        args.dataset, 'single_function_repair.json')
    single_hunk_meta = os.path.join(
        args.dataset, 'single_function_single_hunk_repair.json')
    if command == 'checkout':
        if args.dataset == BugsInPy.__name__:
            bugsinpy \
                .checkout('../BugsInPy', '../bugsinpy-checkout') \
                .dump(checkout_meta)
        elif args.dataset == BugsDotJar.__name__:
            bugs_dot_jar.checkout('../bugs-dot-jar').dump(checkout_meta)
        elif args.dataset == Bears.__name__:
            bears.checkout('../bears-benchmark',
                           '../bears-bugs').dump(checkout_meta)
    elif command == 'filter':
        dataset_cls \
            .load(checkout_meta) \
            .with_logger_tag(args.log_folder_tag) \
            .filter_single_func() \
            .reproduce(120, False) \
            .dump(filter_func_meta) \
            .dump_repair_metadata(single_func_meta) \
            .filter_single_hunk() \
            .dump(filter_hunk_meta) \
            .dump_repair_metadata(single_hunk_meta)
    elif command == 'validate':
        if args.from_metadata:
            if args.option == 'func':
                dataset: SingleFuncRepairBenchmark = ReproduceBenchmark.load(
                    filter_func_meta)
            elif args.option == 'hunk':
                dataset = FilterBenchmark.load(filter_hunk_meta)
            elif args.option == 'line':
                raise NotImplementedError
        else:
            repair_meta = single_func_meta if args.option == 'func' else single_hunk_meta
            with open(repair_meta) as f:
                map_dict = json.load(f)
            if args.option == 'line':
                map_dict = filter_single_line(map_dict)
            dataset = FilterBenchmark(
                dataset_cls.load(checkout_meta), map_dict)
        dataset = dataset.with_logger_tag(args.log_folder_tag)
        if not args.reproduce:
            with open(args.path) as f:
                patch_dict = json.load(f)
            dataset.validate(patch_dict, args.option, 120, False)
        else:
            dataset.reproduce(120, False)

