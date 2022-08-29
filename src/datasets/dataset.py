import json
import os
from typing import Callable, Generic, Iterable, List, Optional, Protocol, Tuple, Type, TypeVar

from tqdm import tqdm
from unidiff import Hunk, PatchSet

import utils
from parse_quixbugs import get_unified_diff


class ReproduceError(Exception):
    pass


class RepoBasedBug(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def extension(self) -> str: ...

    @property
    def working_dir(self) -> str: ...

    def get_patch_set(self) -> PatchSet: ...

    def checkout_buggy(self): ...

    def checkout_fixed(self): ...

    def validate(self, timeout: float, clean_test: bool): ...

    def reproduce(self, timeout: float, clean_test: bool):
        self.checkout_fixed()
        self.validate(timeout, clean_test)

    def apply_patch(self, path: str, patch: str):
        with open(os.path.join(self.working_dir, path), 'w') as f:
            f.write(patch)

    def get_buggy_src(self, path: str) -> str:
        self.checkout_buggy()
        with open(os.path.join(self.working_dir, path)) as f:
            return f.read()

    def get_fixed_src(self, path: str) -> str:
        self.checkout_fixed()
        with open(os.path.join(self.working_dir, path)) as f:
            return f.read()

    def process_single_func(self) -> dict:
        patch_set = self.get_patch_set()
        patch_file = utils.get_single_file_patch(patch_set)
        hunks: List[Hunk] = list(patch_file)
        hunk_intervals = [
            interval for hunk in hunks for interval in utils.get_buggy_regions_for_hunk(hunk)]

        buggy = self.get_buggy_src(patch_file.path)
        fix = self.get_fixed_src(patch_file.path)
        func_start, func_end = utils.get_single_func_interval(
            utils.get_interval_parser(patch_file.path), buggy, hunk_intervals)

        hunk_start, _ = min(hunk_intervals, key=lambda tuple: tuple[0])
        fix_all_func_intervals = utils.get_interval_parser(
            patch_file.path)(fix)
        fix_func_start, fix_func_end = utils.get_minimum_enclosed_region_for_range(
            fix_all_func_intervals, hunk_start, hunk_start + 1)

        data = {}
        data['path'] = patch_file.path
        data['buggy'] = '\n'.join(
            buggy.splitlines()[func_start - 1: func_end - 1])
        data['fix'] = '\n'.join(
            fix.splitlines()[fix_func_start - 1: fix_func_end - 1])
        data['start'] = func_start
        data['end'] = func_end

        return data

    def process_single_consecutive_hunk(self) -> dict:
        patch_set = self.get_patch_set()
        patch_file = utils.get_single_file_patch(patch_set)
        hunk = utils.get_single_hunk(patch_file)
        hunk_start, hunk_end = utils.get_single_consecutive_hunk_region(hunk)

        buggy = self.get_buggy_src(patch_file.path)
        fix = self.get_fixed_src(patch_file.path)
        all_func_intervals = utils.get_interval_parser(patch_file.path)(buggy)
        func_start, func_end = utils.get_minimum_enclosed_region_for_range(
            all_func_intervals, hunk_start, hunk_end)

        fix_all_func_intervals = utils.get_interval_parser(
            patch_file.path)(fix)
        fix_func_start, fix_func_end = utils.get_minimum_enclosed_region_for_range(
            fix_all_func_intervals, hunk_start, hunk_start + 1)

        data = {}
        data['path'] = patch_file.path
        data['prefix'] = '\n'.join(line for (idx, line) in enumerate(
            buggy.splitlines()) if idx + 1 < hunk_start and idx + 1 >= func_start)
        data['suffix'] = '\n'.join(
            line for (idx, line) in enumerate(buggy.splitlines()) if idx + 1 < func_end and idx + 1 >= hunk_end)
        data['buggy'] = '\n'.join(
            buggy.splitlines()[func_start - 1: func_end - 1])
        data['fix'] = '\n'.join(
            fix.splitlines()[fix_func_start - 1: fix_func_end - 1])
        data['start'] = func_start
        data['end'] = func_end

        return data


T = TypeVar('T', bound='RepoBasedBug')
U = TypeVar('U', bound='RepairBenchmark')


class RepairBenchmark(Protocol[T]):
    @property
    def metadata(self) -> dict: ...

    @classmethod
    def load_metadata(cls: Type[U], metadata: dict) -> U: ...

    @classmethod
    def load(cls: Type[U], path: str) -> U:
        with open(path) as f:
            metadata = json.load(f)
        return cls.load_metadata(metadata)
    
    def with_logger_tag(self: U, tag: str) -> U: ...

    @property
    def logger(self) -> Optional[utils.Logger]: ...

    def get_all_bugs(self) -> Iterable[T]: ...

    def dump(self: U, path: str) -> U:
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)  # type: ignore
        return self

    def filter_single_func(self) -> 'FilterBenchmark[T]':
        return self._filter('FUNC', RepoBasedBug.process_single_func)

    def filter_single_hunk(self) -> 'FilterBenchmark[T]':
        return self._filter(
            'HUNK', RepoBasedBug.process_single_consecutive_hunk)

    def _filter(self, option: str, process: Callable[[T], dict]) -> 'FilterBenchmark[T]':
        success_no_validation = 0
        filtered_data: dict = {}
        pbar = tqdm(list(self.get_all_bugs()))
        for idx, bug in enumerate(pbar):
            try:
                data_processed = process(bug)
                success_no_validation += 1
                # if self.validate_successful(bug):
                filtered_data[bug.id] = data_processed
            except ValueError:
                pass
            pbar.set_description(
                f'({option}) ({bug.id}) '
                f'Total: {idx + 1}, '
                f'Succeeded: {len(filtered_data)}, '
            )
        # print(filtered_data)
        # with open(path, 'w') as f:
        #     json.dump(filtered_data, f, indent=2)
        return FilterBenchmark(self, filtered_data)


R = TypeVar('R', bound='SingleFuncRepairBenchmark')


class SingleFuncRepairBenchmark(RepairBenchmark[T]):
    @property
    def repair_metadata(self) -> dict: ...

    def get_start_line(self, bug: T) -> int: ...

    def get_end_line(self, bug: T) -> int: ...

    def get_path(self, bug: T) -> str: ...

    def get_indent(self, bug: T) -> Tuple[int, str]: ...

    def dump_repair_metadata(self: R, path: str) -> R:
        with open(path, 'w') as f:
            json.dump(self.repair_metadata, f, indent=2)
        return self

    def validate(self, patch_dict: dict, tag: str, timeout: float, clean_test: bool):
        all_bugs = list(self.get_all_bugs())
        assert len(all_bugs) == len(patch_dict), (len(all_bugs), len(patch_dict))
        n_success_total = 0
        pbar = tqdm(all_bugs)
        for idx_total, bug in enumerate(pbar):
            bug: T  # type: ignore
            patches = patch_dict[bug.id + bug.extension]
            count, indent_char = self.get_indent(bug)
            n_success = 0
            for idx, patch_info in enumerate(patches):
                patch: str = patch_info['output']
                buggy_src = bug.get_buggy_src(self.get_path(bug))
                fixed_src = bug.get_fixed_src(self.get_path(bug))
                buggy_lines = buggy_src.splitlines()
                file_patch = '\n'.join(
                    buggy_lines[:self.get_start_line(bug) - 1]
                    + [count * indent_char + line for line in patch.splitlines()]
                    + buggy_lines[self.get_end_line(bug) - 1:]
                )
                bug.checkout_fixed()
                bug.apply_patch(self.get_path(bug), file_patch)
                # with open(os.path.join(self.workspace, bug_id, data['path']), 'r') as f:
                #     print(f.read())
                #     print()
                try:
                    bug.validate(timeout, clean_test)
                    n_success += 1
                    patch_info['valid'] = True
                except (ReproduceError, TimeoutError) as e:
                    
                    if self.logger is not None:
                        for arg in e.args:
                            self.logger.log_str(str(arg), f'{type(e).__name__}-validation-error.log')
                if self.logger is not None:
                    buggy_vs_patch = get_unified_diff(buggy_src, file_patch)
                    patch_vs_fixed = get_unified_diff(file_patch, fixed_src)
                    self.logger.log_str(
                        buggy_vs_patch, f'{bug.id}-{idx}-buggy_vs_patch')
                    self.logger.log_str(
                        patch_vs_fixed, f'{bug.id}-{idx}-patch_vs_fixed')

                pbar.set_description(
                    f'({tag.upper()}) ({bug.id}) '
                    f'Total Success Rate: {n_success_total}/{idx_total + 1}; '
                    f'Local Success Rate: {n_success}/{idx + 1}'
                )
            if n_success > 0:
                n_success_total += 1

        # Save and analyze results
        if self.logger is not None:
            self.logger.log_json(patch_dict, 'repair_result.json')
        analysis: dict = {
            'n_bugs': len(patch_dict),
            'n_bug_fixed': 0,
            'n_first_try_bug_fixed': 0,
            'n_patches': 0,
            'n_patch_correct': 0,
            'bugs': {},
        }
        for bug in all_bugs:
            bug_id = bug.id
            patches = patch_dict[bug.id + bug.extension]
            analysis['n_patches'] += len(patches)
            analysis['bugs'][bug_id] = {
                'n_patches': len(patches),
                'n_bug_fixed': 0,
            }
            current = analysis['bugs'][bug_id]
            for idx, patch_info in enumerate(patches):
                if patch_info['valid']:
                    analysis['n_patch_correct'] += 1
                    current['n_bug_fixed'] += 1
                    if idx == 0:
                        analysis['n_first_try_bug_fixed'] += 1
            if current['n_bug_fixed'] > 0:
                analysis['n_bug_fixed'] += 1
        if self.logger is not None:
            self.logger.log_json(analysis, 'repair_analysis.json')

    def reproduce(self, timeout: float, clean_test: bool) -> 'ReproduceBenchmark[T]':
        pbar = tqdm(list(self.get_all_bugs()))
        reproduced: List[str] = []
        timeout_count = 0
        for idx, bug in enumerate(pbar):
            bug: T  # type: ignore
            try:
                bug.reproduce(timeout, clean_test)
                reproduced.append(bug.id)
            except ReproduceError as e:
                if self.logger is not None:
                    self.logger.log_str(str(e), bug.id + '.reproduce_error')
            except TimeoutError as e:
                if self.logger is not None:
                    self.logger.log_str(str(e), bug.id + '.timeout_error')
                timeout_count += 1
            pbar.set_description(
                f'({bug.id}) '
                f'Total: {idx + 1}, '
                f'Reproduced: {len(reproduced)}, '
                f'Timeout: {timeout_count}'
            )
        return ReproduceBenchmark(self, reproduced)


class FilterBenchmark(Generic[T], SingleFuncRepairBenchmark[T]):
    def __init__(self, benchmark: RepairBenchmark[T], map_dict: dict) -> None:
        self.benchmark = benchmark
        self.map_dict = map_dict

    @property
    def repair_metadata(self) -> dict:
        return self.map_dict

    @classmethod
    def load_metadata(cls, metadata: dict) -> 'FilterBenchmark[T]':
        benchmark_cls: Type[RepairBenchmark[T]] = [cls for cls in utils.all_subclasses(RepairBenchmark) if cls.__name__ == metadata['benchmark_cls']][0]  # type: ignore # noqa
        return FilterBenchmark(
            benchmark_cls.load_metadata(metadata['benchmark_meta']),
            metadata['map_dict']
        )

    def get_start_line(self, bug: T) -> int:
        return self.map_dict[bug.id]['start']

    def get_end_line(self, bug: T) -> int:
        return self.map_dict[bug.id]['end']

    def get_path(self, bug: T) -> str:
        return self.map_dict[bug.id]['path']

    def get_indent(self, bug: T) -> Tuple[int, str]:
        buggy: str = self.map_dict[bug.id]['buggy']
        first_line = buggy.splitlines()[0]
        count = len(first_line) - len(first_line.lstrip()) 
        return count, "" if count == 0 else first_line[0]

    @property
    def metadata(self) -> dict:
        return {
            'benchmark_cls': self.benchmark.__class__.__name__,
            'benchmark_meta': self.benchmark.metadata,
            'map_dict': self.map_dict,
        }

    @property
    def logger(self) -> Optional[utils.Logger]:
        return self.benchmark.logger
    
    def with_logger_tag(self, tag: str) -> 'FilterBenchmark[T]':
        self.benchmark = self.benchmark.with_logger_tag(tag)
        return self

    def get_all_bugs(self) -> Iterable[T]:
        return (bug for bug in self.benchmark.get_all_bugs() if bug.id in self.map_dict)


class ReproduceBenchmark(Generic[T], SingleFuncRepairBenchmark[T]):
    def __init__(self, benchmark: SingleFuncRepairBenchmark[T], bug_ids: List[str]) -> None:
        self.benchmark = benchmark
        self.bug_ids = bug_ids

    @classmethod
    def load_metadata(cls, metadata: dict) -> 'ReproduceBenchmark[T]':
        benchmark_cls: Type[SingleFuncRepairBenchmark[T]] = [
            cls for cls in utils.all_subclasses(SingleFuncRepairBenchmark)
            if cls.__name__ == metadata['benchmark_cls']
        ][0]  # type: ignore # noqa

        return ReproduceBenchmark(
            benchmark_cls.load_metadata(metadata['benchmark_meta']),
            metadata['bug_ids'],
        )

    def get_start_line(self, bug: T) -> int:
        return self.benchmark.get_start_line(bug)

    def get_end_line(self, bug: T) -> int:
        return self.benchmark.get_end_line(bug)

    def get_path(self, bug: T) -> str:
        return self.benchmark.get_path(bug)

    def get_indent(self, bug: T) -> Tuple[int, str]:
        return self.benchmark.get_indent(bug)

    @property
    def metadata(self) -> dict:
        return {
            'benchmark_cls': self.benchmark.__class__.__name__,
            'benchmark_meta': self.benchmark.metadata,
            'bug_ids': self.bug_ids,
        }

    @property
    def repair_metadata(self) -> dict:
        return {id: self.benchmark.repair_metadata[id] for id in self.benchmark.repair_metadata if id in self.bug_ids}

    @property
    def logger(self) -> Optional[utils.Logger]:
        return self.benchmark.logger

    def with_logger_tag(self, tag: str) -> 'ReproduceBenchmark[T]':
        self.benchmark = self.benchmark.with_logger_tag(tag)
        return self

    def get_all_bugs(self) -> Iterable[T]:
        return (bug for bug in self.benchmark.get_all_bugs() if bug.id in self.bug_ids)

