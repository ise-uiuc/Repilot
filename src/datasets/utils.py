import ast
import datetime
import json
import os
import pickle
from time import sleep
import time
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, TypeVar
from urllib.error import HTTPError
from urllib.request import urlopen
import javalang
from intervaltree import IntervalTree
from unidiff import Hunk, PatchSet, PatchedFile
from unidiff.patch import Line
import git


def is_single_line(data: dict) -> bool:
    def line_len(s: str) -> int:
        return len(s.splitlines())
    return line_len(data['fix']) == line_len(data['prefix']) + line_len(data['suffix']) + 1


def filter_single_line(dataset: dict) -> dict:
    dataset = {bug_id: data for bug_id, data in filter(
        lambda item: is_single_line(item[1]), dataset.items())}
    return dataset


def all_subclasses(cls: type) -> Set[type]:
    return set(cls.__subclasses__()).union([
        s for c in cls.__subclasses__() for s in all_subclasses(c)
    ])


def repeat_fetch(url: str, interval: float = 1.0, max_trials: int = 500) -> Any:
    trials = 0
    while True:
        try:
            return urlopen(url)
        except HTTPError as e:
            # too many requests
            if e.code == 429:
                if trials >= max_trials:
                    raise e
                trials += 1
                sleep(interval)
            else:
                raise e


def parse_python_func_intervals(content: str) -> Set[Tuple[int, int]]:

    def compute_interval(node: ast.AST) -> Tuple[int, int]:
        min_lineno = node.lineno
        max_lineno = node.lineno
        for node in ast.walk(node):
            if hasattr(node, "lineno"):
                min_lineno = min(min_lineno, node.lineno)
                max_lineno = max(max_lineno, node.lineno)
        return (min_lineno, max_lineno + 1)

    func_intervals = set()
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, ast.FunctionDef):
            start, end = compute_interval(node)
            func_intervals.add((start, end))
    return func_intervals


def get_minimum_enclosed_region(
    regions: Iterable[Tuple[int, int]],
    key: int,
) -> Optional[Tuple[int, int]]:
    tree = IntervalTree.from_tuples(regions)
    matches = tree[key]
    if len(matches) == 0:
        return None
    result = min(matches, key=lambda i: i.length())
    return (result.begin, result.end)


def get_minimum_enclosed_region_for_range(
    regions: Iterable[Tuple[int, int]],
    start: int,
    end_exclusive: int,
) -> Tuple[int, int]:
    assert start <= end_exclusive
    start_match = get_minimum_enclosed_region(regions, start)
    end_match = get_minimum_enclosed_region(regions, end_exclusive - 1)
    if start_match is None or end_match is None:
        raise ValueError(start_match, end_match)
    if start_match != end_match:
        raise ValueError(start_match, end_match)
    return start_match


def get_minimum_enclosed_region_for_range2(
    regions: Iterable[Tuple[int, int]],
    start: int,
    end_exclusive: int,
) -> Optional[Tuple[int, int]]:
    assert start <= end_exclusive
    start_match = get_minimum_enclosed_region(regions, start)
    end_match = get_minimum_enclosed_region(regions, end_exclusive - 1)
    if start_match is None or end_match is None:
        return None
    if start_match != end_match:
        return None
    return start_match


def parse_java_func_intervals(content: str) -> Set[Tuple[int, int]]:
    func_intervals = set()
    for _, node in javalang.parse.parse(content):
        if isinstance(node, (javalang.tree.MethodDeclaration,
                             javalang.tree.ConstructorDeclaration)):
            func_intervals.add((
                node.start_position.line,
                node.end_position.line + 1,
            ))

    return func_intervals


def get_interval_parser(fname: str) -> Callable:
    if fname.endswith('.java'):
        return parse_java_func_intervals
    elif fname.endswith('.py'):
        return parse_python_func_intervals
    else:
        raise NotImplementedError


def get_single_file_patch(patch_set: PatchSet) -> PatchedFile:
    if len(patch_set) == 1 \
            and patch_set[0].is_modified_file:
        return patch_set[0]
    else:
        raise ValueError(patch_set)


def get_single_hunk(patch_file: PatchedFile) -> Hunk:
    if len(patch_file) == 1:
        return patch_file[0]
    else:
        raise ValueError(patch_file)


def get_single_consecutive_hunk_region(hunk: Hunk) -> Tuple[int, int]:
    regions = get_buggy_regions_for_hunk(hunk)
    if len(regions) == 1:
        return list(regions)[0]
    else:
        raise ValueError(hunk)


def get_single_func_interval(parse_func_intervals: Callable[[str], Iterable[Tuple[int, int]]], src: str, hunk_intervals: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    all_func_intervals = parse_func_intervals(src)
    func_intervals_for_hunks = set([get_minimum_enclosed_region_for_range(
        all_func_intervals, hunk_start, hunk_end) for (hunk_start, hunk_end) in hunk_intervals])
    if len(func_intervals_for_hunks) == 1:
        return list(func_intervals_for_hunks)[0]
    else:
        raise ValueError(src, hunk_intervals)


def get_buggy_regions_for_hunk(hunk: Hunk) -> Set[Tuple[int, int]]:
    lines: List[Line] = [
        line for line in hunk if line.is_added or line.is_removed]
    consecutive_modifications: List[List[Line]] = []
    current_block: List[Line] = [lines[0]]
    for idx in range(1, len(lines)):
        if lines[idx].diff_line_no == current_block[-1].diff_line_no + 1:
            current_block.append(lines[idx])
        else:
            consecutive_modifications.append(current_block)
            current_block = [lines[idx]]
    consecutive_modifications.append(current_block)

    regions: Set[Tuple[int, int]] = set()
    for block in consecutive_modifications:
        deletions = [line for line in block if line.is_removed]
        additions = [line for line in block if line.is_added]
        if len(deletions) > 0 and len(additions) > 0:
            assert deletions[-1].diff_line_no + 1 == additions[0].diff_line_no
        if len(deletions) == 0:
            # prefix end (exclusive), suffix start (inclusive)
            regions.add((additions[0].target_line_no,
                        additions[0].target_line_no))
        else:
            regions.add((deletions[0].source_line_no,
                        deletions[-1].source_line_no + 1))
    return regions


def retrieve_diff(url: str) -> PatchSet:
    diff = repeat_fetch(url)
    encoding = diff.headers.get_charsets()[0]
    return PatchSet(diff, encoding)
    # except (UnidiffParseError, UnicodeDecodeError):
    #     return None


def retrieve_file(url: str) -> bytes:
    content = repeat_fetch(url)
    content = content.read()
    return content


def make_github_diff_url(repo: str, failed_sha: str, passed_sha: str) -> str:
    # FIXME: GitHub doesn't provide a way to retrieve two-dot `.diff` file. Consider pull it locally
    # e.g., https://github.com/scikit-learn/scikit-learn/compare/03fac80b67078b236998275c9e2fd1f940968eb0..54d17a04e9170a56037472267aa8e788868fafc0.diff
    return 'https://github.com/{repo}/compare/{failed}..{passed}.patch'.format(
        repo=repo,
        failed=failed_sha,
        passed=passed_sha,
    )


def make_github_file_url(repo: str, sha: str, path: str) -> str:
    return 'https://github.com/{repo}/raw/{sha}/{file}'.format(
        repo=repo,
        sha=sha,
        file=path,
    )


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.needs_update = False
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.cached_data: dict = pickle.load(f)
        else:
            self.cached_data = {}

    _T = TypeVar('_T')

    def get(self, key: str, make_datum: Callable[[Any], _T]) -> _T:
        if key in self.cached_data:
            return self.cached_data[key]
        self.needs_update = True
        ret = make_datum(key)
        self.cached_data[key] = ret
        return ret

    def save(self):
        if not self.needs_update:
            return
        self.needs_update = False
        with open(self.path, 'wb') as f:
            pickle.dump(self.cached_data, f)


class Logger:
    def __init__(self, prefix="", use_existing_dir=False):
        self.log_folder = f'repair-log-{time.strftime("%y_%m%d_%H%M%S")}'
        if prefix != "":
            self.log_folder = prefix + '-' + self.log_folder

        if use_existing_dir:
            assert os.path.exists(self.log_folder)
        else:
            os.mkdir(self.log_folder)
            print(f'Create log folder: {self.log_folder}')

        print(f'Using `{self.log_folder}` as the logging folder')
        with open(os.path.join(self.log_folder, 'meta.txt'), 'w') as f:
            fuzz_repo = git.Repo(search_parent_directories=True)

            def _log_repo(f, tag, repo: git.Repo):
                f.write(f'{tag} GIT HASH: {repo.head.object.hexsha}\n')
                f.write(f'{tag} GIT STATUS: ')
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')
                f.write(repo.git.status())
                f.write(
                    '\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n')

            f.write(f'START TIME: {datetime.datetime.now()}')
            _log_repo(f, 'Repair', fuzz_repo)

    def log_str(self, content: str, file: str, auto_newline: bool = True):
        with open(os.path.join(self.log_folder, file), 'a') as f:
            f.write(content + '\n')
            if auto_newline:
                f.write('\n')

    def log_json(self, content: dict, file: str):
        with open(os.path.join(self.log_folder, file), 'w') as f:
            json.dump(content, f, indent=2)
