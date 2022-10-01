from functools import partial
import itertools
from os import PathLike
from pathlib import Path
import subprocess
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple
from unidiff import PatchSet, PatchedFile, Hunk
from unidiff.patch import Line
from realm import utils


class Change(NamedTuple):
    start: int
    removed_lines: List[str]
    added_lines: List[str]


class BuggyFile(NamedTuple):
    path: str
    changes: List[Change]

    @staticmethod
    def from_patch_file(reversed: bool, patch_file: PatchedFile) -> 'BuggyFile':
        changes: List[Change] = []
        lines_iter: Iterator[Line] = (
            line for hunk in patch_file for line in hunk)

        try:
            last_context = next(lines_iter)
            while True:
                start_line = next(lines_iter)
                if start_line.is_added:
                    assert last_context.is_context
                    added_lines, line_iter = utils.take_while_two(
                        lambda _: True,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        itertools.chain([start_line], lines_iter))
                    removed_lines: List[Line] = []
                elif start_line.is_removed:
                    assert last_context.is_context
                    removed_lines, line_iter = utils.take_while_two(
                        lambda _: True,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        itertools.chain([start_line], lines_iter))
                    added_lines, line_iter = utils.take_while_two(
                        lambda elem: elem.is_added,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        line_iter)
                else:
                    if start_line.is_context:
                        last_context = start_line
                    continue
                if reversed:
                    removed_lines, added_lines = added_lines, removed_lines
                if len(removed_lines) == 0 and len(added_lines) > 0:
                    start = 1 + \
                        (last_context.target_line_no if reversed else last_context.source_line_no)
                else:
                    start = removed_lines[0].target_line_no if reversed else removed_lines[0].source_line_no
                assert start is not None
                changes.append(Change(
                    start,
                    # Eliminate the '-'/'+'
                    [str(line)[1:] for line in removed_lines],
                    [str(line)[1:] for line in added_lines],
                ))
        except StopIteration:
            def remove_prefix(diff_fname: str) -> str:
                prefixes = ['a/', 'b/']
                prefix = next(filter(diff_fname.startswith, prefixes), '')
                return diff_fname[len(prefix):]

            return BuggyFile(remove_prefix(patch_file.source_file), changes)


class Bug(NamedTuple):
    buggy_files: List[BuggyFile]
    proj_path: str


BugId = str
BenchmarkMetadata = Dict[BugId, Bug]


class Defects4J:
    def __init__(self, d4j_home: PathLike, metadata: List[dict]) -> None:
        self.d4j_home = Path(d4j_home)
        self.metadata = metadata

    def buggy_files(self, bug: dict) -> List[BuggyFile]:
        patch_file = self.d4j_home / 'framework' / 'projects' / \
            bug['proj'] / 'patches' / f"{bug['bug_id']}.src.patch"
        patch_set = PatchSet.from_filename(patch_file, errors='ignore')
        patch_files: Iterator[PatchedFile] = filter(
            lambda f: f.is_modified_file, patch_set)
        return list(map(partial(BuggyFile.from_patch_file, True), patch_files))

    @staticmethod
    def bug_id(bug: dict) -> str:
        return f"{bug['proj']}-{bug['bug_id']}"

    def all_bugs(self) -> BenchmarkMetadata:
        return {self.bug_id(bug): Bug(
            buggy_files=self.buggy_files(bug),
            proj_path=bug['path']
        ) for bug in self.metadata}
