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
    def from_patch_file(patch_file: PatchedFile) -> 'BuggyFile':
        changes: List[Change] = []
        lines_iter: Iterator[Line] = (
            line for hunk in patch_file for line in hunk)
        for start_line in lines_iter:
            if start_line.is_context:
                continue
            if start_line.is_added:
                added_lines, iter_remaining = utils.take_while(
                    lambda line: line.is_added, lines_iter)
                removed_lines: List[Line] = []
                changes.append(
                    Change(start_line.source_line_no, [], added_lines))
            elif start_line.is_removed:
                removed_lines, iter_remaining = utils.take_while(
                    lambda line: line.is_removed, lines_iter)
                added_lines, iter_remaining = utils.take_while(
                    lambda line: line.is_added, iter_remaining)
                # print(added_lines)
            changes.append(Change(
                start_line.source_line_no,
                removed_lines,
                added_lines
            ))
            lines_iter = iter_remaining

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
    def __init__(self, d4j_home: PathLike, metadata: List[dict], buggy_location_path: PathLike) -> None:
        self.d4j_home = Path(d4j_home)
        self.metadata = metadata
        self.buggy_location_path = Path(buggy_location_path)
        assert self.buggy_location_path.exists()

    def buggy_files(self, bug: dict) -> List[BuggyFile]:
        patch_file = self.d4j_home / 'framework' / 'projects' / \
            bug['proj'] / 'patches' / f"{bug['bug_id']}.src.patch"
        patch_set = PatchSet.from_filename(patch_file, errors='ignore')
        patch_files: Iterator[PatchedFile] = filter(
            lambda f: f.is_modified_file, patch_set)
        return list(map(BuggyFile.from_patch_file, patch_files))

    @staticmethod
    def bug_id(bug: dict) -> str:
        return f"{bug['proj']}-{bug['bug_id']}"

    def all_bugs(self) -> BenchmarkMetadata:
        return {self.bug_id(bug): Bug(
            buggy_files=self.buggy_files(bug),
            proj_path=bug['path']
        ) for bug in self.metadata}
