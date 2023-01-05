from functools import partial
import itertools
from os import PathLike
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple
from unidiff import PatchSet, PatchedFile
from unidiff.patch import Line
from realm import utils
import subprocess
import multiprocessing as mp
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TypeVar
from realm.utils import chunked
import git

Metadata = Dict[str, List[Dict[str, str]]]

T = TypeVar("T")


# print(_get_all_bugs()['Lang'][0])

# ROOT = Path('/home/yuxiang/fastd/Developer/d4j-checkout')


# def run_process(*cmd: str):
#     subprocess.run(cmd)

# TODO: add to main
# BATCH_SIZE = 64
# if __name__ == '__main__':
#     for d_sub in chunked(BATCH_SIZE, data):
#         processes: List[mp.Process] = []
#         for d in d_sub:
#             p = mp.Process(target=run_process, args=d['cmd'])
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()


class Change(NamedTuple):
    start: int
    removed_lines: List[str]
    added_lines: List[str]


class BuggyFile(NamedTuple):
    path: str
    changes: List[Change]

    @staticmethod
    def from_patch_file(reversed: bool, patch_file: PatchedFile) -> "BuggyFile":
        changes: List[Change] = []
        lines_iter: Iterator[Line] = (line for hunk in patch_file for line in hunk)

        try:
            last_context = next(lines_iter)
            while True:
                start_line = next(lines_iter)
                if start_line.is_added:
                    assert last_context.is_context
                    added_lines, line_iter = utils.take_while_two(
                        lambda _: True,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        itertools.chain([start_line], lines_iter),
                    )
                    removed_lines: List[Line] = []
                elif start_line.is_removed:
                    assert last_context.is_context
                    removed_lines, line_iter = utils.take_while_two(
                        lambda _: True,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        itertools.chain([start_line], lines_iter),
                    )
                    added_lines, line_iter = utils.take_while_two(
                        lambda elem: elem.is_added,
                        lambda lhs, rhs: utils.line_consecutive(lhs, rhs),
                        line_iter,
                    )
                else:
                    if start_line.is_context:
                        last_context = start_line
                    continue
                if reversed:
                    removed_lines, added_lines = added_lines, removed_lines
                if len(removed_lines) == 0 and len(added_lines) > 0:
                    start = 1 + (
                        last_context.target_line_no
                        if reversed
                        else last_context.source_line_no
                    )
                else:
                    start = (
                        removed_lines[0].target_line_no
                        if reversed
                        else removed_lines[0].source_line_no
                    )
                assert start is not None
                changes.append(
                    Change(
                        start,
                        # Eliminate the '-'/'+'
                        [str(line)[1:] for line in removed_lines],
                        [str(line)[1:] for line in added_lines],
                    )
                )
        except StopIteration:

            def remove_prefix(diff_fname: str) -> str:
                prefixes = ["a/", "b/"]
                prefix = next(filter(diff_fname.startswith, prefixes), "")
                return diff_fname[len(prefix) :]

            return BuggyFile(remove_prefix(patch_file.source_file), changes)


class Bug(NamedTuple):
    buggy_files: List[BuggyFile]
    proj_path: str


BugId = str
BenchmarkMetadata = Dict[BugId, Bug]


class Defects4J:
    def __init__(self, d4j_home: Path, d4j_checkout_root: Path) -> None:
        self.d4j_home = d4j_home
        self.d4j_checkout_root = d4j_checkout_root
        assert d4j_home.exists()
        assert self.d4j_executable.exists()
        assert d4j_checkout_root.exists()
        self.metadata = self._get_metadata()
        self.all_bugs = self._all_bugs()
        self.single_hunk_bugs = {
            id: bug
            for (id, bug) in self.all_bugs.items()
            if len(bug.buggy_files) == 1 and len(bug.buggy_files[0].changes) == 1
        }

    @staticmethod
    def split_bug_id(bug_id: str) -> tuple[str, str]:
        proj, id_str = bug_id.split("-")
        return proj, id_str

    @staticmethod
    def form_bug_id(proj: str, id_str: str) -> str:
        return proj + "-" + id_str

    @property
    def d4j_executable(self) -> Path:
        return self.d4j_home / "framework" / "bin" / "defects4j"

    def checkout_buggy(self, bug_id: str, bug_proj_path: str):
        proj, id_str = self.split_bug_id(bug_id)
        repo = git.Repo(bug_proj_path)
        repo.git.execute(["git", "checkout", "HEAD", "-f", "."])
        subprocess.run(
            [
                str(self.d4j_executable),
                "checkout",
                "-p",
                proj,
                f"-v{id_str}b",
                "-w",
                bug_proj_path,
            ]
        )
        repo.git.execute(["git", "checkout", "HEAD", "-f", "."])
        repo.git.execute(["git", "clean", "-xfd"])
        repo.close()

    def buggy_files(self, bug: dict) -> List[BuggyFile]:
        patch_file = (
            self.d4j_home
            / "framework"
            / "projects"
            / bug["proj"]
            / "patches"
            / f"{bug['bug_id']}.src.patch"
        )
        patch_set = PatchSet.from_filename(patch_file, errors="ignore")
        patch_files: Iterator[PatchedFile] = filter(
            lambda f: f.is_modified_file, patch_set
        )
        return list(map(partial(BuggyFile.from_patch_file, True), patch_files))

    @staticmethod
    def bug_id(bug: dict) -> str:
        return f"{bug['proj']}-{bug['bug_id']}"

    def _all_bugs(self) -> BenchmarkMetadata:
        return {
            self.bug_id(bug): Bug(
                buggy_files=self.buggy_files(bug), proj_path=bug["path"]
            )
            for bug in self.metadata
        }

    def _get_checkout_meta(self, proj: str, bug: Dict[str, str]) -> Dict:
        path = self.d4j_checkout_root / f'{proj}-{bug["bug.id"]}'
        bug_id = bug["bug.id"]
        return {
            "proj": proj,
            "bug_id": bug_id,
            "buggy_commit": bug["revision.id.buggy"],
            "url": bug["report.url"],
            "fixed_commit": bug["revision.id.fixed"],
            "path": str(path.absolute()),
            "cmd": [
                str(self.d4j_executable),
                "checkout",
                "-p",
                proj,
                "-v",
                f"{bug_id}f",
                "-w",
                str(path.absolute()),
            ],
        }

    def _get_all_checkout_meta(self, bugs: Metadata) -> List[Dict[str, str]]:
        return [
            self._get_checkout_meta(proj, bug)
            for proj, proj_bugs in bugs.items()
            for bug in proj_bugs
        ]

    def _get_metadata(self) -> List[Dict[str, str]]:
        all_bugs = self._get_all_bugs()
        data = self._get_all_checkout_meta(all_bugs)
        return data

    def _get_all_bugs(self) -> Metadata:
        def impl():
            proj_dir = self.d4j_home / "framework" / "projects"
            for path_i in proj_dir.iterdir():
                if not path_i.is_dir():
                    continue
                for path_j in path_i.iterdir():
                    if path_j.name == "active-bugs.csv":
                        with open(path_j) as f:
                            dataset = csv.reader(f)
                            keys = next(dataset)
                            kv_list = (zip(keys, values) for values in dataset)
                            bugs = [{k: v for k, v in kv} for kv in kv_list]
                            yield path_i.name, bugs

        return {proj: bugs for proj, bugs in impl()}
