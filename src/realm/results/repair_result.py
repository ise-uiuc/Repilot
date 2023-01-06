import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import git

from realm.config import MetaConfig, RepairConfig
from realm.generation_defs import SynthesisResultBatch, AvgSynthesisResult
from realm.utils import JsonSerializable, IORetrospective
from realm.lsp import TextFile

META_CONFIG_FNAME = "meta_config.json"
REPAIR_CONFIG_FNAME = "repair_config.json"
REPAIR_RESULT_PATH_PREFIX = "Repair-"


@dataclass(frozen=True)
class BuggyHunk(JsonSerializable):
    file: TextFile
    start: int
    end: int

    def to_json(self) -> Any:
        return {
            "file": self.file.to_json(),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_json(cls, d: dict) -> "BuggyHunk":
        return BuggyHunk(
            TextFile.from_json(d["file"]),
            int(d["start"]),
            int(d["end"]),
        )


@dataclass
class TaggedResult(JsonSerializable):
    synthesis_result_batch: SynthesisResultBatch
    is_dumpped: bool

    def to_json(self) -> Any:
        return self.synthesis_result_batch.to_json()

    @classmethod
    def from_json(cls, d: Any) -> "TaggedResult":
        return TaggedResult(
            SynthesisResultBatch.from_json(d),
            True,
        )


@dataclass
class HunkRepairResult:
    buggy_hunk: BuggyHunk
    results: list[TaggedResult]


# bug_id -> file_id -> hunk_id -> result
RepairResultDict = dict[str, list[list[HunkRepairResult]]]
# repair_idx -> repair_result
RepairResultDicts = list[RepairResultDict]


@dataclass
class RepairResult(IORetrospective):
    config: MetaConfig
    results: RepairResultDicts
    repair_configs: list[RepairConfig]

    is_config_saved: bool
    is_repair_config_saved: list[bool]

    @staticmethod
    def init(config: MetaConfig) -> "RepairResult":
        # print(f"Metadata will be saved in {root}")
        return RepairResult(config, [], [], False, [])

    @classmethod
    def load(cls, path: Path) -> "RepairResult":
        assert path.exists()
        meta_config_path = path / META_CONFIG_FNAME
        assert meta_config_path.exists()
        with open(meta_config_path) as f:
            config: MetaConfig = json.load(f)
        repair_configs: list[RepairConfig] = []
        results: RepairResultDicts = []
        repair_dirs = list(filter(Path.is_dir, path.iterdir()))
        repair_dirs.sort(
            key=lambda dir: int(dir.name[len(REPAIR_RESULT_PATH_PREFIX) :])
        )
        for d_repair in repair_dirs:
            repair_configs.append(
                RepairConfig.from_json_file(d_repair / REPAIR_CONFIG_FNAME)
            )
            result_dict: RepairResultDict = {}
            for d_bug_id in filter(Path.is_dir, d_repair.iterdir()):
                bug_id = d_bug_id.name
                files = result_dict.setdefault(bug_id, [])
                d_files = [
                    (tuple(map(int, p.name.split("-"))), p) for p in d_bug_id.iterdir()
                ]
                d_files.sort(key=lambda tp: tp[0])
                for (f_id, h_id), d_hunks in d_files:
                    assert d_hunks.is_dir()
                    buggy_hunk = BuggyHunk.from_json_file(d_hunks / "bug.json")
                    assert len(files) == f_id
                    if h_id == 0:
                        hunks: list[HunkRepairResult] = []
                        files.append(hunks)
                    else:
                        hunks = files[-1]
                    assert len(hunks) == h_id
                    hunk = HunkRepairResult(buggy_hunk, [])
                    f_tagged_result_jsons = list(
                        filter(lambda p: p.name != "bug.json", d_hunks.iterdir())
                    )
                    f_tagged_result_jsons.sort(key=lambda f: int(f.stem))
                    for f_tagged_result_json in f_tagged_result_jsons:
                        assert f_tagged_result_json.is_file()
                        tagged_result = TaggedResult.from_json_file(
                            f_tagged_result_json
                        )
                        hunk.results.append(tagged_result)
                    hunks.append(hunk)
            results.append(result_dict)
        return RepairResult(
            config, results, repair_configs, True, [True] * len(repair_configs)
        )

    def dump_metadata(self, path: Path):
        if not self.is_config_saved:
            self.config.save_json(path / META_CONFIG_FNAME)
            with open(path / "sys_args.txt", "w") as f:
                json.dump(sys.argv, f)
            with open(path / "meta.txt", "w") as f:
                log_git_repo(f, "Repair tool", Path("."))
                log_git_repo(f, "Defects4J", self.config.d4j_home)
                log_git_repo(f, "Language server", self.config.jdt_ls_repo)
                f.write(f"Defects4J checkout path: {self.config.d4j_checkout_root}\n")
            self.is_config_saved = True

    def check(self):
        assert len(self.repair_configs) == len(self.is_repair_config_saved)
        assert len(self.repair_configs) == len(self.results)

    def dump_repair_configs(self, path: Path):
        self.check()
        for idx, repair_config in enumerate(self.repair_configs):
            if self.is_repair_config_saved[idx]:
                continue
            path = path / (REPAIR_RESULT_PATH_PREFIX + str(idx))
            path.mkdir()
            repair_config.save_json(path / REPAIR_CONFIG_FNAME)
            self.is_repair_config_saved[idx] = True

    def report_timeout_error(self):
        path = self.root / "timeout.times"
        if path.exists():
            with open(path) as f:
                times = int(f.read().strip()) + 1
        else:
            times = 1
        with open(path, "w") as f:
            f.write(str(times))

    def add_repair_config(self, config: RepairConfig):
        self.repair_configs.append(config)
        self.is_repair_config_saved.append(False)
        self.results.append({})

    def add(
        self,
        bug_id: str,
        hunk_id: tuple[int, int],
        result: SynthesisResultBatch,
        buggy_text_file: TextFile,
        buggy_hunk_start_index: int,
        buggy_hunk_end_index: int,
    ) -> None:
        """Remember to add repair config first"""
        self.check()
        f_id, h_id = hunk_id
        result_dict = self.results[-1]
        files = result_dict.setdefault(bug_id, [])
        # Ensure that results are added wrt the order [(0,0), (0,1), (0,2), (1,0), (1,1)...]
        if f_id >= len(files):
            assert f_id == len(files)
            files.append([])
        assert len(files) > 0
        hunks = files[f_id]
        if h_id >= len(hunks):
            assert h_id == len(hunks)
            buggy_hunk = BuggyHunk(
                buggy_text_file,
                buggy_hunk_start_index,
                buggy_hunk_end_index,
            )
            hunks.append(HunkRepairResult(buggy_hunk, []))
        assert len(hunks) >= 0
        hunk = hunks[h_id]
        hunk.results.append(TaggedResult(result, False))

    def dump(self, path: Path):
        self.check()
        self.dump_metadata(path)
        self.dump_repair_configs(path)
        for idx, _ in enumerate(self.repair_configs):
            repair_dir = REPAIR_RESULT_PATH_PREFIX + str(idx)
            result_dict = self.results[idx]
            repair_path = path / repair_dir
            repair_path.mkdir(exist_ok=True)
            for bug_id, files in result_dict.items():
                bug_path = repair_path / bug_id
                bug_path.mkdir(exist_ok=True)
                for f_id, hunks in enumerate(files):
                    for h_id, hunk in enumerate(hunks):
                        hunk_idx_str = f"{f_id}-{h_id}"
                        hunk_path = bug_path / hunk_idx_str
                        hunk_path.mkdir(exist_ok=True)
                        hunk.buggy_hunk.save_json(hunk_path / "bug.json")
                        for idx, tagged_result in enumerate(hunk.results):
                            if tagged_result.is_dumpped:
                                continue
                            tagged_result.is_dumpped = True
                            tagged_result.save_json(hunk_path / f"{idx}.json")


RULE = "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"


def log_git_repo(f: io.TextIOBase, tag: str, repo_path: Path, new_line: bool = True):
    repo = git.Repo(repo_path)
    f.writelines(
        [
            f"[{tag}] Git hash: {repo.head.object.hexsha}\n",
            f"[{tag}] Git status: ",
            RULE,
            repo.git.status(),
            RULE,
            f"[{tag}] Git diff:",
            repo.git.diff(),
            RULE,
        ]
    )
    if new_line:
        f.write("\n")


# for result in result_batch.results:
#     idx += 1
#     if isinstance(result, Unfinished):
#         (save_dir / str(Unfinished)).touch(exist_ok=True)
#     elif isinstance(result, PrunedHalfway):
#         (save_dir / str(PrunedHalfway)).touch(exist_ok=True)
#     else:
#         assert isinstance(result, SynthesisSuccessful)
#         debug_dir = save_dir / "debug"
#         debug_dir.mkdir(exist_ok=True, parents=False)
#         buggy_file_path = tagged_result.buggy_file_path
#         with open(buggy_file_path) as f:
#             buggy_file_lines = f.readlines()
#             assert isinstance(result, SynthesisSuccessful)
#             unified_diff = difflib.unified_diff(
#                 buggy_file_lines,
#                 result.patch.content.splitlines(keepends=True),
#                 fromfile="bug",
#                 tofile="patch",
#             )
#         with open(
#             save_dir / result.patch.path.with_suffix(".json").name,
#             "w",
#         ) as f:
#             json.dump(
#                 {
#                     "path": str(result.patch.path.absolute()),
#                     "content": result.patch.content,
#                     "time": avg_time,
#                     "hunk": hunk_idx,
#                     "synthesis_config": len(self.repair_configs)
#                     - 1,
#                 },
#                 f,
#                 indent=2,
#             )
#         with open(
#             debug_dir / buggy_file_path.with_suffix(".hunk").name,
#             "w",
#         ) as f:
#             f.write(result.hunk)
#         with open(
#             debug_dir / buggy_file_path.with_suffix(".diff").name,
#             "w",
#         ) as f:
#             f.writelines(unified_diff)
# tagged_result.is_dumpped = True
