import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import git

from .config import MetaConfig, RepairConfig
from .generation_defs import SynthesisResultBatch, AvgSynthesisResult
from .utils import JsonSerializable, IORetrospective

META_CONFIG_FNAME = "meta_config.json"
REPAIR_CONFIG_FNAME = "repair_config.json"
REPAIR_RESULT_PATH_PREFIX = "Repair-"


@dataclass
class TaggedResult(JsonSerializable):
    synthesis_result_batch: SynthesisResultBatch
    buggy_file_path: Path
    is_dumpped: bool

    def to_json(self) -> Any:
        return {
            "synthesis_result_batch": self.synthesis_result_batch.to_json(),
            "buggy_file_path": str(self.buggy_file_path),
            "is_dumpped": self.is_dumpped,
        }

    @classmethod
    def from_json(cls, d: Any) -> "TaggedResult":
        return TaggedResult(
            SynthesisResultBatch.from_json(d["synthesis_result_batch"]),
            Path(d["buggy_file_path"]),
            bool(d["is_dumpped"]),
        )


# bug_id -> hunk_id -> result
RepairResult = dict[str, dict[tuple[int, int], list[TaggedResult]]]
# repair_idx -> repair_result
RepairResults = list[RepairResult]


@dataclass(frozen=True)
class AvgHunkPatch(JsonSerializable):
    avg_result: AvgSynthesisResult
    buggy_file_path: Path
    is_duplicate: bool

    def to_json(self) -> Any:
        return {
            "avg_result": AvgSynthesisResult,
            "buggy_file_path": str(self.buggy_file_path),
            "is_duplicate": self.is_duplicate,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgHunkPatch":
        return AvgHunkPatch(
            AvgSynthesisResult.from_json(d["avg_result"]),
            Path(d["buggy_file_path"]),
            bool(d["is_duplicate"]),
        )


AvgPatch = list[AvgHunkPatch]


@dataclass(frozen=True)
class AnalysisResult(JsonSerializable):
    # bug_id -> patch_idx -> entire patch
    result_dict: dict[str, list[AvgPatch]]

    def to_json(self) -> Any:
        return {
            bug_id: [
                [hunk.to_json() for hunk in avg_patch] for avg_patch in avg_patches
            ]
            for bug_id, avg_patches in self.result_dict.items()
        }

    @classmethod
    def from_json(cls, d: dict) -> "AnalysisResult":
        return AnalysisResult(
            {
                bug_id: [
                    [AvgHunkPatch.from_json(hunk) for hunk in avg_patch]
                    for avg_patch in avg_patches
                ]
                for bug_id, avg_patches in d.items()
            }
        )


AnalysisResults = list[AnalysisResult]


@dataclass
class Reporter(IORetrospective):
    root: Path
    config: MetaConfig
    results: RepairResults
    repair_configs: list[RepairConfig]

    def __post_init__(self):
        assert self.root.exists()

    def analysis(self) -> AnalysisResults:
        all_appeared: dict[str, set[str]] = {}
        a_results: AnalysisResults = []
        for result in self.results:
            result_dict: dict[str, list[AvgPatch]] = {}
            for bug_id, hunk_dict in result.items():
                patches = result_dict.setdefault(bug_id, [])
                # zipped = zip(
                #     *(
                #         tagged_results
                #         for _, tagged_results in hunk_dict.items()
                #     )
                # )
                result_iter = (
                    [
                        (
                            avg_result,
                            tagged_result.buggy_file_path,
                        )
                        for tagged_result in tagged_results
                        for avg_result in tagged_result.synthesis_result_batch.to_average_results()
                    ]
                    for _, tagged_results in hunk_dict.items()
                )
                zipped = zip(*result_iter)
                appeared = all_appeared.setdefault(bug_id, set())
                for patch in zipped:
                    assert len(set(str(hunk[1]) for hunk in patch)) == 1
                    is_duplicate = False
                    if all(
                        avg_result.result.successful_result is not None
                        for avg_result, _ in patch
                    ):
                        concat_hunk_str = "".join(
                            avg_result.result.successful_result.hunk
                            for avg_result, _ in patch
                        )
                        if concat_hunk_str in appeared:
                            is_duplicate = True
                        else:
                            appeared.add(concat_hunk_str)
                    patches.append(
                        [
                            AvgHunkPatch(avg_result, path, is_duplicate)
                            for avg_result, path in patch
                        ]
                    )
                # for patch_batch in zipped:
                #     if any(hunk.synthesis_result_batch for hunk in patch)

                # for hunk_id, tagged_results in hunk_dict.items():
                #     result_iter = (
                #         (
                #             avg_result,
                #             tagged_result.buggy_file_path,
                #         )
                #         for tagged_result in tagged_results
                #         for avg_result in tagged_result.synthesis_result_batch.to_average_results()
                #     )
                #     for patch_idx, (avg_result, buggy_path) in enumerate(result_iter):
                #         if patch_idx >= len(patches):
                #             patch: AvgPatch = []
                #             patches.append(patch)
                #         else:
                #             patch = patches[patch_idx]
                #         bug_hunk_id = (bug_id, hunk_id[0], hunk_id[1])
                #         appeared = all_appeared.setdefault(bug_hunk_id, set())
                #         success = avg_result.result.successful_result
                #         is_duplicate = False
                #         if success is not None:
                #             if success.hunk in appeared:
                #                 is_duplicate = True
                #             else:
                #                 appeared.add(success.hunk)
                #         patch.append(AvgHunkPatch(avg_result, buggy_path, is_duplicate))
            a_results.append(AnalysisResult(result_dict))
        return a_results

    @staticmethod
    def create(root: Path, config: MetaConfig) -> "Reporter":
        root.mkdir()
        print(f"Metadata will be saved in {root}")
        reporter = Reporter(root, config, [], [])
        reporter.dump_metadata()
        return reporter

    @classmethod
    def load(cls, root: Path) -> "Reporter":
        assert root.exists()
        with open(root / META_CONFIG_FNAME) as f:
            config: MetaConfig = json.load(f)
        repair_configs: list[RepairConfig] = []
        results: RepairResults = []
        repair_dirs = list(filter(Path.is_dir, root.iterdir()))
        repair_dirs.sort(
            key=lambda dir: int(dir.name[len(REPAIR_RESULT_PATH_PREFIX) :])
        )
        for d_repair in repair_dirs:
            repair_configs.append(
                RepairConfig.from_json_file(d_repair / REPAIR_CONFIG_FNAME)
            )
            result_dict: RepairResult = {}
            for d_bug_id in filter(Path.is_dir, d_repair.iterdir()):
                bug_id = d_bug_id.name
                hunk_dict = result_dict.setdefault(bug_id, {})
                for d_hunk_id in d_bug_id.iterdir():
                    assert d_hunk_id.is_dir()
                    hunk_id = cast(
                        tuple[int, int], tuple(map(int, d_hunk_id.name.split("-")))
                    )
                    assert len(hunk_id) == 2
                    tagged_results = hunk_dict.setdefault(hunk_id, [])
                    f_tagged_result_jsons = list(d_hunk_id.iterdir())
                    f_tagged_result_jsons.sort(key=lambda f: int(f.stem))
                    for f_tagged_result_json in f_tagged_result_jsons:
                        assert f_tagged_result_json.is_file()
                        tagged_result = TaggedResult.from_json_file(
                            f_tagged_result_json
                        )
                        tagged_results.append(tagged_result)
            results.append(result_dict)
        return Reporter(root, config, results, repair_configs)

    def dump_metadata(self):
        self.config.save_json(self.root / META_CONFIG_FNAME)
        with open(self.root / "sys_args.txt", "w") as f:
            json.dump(sys.argv, f)
        with open(self.root / "meta.txt", "w") as f:
            log_git_repo(f, "Repair tool", Path("."))
            log_git_repo(f, "Defects4J", self.config.d4j_home)
            log_git_repo(f, "Language server", self.config.jdt_ls_repo)
            f.write(f"Defects4J checkout path: {self.config.d4j_checkout_root}\n")

    def dump_repair_config(self, config: RepairConfig):
        path = self.root / (REPAIR_RESULT_PATH_PREFIX + str(len(self.repair_configs)))
        path.mkdir()
        config.save_json(path / REPAIR_CONFIG_FNAME)
        self.repair_configs.append(config)

    def report_timeout_error(self):
        path = self.root / "timeout.times"
        if path.exists():
            with open(path) as f:
                times = int(f.read().strip()) + 1
        else:
            times = 1
        with open(path, "w") as f:
            f.write(str(times))

    def add(
        self,
        bug_id: str,
        hunk_id: tuple[int, int],
        result: SynthesisResultBatch,
        buggy_file_path: Path,
    ) -> None:
        assert len(self.repair_configs) > 0
        if len(self.results) != len(self.repair_configs):
            assert len(self.repair_configs) == len(self.results) + 1
            self.results.append({})
        result_dict = self.results[-1]
        hunk_dict = result_dict.setdefault(bug_id, {})
        tagged_results = hunk_dict.setdefault(hunk_id, [])
        tagged_results.append(TaggedResult(result, buggy_file_path, False))

    def dump(self, root: Path):
        assert len(self.repair_configs) > 0
        assert len(self.repair_configs) == len(self.results)
        repair_dir = REPAIR_RESULT_PATH_PREFIX + str(len(self.repair_configs) - 1)
        result_dict = self.results[-1]
        for bug_id, hunk_dict in result_dict.items():
            for hunk_idx, tagged_results in hunk_dict.items():
                for idx, tagged_result in enumerate(tagged_results):
                    if tagged_result.is_dumpped:
                        continue
                    hunk_idx_str = f"{hunk_idx[0]}-{hunk_idx[1]}"
                    save_dir = root / repair_dir / bug_id / hunk_idx_str
                    save_dir.mkdir(exist_ok=True, parents=True)
                    tagged_result.is_dumpped = True
                    tagged_result.save_json(save_dir / f"{idx}.json")

    def save(self):
        self.dump(self.root)


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
