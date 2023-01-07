from dataclasses import dataclass
from pathlib import Path
from realm.config import MetaConfig, RepairConfig
from realm.generation_defs import SynthesisResultBatch, AvgSynthesisResult
from realm.utils import JsonSerializable, JsonSpecificDirectoryDumpable
from .repair_result import RepairResult
from typing import Any
from realm.lsp.text import TextFile


ANALYSIS_FNAME = "repair_analysis.json"


# @dataclass(frozen=True)
# class AvgHunkPatch(JsonSerializable):
#     avg_result: AvgSynthesisResult
#     buggy_file_path: Path

#     def to_json(self) -> Any:
#         return {
#             "avg_result": self.avg_result.to_json(),
#             "buggy_file_path": str(self.buggy_file_path),
#         }

#     @classmethod
#     def from_json(cls, d: dict) -> "AvgHunkPatch":
#         return AvgHunkPatch(
#             AvgSynthesisResult.from_json(d["avg_result"]),
#             Path(d["buggy_file_path"]),
#         )


@dataclass(frozen=True)
class AvgFilePatch:
    hunks: list[AvgSynthesisResult]
    bug: TextFile
    buggy_hunk_indices: list[tuple[int, int]]

    @property
    def is_broken(self):
        return any(hunk.result.hunk is None for hunk in self.hunks)

    def to_json(self) -> Any:
        return {
            "hunks": [hunk.to_json() for hunk in self.hunks],
            # "patch": None if self.patch is None else self.patch.to_json(),
            "bug": self.bug.to_json(),
            "buggy_hunk_indices": self.buggy_hunk_indices,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgFilePatch":
        return AvgFilePatch(
            [AvgSynthesisResult.from_json(hunk) for hunk in d["hunks"]],
            # TextFile.from_json(p) if (p := d["patch"]) is not None else None,
            TextFile.from_json(d["bug"]),
            d["buggy_hunk_indices"],
        )

    def compute_patch(self) -> TextFile | None:
        assert self.buggy_hunk_indices == sorted(
            self.buggy_hunk_indices, reverse=True, key=lambda tp: tp[0]
        )
        assert len(self.hunks) > 0
        assert len(self.hunks) == len(self.buggy_hunk_indices)
        patch = self.bug.copy()
        # [first_hunk, *rest_hunks] = hunks
        # last_success = first_hunk.result.successful_result
        # if last_success is None:
        #     return AvgFilePatch(hunks, None, bug, buggy_hunk_indices)
        # last_success = None
        for hunk, (start, end) in zip(self.hunks, self.buggy_hunk_indices):
            assert start <= end
            success = hunk.result.hunk
            if success is None:
                return None
            # assert success.patch.path == patch.path
            # assert success.hunk_start_idx == start
            # assert success.hunk_start_idx <= success.hunk_end_idx
            # if last_success is not None:
            #     assert success.hunk_end_idx <= last_success.hunk_start_idx
            # last_success = success
            # "\n" is impotant. Imagine `start` and `end` are the same.
            patch.modify(start, end, success + "\n")
        # if len(hunks) == 1 and patch is not None:
        #     success = hunks[0].result.hunk
        #     assert success is not None
        #     assert patch.content == success.patch.content
        return patch


@dataclass
class AvgPatch(JsonSerializable):
    file_patches: list[AvgFilePatch]
    is_duplicate: bool

    @property
    def is_broken(self) -> bool:
        return any(file_patch.is_broken for file_patch in self.file_patches)

    def to_json(self) -> Any:
        return {
            "file_patches": [file_patch.to_json() for file_patch in self.file_patches],
            "is_duplicate": self.is_duplicate,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgPatch":
        return AvgPatch(
            [AvgFilePatch.from_json(file_patch) for file_patch in d["file_patches"]],
            bool(d["is_duplicate"]),
        )


@dataclass(frozen=True)
class RepairAnalysisResult(JsonSerializable):
    result_dict: dict[str, list[AvgPatch]]

    def to_json(self) -> Any:
        return {
            bug_id: [avg_patch.to_json() for avg_patch in avg_patches]
            for bug_id, avg_patches in self.result_dict.items()
        }

    @classmethod
    def from_json(cls, d: dict) -> "RepairAnalysisResult":
        return RepairAnalysisResult(
            {
                bug_id: [AvgPatch.from_json(avg_patch) for avg_patch in avg_patches]
                for bug_id, avg_patches in d.items()
            }
        )


@dataclass(frozen=True)
class RepairAnalysisResults(JsonSpecificDirectoryDumpable):
    results: list[RepairAnalysisResult]

    @classmethod
    def name(cls):
        return ANALYSIS_FNAME

    def to_json(self) -> Any:
        return [result.to_json() for result in self.results]

    @classmethod
    def from_json(cls, d: list) -> "RepairAnalysisResults":
        return RepairAnalysisResults([RepairAnalysisResult.from_json(r) for r in d])
