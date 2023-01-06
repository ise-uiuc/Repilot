from dataclasses import dataclass
from pathlib import Path
from realm.config import MetaConfig, RepairConfig
from realm.generation_defs import SynthesisResultBatch, AvgSynthesisResult
from realm.utils import JsonSerializable, IORetrospective
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
    patch: TextFile | None
    bug: TextFile
    buggy_hunk_indices: list[tuple[int, int]]

    def to_json(self) -> Any:
        return {
            "hunks": [hunk.to_json() for hunk in self.hunks],
            "patch": None if self.patch is None else self.patch.to_json(),
            "bug": self.bug.to_json(),
            "buggy_hunk_indices": self.buggy_hunk_indices,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgFilePatch":
        return AvgFilePatch(
            [AvgSynthesisResult.from_json(hunk) for hunk in d["hunks"]],
            TextFile.from_json(p) if (p := d["patch"]) is not None else None,
            TextFile.from_json(d["bug"]),
            d["buggy_hunk_indices"],
        )

    @staticmethod
    def init(
        hunks: list[AvgSynthesisResult],
        bug: TextFile,
        buggy_hunk_indices: list[tuple[int, int]],
    ) -> "AvgFilePatch":
        assert buggy_hunk_indices == sorted(
            buggy_hunk_indices, reverse=True, key=lambda tp: tp[0]
        )
        assert len(hunks) > 0
        assert len(hunks) == len(buggy_hunk_indices)
        patch = bug.copy()
        # [first_hunk, *rest_hunks] = hunks
        # last_success = first_hunk.result.successful_result
        # if last_success is None:
        #     return AvgFilePatch(hunks, None, bug, buggy_hunk_indices)
        last_success = None
        for hunk, (start, end) in zip(hunks, buggy_hunk_indices):
            assert start <= end
            success = hunk.result.successful_result
            if success is None:
                return AvgFilePatch(hunks, None, bug, buggy_hunk_indices)
            assert success.patch.path == patch.path
            assert success.hunk_start_idx == start
            assert success.hunk_start_idx <= success.hunk_end_idx
            if last_success is not None:
                assert success.hunk_end_idx <= last_success.hunk_start_idx
            last_success = success
            # "\n" is impotant. Imagine `start` and `end` are the same.
            patch.modify(start, end, success.hunk + "\n")
        return AvgFilePatch(hunks, patch, bug, buggy_hunk_indices)


@dataclass
class AvgPatch(JsonSerializable):
    file_patches: list[AvgFilePatch]
    is_duplicate: bool

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
class AnalysisResult(JsonSerializable):
    result_dict: dict[str, list[AvgPatch]]

    def to_json(self) -> Any:
        return {
            bug_id: [avg_patch.to_json() for avg_patch in avg_patches]
            for bug_id, avg_patches in self.result_dict.items()
        }

    @classmethod
    def from_json(cls, d: dict) -> "AnalysisResult":
        return AnalysisResult(
            {
                bug_id: [AvgPatch.from_json(avg_patch) for avg_patch in avg_patches]
                for bug_id, avg_patches in d.items()
            }
        )


@dataclass(frozen=True)
class AnalysisResults(JsonSerializable):
    results: list[AnalysisResult]

    def to_json(self) -> Any:
        return [result.to_json() for result in self.results]

    @classmethod
    def from_json(cls, d: list) -> "AnalysisResults":
        return AnalysisResults([AnalysisResult.from_json(r) for r in d])


@dataclass
class RepairAnalysisResult(IORetrospective):
    repair_result: RepairResult
    analysis_results: AnalysisResults

    @staticmethod
    def file_exists(path: Path) -> bool:
        return (path / ANALYSIS_FNAME).exists()

    @classmethod
    def load(cls, path: Path) -> "RepairAnalysisResult":
        repair_result = RepairResult.load(path)
        analysis_results = AnalysisResults.from_json_file(path / ANALYSIS_FNAME)
        return RepairAnalysisResult(repair_result, analysis_results)

    def dump(self, path: Path):
        self.repair_result.dump(path)
        self.analysis_results.save_json(path / ANALYSIS_FNAME)
