from dataclasses import dataclass
from pathlib import Path
from realm.config import MetaConfig, RepairConfig
from realm.generation_defs import SynthesisResultBatch, AvgSynthesisResult
from realm.utils import JsonSerializable, IORetrospective
from .repair_result import RepairResult
from typing import Any


ANALYSIS_FNAME = "repair_analysis.json"


@dataclass(frozen=True)
class AvgHunkPatch(JsonSerializable):
    avg_result: AvgSynthesisResult
    buggy_file_path: Path
    is_duplicate: bool

    def to_json(self) -> Any:
        return {
            "avg_result": self.avg_result.to_json(),
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
