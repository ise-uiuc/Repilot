from .results import (
    RepairResult,
    RepairAnalysisResult,
    AnalysisResult,
    AnalysisResults,
    AvgPatch,
    AvgHunkPatch,
)
from .utils import IORetrospective
from pathlib import Path
from dataclasses import dataclass
from .config import MetaConfig

Result = RepairAnalysisResult | RepairResult


@dataclass
class Reporter(IORetrospective):
    root: Path
    result: RepairResult | RepairAnalysisResult

    def __post_init__(self):
        assert self.root.exists()

    @staticmethod
    def create_repair(root: Path, config: MetaConfig):
        root.mkdir()
        print("Metadata will be saved to", root)
        return Reporter(root, RepairResult.init(config))

    def save(self):
        self.dump(self.root)

    def dump(self, path: Path):
        self.result.dump(path)

    @classmethod
    def load(cls, path: Path) -> "Reporter":
        if RepairAnalysisResult.file_exists(path):
            result: Result = RepairAnalysisResult.load(path)
        else:
            result = RepairResult.load(path)
        return Reporter(path, result)

    def analyze(self):
        if not isinstance(self.result, RepairResult):
            return
        all_appeared: dict[str, set[str]] = {}
        a_results: list[AnalysisResult] = []
        for result in self.result.results:
            result_dict: dict[str, list[AvgPatch]] = {}
            for bug_id, hunk_dict in result.items():
                patches = result_dict.setdefault(bug_id, [])
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
            a_results.append(AnalysisResult(result_dict))
        self.result = RepairAnalysisResult(self.result, AnalysisResults(a_results))
