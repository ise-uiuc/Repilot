from .results import (
    RepairResult,
    RepairAnalysisResult,
    AnalysisResult,
    RepairAnalysisResult,
    AvgFilePatch,
    AvgPatch,
    TaggedResult,
    BuggyHunk,
    HunkRepairResult,
)
from .generation_defs import AvgSynthesisResult
from .lsp import TextFile
from .utils import IORetrospective
from pathlib import Path
from dataclasses import dataclass
from .config import MetaConfig
from typing import Iterator, cast


@dataclass
class Reporter(IORetrospective):
    root: Path
    repair_result: RepairResult
    analysis_result: RepairAnalysisResult | None

    def __post_init__(self):
        assert self.root.exists()

    @staticmethod
    def create_repair(root: Path, config: MetaConfig):
        root.mkdir()
        print("Metadata will be saved to", root)
        return Reporter(root, RepairResult.init(config), None)

    def save(self):
        self.dump(self.root)

    def dump(self, path: Path):
        self.repair_result.dump(path)
        if self.analysis_result is not None:
            self.analysis_result.dump(path)

    @classmethod
    def load(cls, path: Path) -> "Reporter":
        repair_result = RepairResult.load(path)
        analysis_result = (
            RepairAnalysisResult.load(path)
            if RepairAnalysisResult.file_exists(path)
            else None
        )
        return Reporter(path, repair_result, analysis_result)

    def analyze(self):
        if not isinstance(self.repair_result, RepairResult):
            return
        all_appeared: dict[str, set[str]] = {}
        a_results: list[AnalysisResult] = []
        for result in self.repair_result.results:
            result_dict: dict[str, list[AvgPatch]] = {}
            for bug_id, files in result.items():
                appeared = all_appeared.setdefault(bug_id, set())
                patches: list[AvgPatch] = []
                for patch in iter_files(files):
                    if any(
                        hunk.result.hunk is None
                        for file in patch
                        for hunk in file.hunks
                    ):
                        is_duplicate = False
                    else:
                        concat_hunk_str = "".join(
                            cast(str, hunk_patch.result.hunk)
                            for file_patch in patch
                            for hunk_patch in file_patch.hunks
                        )
                        if concat_hunk_str in appeared:
                            is_duplicate = True
                        else:
                            appeared.add(concat_hunk_str)
                            is_duplicate = False
                    patches.append(AvgPatch(patch, is_duplicate))
                result_dict[bug_id] = patches
            a_results.append(AnalysisResult(result_dict))
        self.analysis_result = RepairAnalysisResult(a_results)


_AvgResult = tuple[AvgSynthesisResult, BuggyHunk]


def iter_files(
    file_results: list[list[HunkRepairResult]],
) -> Iterator[list[AvgFilePatch]]:
    # For one bug
    # items = list(hunk_dict.items())
    # items.sort(key=lambda kv: kv[0])
    groups: list[list[list[_AvgResult]]] = []
    # TODO: maybe take a look at `itertools.groupby`
    # last_f: int | None = None
    for hunk_results in file_results:
        groups.append([])
        group = groups[-1]
        for hunk_result in hunk_results:
            avg_results = [
                (avg_result, hunk_result.buggy_hunk)
                for tagged_result in hunk_result.results
                for avg_result in tagged_result.synthesis_result_batch.to_average_results()
            ]
            # assert len(avg_results) > 0
            # assert f_idx == len(groups)
            # if last_f is None or f_idx != last_f:
            #     group: list[list[_AvgResult]] = []
            #     groups.append(group)
            # else:
            #     group = groups[-1]
            # assert h_idx == len(group)
            group.append(avg_results)
    for group in groups:
        assert len(group) > 0
        assert len(set(len(data) for data in group)) == 1
        assert len(set(t[1].file.path for data in group for t in data)) == 1

    for file_groups in zip(*(zip(*group) for group in groups)):
        assert len(file_groups) > 0
        file_patches: list[AvgFilePatch] = []
        for file_group in file_groups:
            hunks: list[AvgSynthesisResult] = []
            buggy_hunk_indices: list[tuple[int, int]] = []
            assert len(file_group) > 0
            bug: TextFile | None = None
            for avg_result, buggy_hunk in file_group:
                bug = buggy_hunk.file
                buggy_hunk_indices.append((buggy_hunk.start, buggy_hunk.end))
                hunks.append(avg_result)
            assert bug is not None
            file_patches.append(AvgFilePatch(hunks, bug, buggy_hunk_indices))
        yield file_patches
