from dataclasses import dataclass
from typing import Any, cast

from repilot.lsp import TextFile
from repilot.utils import (
    BUGGY_HUNK_PLACEHOLDER,
    JsonSerializable,
    JsonSpecificDirectoryDumpable,
)

from .repair_result import AvgSynthesisResult

TRANSFORM_FNAME = "repair_transformed.json"


@dataclass(frozen=True)
class AvgFilePatch:
    hunks: list[AvgSynthesisResult]
    # bug: TextFile
    buggy_hunk_indices: list[tuple[int, int]]

    @property
    def total_gen_time(self) -> float:
        return sum(hunk.avg_time_cost for hunk in self.hunks)

    @property
    def is_broken(self):
        return any(hunk.result.hunk is None for hunk in self.hunks)

    @property
    def is_pruned(self):
        return any(hunk.result.is_pruned_halfway for hunk in self.hunks)

    @property
    def is_unfinished(self):
        return any(hunk.result.is_unfinished for hunk in self.hunks)

    def to_json(self) -> Any:
        return {
            "hunks": [hunk.to_json() for hunk in self.hunks],
            # "patch": None if self.patch is None else self.patch.to_json(),
            # "bug": self.bug.to_json(),
            "buggy_hunk_indices": self.buggy_hunk_indices,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgFilePatch":
        return AvgFilePatch(
            [AvgSynthesisResult.from_json(hunk) for hunk in d["hunks"]],
            # TextFile.from_json(p) if (p := d["patch"]) is not None else None,
            # TextFile.from_json(d["bug"]),
            d["buggy_hunk_indices"],
        )

    def compute_patch(self, bug: TextFile) -> TextFile | None:
        assert self.buggy_hunk_indices == sorted(
            self.buggy_hunk_indices, reverse=True, key=lambda tp: tp[0]
        ), f"{self.buggy_hunk_indices}"
        assert len(self.hunks) > 0
        assert len(self.hunks) == len(self.buggy_hunk_indices)
        patch = bug.copy()
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
            if success == BUGGY_HUNK_PLACEHOLDER:
                continue
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
    def total_gen_time(self) -> float:
        return sum(file_patch.total_gen_time for file_patch in self.file_patches)

    @property
    def is_broken(self) -> bool:
        return any(file_patch.is_broken for file_patch in self.file_patches)

    @property
    def is_unfinished(self) -> bool:
        return any(file_patch.is_unfinished for file_patch in self.file_patches)

    @property
    def is_pruned(self) -> bool:
        return any(file_patch.is_pruned for file_patch in self.file_patches)

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
class RepairTransformedResult(JsonSpecificDirectoryDumpable):
    # bug_id -> (bug_specs, list[patch])
    result_dict: dict[str, tuple[list[TextFile], list[AvgPatch]]]
    all_appeared: dict[str, list[str]]

    @classmethod
    def name(cls):
        return TRANSFORM_FNAME

    def to_json(self) -> Any:
        return {
            "result_dict": {
                bug_id: (
                    [text_file.to_json() for text_file in text_files],
                    [avg_patch.to_json() for avg_patch in avg_patches],
                )
                for bug_id, (text_files, avg_patches) in self.result_dict.items()
            },
            "all_appeared": {
                bug_id: list(appeared) for bug_id, appeared in self.all_appeared.items()
            },
        }

    @classmethod
    def from_json(cls, d: dict) -> "RepairTransformedResult":
        return RepairTransformedResult(
            {
                bug_id: (
                    [TextFile.from_json(text_file) for text_file in text_files],
                    [AvgPatch.from_json(avg_patch) for avg_patch in avg_patches],
                )
                for bug_id, (text_files, avg_patches) in d["result_dict"].items()
            },
            {bug_id: list(appeared) for bug_id, appeared in d["all_appeared"].items()},
        )


def concat_hunks(file_patches: list[AvgFilePatch], delim: str = "") -> str:
    return delim.join(
        cast(str, hunk_patch.result.hunk)
        for file_patch in file_patches
        for hunk_patch in file_patch.hunks
    )
