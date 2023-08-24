from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repilot.config import RepairConfig
from repilot.lsp import TextFile
from repilot.utils import IORetrospective, JsonSerializable

# REPAIR_RESULT_PATH_PREFIX = "Repair-"


@dataclass(frozen=True)
class SynthesisResult(JsonSerializable):
    hunk: str | None
    is_pruned_halfway: bool
    is_unfinished: bool

    def to_json(self) -> Any:
        return {
            "hunk": self.hunk,
            "is_pruned_halfway": self.is_pruned_halfway,
            "is_unfinished": self.is_unfinished,
        }

    @classmethod
    def from_json(cls, d: Any) -> "SynthesisResult":
        return SynthesisResult(
            d["hunk"],
            bool(d["is_pruned_halfway"]),
            bool(d["is_unfinished"]),
        )


@dataclass(frozen=True)
class AvgSynthesisResult(JsonSerializable):
    result: SynthesisResult
    avg_time_cost: float

    def to_json(self) -> Any:
        return {
            "result": self.result.to_json(),
            "avg_time_cost": self.avg_time_cost,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AvgSynthesisResult":
        return AvgSynthesisResult(
            SynthesisResult.from_json(d["result"]), float(d["avg_time_cost"])
        )


@dataclass(frozen=True)
class SynthesisResultBatch(JsonSerializable):
    # None indicates a failed generation (e.g., due to being unfinished)
    results: list[SynthesisResult]
    time_cost: float

    def to_average_results(self) -> "list[AvgSynthesisResult]":
        avg_time = self.time_cost / len(self.results)
        return [AvgSynthesisResult(result, avg_time) for result in self.results]

    def to_json(self) -> Any:
        return {
            "results": [result.to_json() for result in self.results],
            "time_cost": self.time_cost,
        }

    @classmethod
    def from_json(cls, d: dict) -> "SynthesisResultBatch":
        return SynthesisResultBatch(
            [SynthesisResult.from_json(result) for result in d["results"]],
            float(d["time_cost"]),
        )


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
# RepairResultDicts = list[RepairResultDict]


@dataclass
class RepairResult(IORetrospective):
    result_dict: RepairResultDict
    repair_config: RepairConfig

    # @staticmethod
    # def init() -> "RepairResult":
    #     # print(f"Metadata will be saved in {root}")
    #     return RepairResult([], [], [])
    @classmethod
    def try_load(cls, path: Path) -> "RepairResult | None":
        if not (RepairConfig.json_save_path(path)).exists():
            return None
        return cls.load(path)

    @classmethod
    def load(cls, path: Path) -> "RepairResult":
        assert path.exists()
        assert (RepairConfig.json_save_path(path)).exists()
        # repair_configs: RepairConfig = []
        repair_config = RepairConfig.load(path)
        # results: RepairResultDicts = []
        # repair_dirs = list(filter(Path.is_dir, path.iterdir()))
        # repair_dirs.sort(
        #     key=lambda dir: int(dir.name[len(REPAIR_RESULT_PATH_PREFIX) :])
        # )
        # for d_repair in repair_dirs:
        # repair_configs.append(
        # )
        result_dict: RepairResultDict = {}
        for d_bug_id in filter(Path.is_dir, path.iterdir()):
            if d_bug_id.name == "val_results":
                continue
            bug_id = d_bug_id.name
            files = result_dict.setdefault(bug_id, [])
            d_files = [
                (tuple(map(int, p.name.split("-"))), p) for p in d_bug_id.iterdir()
            ]
            d_files.sort(key=lambda tp: tp[0])
            for (f_id, h_id), d_hunks in d_files:
                assert d_hunks.is_dir()
                buggy_hunk = BuggyHunk.from_json_file(d_hunks / "bug.json")
                if h_id == 0:
                    hunks: list[HunkRepairResult] = []
                    files.append(hunks)
                else:
                    hunks = files[-1]
                assert len(files) == f_id + 1, f"{len(files)} != {f_id}"
                assert len(hunks) == h_id
                hunk = HunkRepairResult(buggy_hunk, [])
                f_tagged_result_jsons = list(
                    filter(lambda p: p.name != "bug.json", d_hunks.iterdir())
                )
                f_tagged_result_jsons.sort(key=lambda f: int(f.stem))
                for f_tagged_result_json in f_tagged_result_jsons:
                    assert f_tagged_result_json.is_file()
                    tagged_result = TaggedResult.from_json_file(f_tagged_result_json)
                    hunk.results.append(tagged_result)
                hunks.append(hunk)
        return RepairResult(result_dict, repair_config)

    def dump_repair_config(self, path: Path):
        if not (path := RepairConfig.json_save_path(path)).exists():
            self.repair_config.save_json(path)

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
        f_id, h_id = hunk_id
        result_dict = self.result_dict
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
        self.dump_repair_config(path)
        result_dict = self.result_dict
        for bug_id, files in result_dict.items():
            bug_path = path / bug_id
            bug_path.mkdir(exist_ok=True)
            for f_id, hunks in enumerate(files):
                for h_id, hunk in enumerate(hunks):
                    hunk_idx_str = f"{f_id}-{h_id}"
                    hunk_path = bug_path / hunk_idx_str
                    hunk_path.mkdir(exist_ok=True)
                    if not (hunk_path / "bug.json").exists():
                        hunk.buggy_hunk.save_json(hunk_path / "bug.json")
                    for idx, tagged_result in enumerate(hunk.results):
                        if tagged_result.is_dumpped:
                            continue
                        tagged_result.is_dumpped = True
                        tagged_result.save_json(hunk_path / f"{idx}.json")
