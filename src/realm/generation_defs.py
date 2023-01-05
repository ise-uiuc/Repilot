from typing import Dict, List, Set, Optional, NamedTuple, Any
from dataclasses import dataclass
from realm import utils
from realm.model import CodeT5Large
from realm.lsp.text import TextFile

GenerationState = bytes


@dataclass
class Memorization:
    infeasible_token_ids: Dict[GenerationState, List[int]]
    feasible_token_ids: Dict[GenerationState, Set[int]]
    completions: Dict[GenerationState, Optional[List[dict]]]
    denied_tokens: Dict[GenerationState, utils.Trie]

    @staticmethod
    def init() -> "Memorization":
        return Memorization({}, {}, {}, {})


@dataclass
class GenerationContext:
    generated_tokens: List[str]
    generated_ids: List[int]


@dataclass(frozen=True)
class SynthesisSuccessful(utils.JsonSerializable):
    patch: TextFile
    hunk_start_idx: int
    hunk_end_idx: int
    hunk: str

    def to_json(self) -> Any:
        return {
            "patch": self.patch.to_json(),
            "hunk_start_idx": self.hunk_start_idx,
            "hunk_end_idx": self.hunk_end_idx,
            "hunk": self.hunk,
        }

    @classmethod
    def from_json(cls, d: Any) -> "SynthesisSuccessful":
        return SynthesisSuccessful(
            TextFile.from_json(d["patch"]),
            int(d["hunk_start_idx"]),
            int(d["hunk_end_idx"]),
            str(d["hunk"]),
        )


@dataclass(frozen=True)
class SynthesisResult(utils.JsonSerializable):
    successful_result: Optional[SynthesisSuccessful]
    is_pruned_halfway: bool
    is_unfinished: bool

    def to_json(self) -> Any:
        return {
            "successful_result": None
            if self.successful_result is None
            else self.successful_result.to_json(),
            "is_pruned_halfway": self.is_pruned_halfway,
            "is_unfinished": self.is_unfinished,
        }

    @classmethod
    def from_json(cls, d: Any) -> "SynthesisResult":
        return SynthesisResult(
            None
            if (result := d["successful_result"]) is None
            else SynthesisSuccessful.from_json(result),
            bool(d["is_pruned_halfway"]),
            bool(d["is_unfinished"]),
        )


class AvgSynthesisResult(NamedTuple):
    result: SynthesisResult
    avg_time_cost: float


@dataclass(frozen=True)
class SynthesisResultBatch(utils.JsonSerializable):
    # None indicates a failed generation (e.g., due to being unfinished)
    results: List[SynthesisResult]
    time_cost: float

    def to_average_results(self) -> "List[AvgSynthesisResult]":
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


# MODEL = CodeT5Large.init().to(utils.DEVICE)  # type: ignore # noqa


@dataclass(frozen=True)
class LMInferenceConfig(utils.JsonSerializable):
    batch_size: int
    temperature: float
    top_k: int
    max_new_tokens: int

    def to_json(self) -> Any:
        return {
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def from_json(cls, d: dict) -> "LMInferenceConfig":
        return LMInferenceConfig(
            int(d["batch_size"]),
            float(d["temperature"]),
            int(d["top_k"]),
            int(d["max_new_tokens"]),
        )
