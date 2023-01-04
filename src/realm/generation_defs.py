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


class PrunedHalfway:
    pass


class Unfinished:
    pass


class SynthesisSuccessful(NamedTuple):
    patch: TextFile
    hunk_start_idx: int
    hunk_end_idx: int
    hunk: str


SynthesisResult = SynthesisSuccessful | PrunedHalfway | Unfinished


class AvgSynthesisResult(NamedTuple):
    result: SynthesisResult
    avg_time_cost: float


class SynthesisResultBatch(NamedTuple):
    # None indicates a failed generation (e.g., due to being unfinished)
    results: List[SynthesisResult]
    time_cost: float

    def to_average_results(self) -> "List[AvgSynthesisResult]":
        avg_time = self.time_cost / len(self.results)
        return [AvgSynthesisResult(result, avg_time) for result in self.results]


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
