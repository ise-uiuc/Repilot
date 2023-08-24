from dataclasses import dataclass
from typing import Dict, Optional

from repilot import utils

GenerationState = bytes


@dataclass
class Memorization:
    infeasible_token_ids: Dict[GenerationState, list[int]]
    feasible_token_ids: Dict[GenerationState, dict[int, str | None]]
    completions: Dict[GenerationState, Optional[list[dict]]]
    denied_tokens: Dict[GenerationState, utils.Trie]

    @staticmethod
    def init() -> "Memorization":
        return Memorization({}, {}, {}, {})


@dataclass
class GenerationContext:
    generated_tokens: list[str]
    generated_ids: list[int]
