from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from realm import utils
from realm.model import CodeT5Large

GenerationState = bytes


@dataclass
class Memorization:
    infeasible_token_ids: Dict[GenerationState, List[int]]
    feasible_token_ids: Dict[GenerationState, Set[int]]
    completions: Dict[GenerationState, Optional[List[dict]]]
    denied_tokens: Dict[GenerationState, utils.Trie]

    @staticmethod
    def init() -> 'Memorization':
        return Memorization({}, {}, {}, {})


@dataclass
class GenerationContext:
    generated_tokens: List[str]
    generated_ids: List[int]

MODEL = CodeT5Large.init().to(utils.DEVICE)  # type: ignore # noqa
