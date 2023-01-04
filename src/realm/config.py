from typing import NamedTuple
from pathlib import Path
from enum import Enum, auto
from .generation_defs import LMInferenceConfig
from typing import Any


class MetaConfig(NamedTuple):
    """Repair configurations that are not constantly changed once set"""

    d4j_home: Path
    d4j_checkout_root: Path
    jdt_ls_repo: Path
    seed: int

    def to_json(self) -> Any:
        return {
            "d4j_home": str(self.d4j_home),
            "d4j_checkout_root": str(self.d4j_checkout_root),
            "jdt_ls_repo": str(self.jdt_ls_repo),
            "seed": self.seed,
        }


class SynthesisMethod(Enum):
    PLAIN = auto()
    PRUNED_MEM = auto()
    PRUNED_NO_MEM = auto()

    def is_plain(self) -> bool:
        return self == SynthesisMethod.PLAIN

    def is_pruned(self) -> bool:
        return (
            self == SynthesisMethod.PRUNED_MEM or self == SynthesisMethod.PRUNED_NO_MEM
        )

    def use_mem(self) -> bool:
        return self == SynthesisMethod.PRUNED_MEM

    def to_json(self) -> Any:
        return str(self)


class SynthesisConfig(NamedTuple):
    n_samples: int
    lm_inference_config: LMInferenceConfig
    method: SynthesisMethod

    @property
    def batch_size(self) -> int:
        return self.lm_inference_config.batch_size

    def to_json(self) -> Any:
        return {
            "n_samples": self.n_samples,
            "lm_inference_config": self.lm_inference_config.to_json(),
            "method": self.method.to_json(),
        }
