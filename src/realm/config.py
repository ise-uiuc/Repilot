from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .d4j import Defects4J
from .utils import JsonSerializable, JsonSpecificDirectoryDumpable


@dataclass(frozen=True)
class MetaConfig(JsonSpecificDirectoryDumpable):
    """Repair configurations that are not constantly changed once set"""

    d4j_home: Path
    d4j_checkout_root: Path
    jdt_ls_repo: Path
    java8_home: Path
    # Should be specified without the '--path' argument
    # And note to call it with a higher version of Java (e.g., Java 18)
    language_server_cmd: list[str]
    seed: int

    @classmethod
    def name(cls) -> str:
        return "meta_config.json"

    def to_json(self) -> Any:
        return {
            "d4j_home": str(self.d4j_home),
            "d4j_checkout_root": str(self.d4j_checkout_root),
            "jdt_ls_repo": str(self.jdt_ls_repo),
            "java8_home": str(self.java8_home),
            "language_server_cmd": self.language_server_cmd,
            "seed": self.seed,
        }

    @classmethod
    def from_json(cls, d: dict) -> "MetaConfig":
        return MetaConfig(
            Path(d["d4j_home"]),
            Path(d["d4j_checkout_root"]),
            Path(d["jdt_ls_repo"]),
            Path(d["java8_home"]),
            d["language_server_cmd"],
            d["seed"],
        )

    def d4j(self) -> Defects4J:
        return Defects4J(self.d4j_home, self.d4j_checkout_root, self.java8_home)


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
        return SYNTHESIS_METHOD_MAP[self]

    @classmethod
    def from_json(cls, d: str) -> "SynthesisMethod":
        return SYNTHESIS_METHOD_REV_MAP[d]


SYNTHESIS_METHOD_MAP = {
    SynthesisMethod.PRUNED_NO_MEM: str(SynthesisMethod.PRUNED_NO_MEM),
    SynthesisMethod.PRUNED_MEM: str(SynthesisMethod.PRUNED_MEM),
    SynthesisMethod.PLAIN: str(SynthesisMethod.PLAIN),
}

SYNTHESIS_METHOD_REV_MAP = {value: key for key, value in SYNTHESIS_METHOD_MAP.items()}


@dataclass(frozen=True)
class LMInferenceConfig(JsonSerializable):
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


@dataclass(frozen=True)
class RepairConfig(JsonSpecificDirectoryDumpable):
    n_samples: int
    lm_inference_config: LMInferenceConfig
    method: SynthesisMethod
    bug_pattern: str
    hunk_only: bool

    @classmethod
    def name(cls) -> str:
        return "repair_config.json"

    @property
    def batch_size(self) -> int:
        return self.lm_inference_config.batch_size

    def to_json(self) -> Any:
        return {
            "n_samples": self.n_samples,
            "lm_inference_config": self.lm_inference_config.to_json(),
            "method": self.method.to_json(),
            "bug_pattern": self.bug_pattern,
            "hunk_only": self.hunk_only,
        }

    @classmethod
    def from_json(cls, d: dict) -> "RepairConfig":
        inference_config = LMInferenceConfig.from_json(d["lm_inference_config"])
        return RepairConfig(
            int(d["n_samples"]),
            inference_config,
            SynthesisMethod.from_json(d["method"]),
            str(d["bug_pattern"]),
            bool(d["hunk_only"]),
        )


@dataclass(frozen=True)
class ValidationConfig(JsonSerializable):
    n_cores: int
    bug_pattern: str

    def to_json(self) -> Any:
        return {
            "n_cores": self.n_cores,
            "bug_pattern": self.bug_pattern,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ValidationConfig":
        return ValidationConfig(
            int(d["n_cores"]),
            str(d["bug_pattern"]),
        )
