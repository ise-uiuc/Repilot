import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from repilot.config import ValidationConfig
from repilot.utils import JsonSerializable, JsonSpecificDirectoryDumpable

VALIDATION_FNAME = "validation_result.json"


class Outcome(Enum):
    ParseError = 0
    CompilationError = 1
    TestingError = 2
    Success = 3

    def to_json(self) -> Any:
        return OUTCOME_MAP[self]

    @classmethod
    def from_json(cls, d: str) -> "Outcome":
        return OUTCOME_REV_MAP[d]


OUTCOME_MAP = {
    Outcome.ParseError: str(Outcome.ParseError),
    Outcome.CompilationError: str(Outcome.CompilationError),
    Outcome.TestingError: str(Outcome.TestingError),
    Outcome.Success: str(Outcome.Success),
}

OUTCOME_REV_MAP = {value: key for key, value in OUTCOME_MAP.items()}


@dataclass(frozen=True)
class PatchValidationResult(JsonSerializable):
    outcome: Outcome
    time_cost: float
    stdout: str
    stderr: str

    def to_json(self) -> Any:
        return {
            "outcome": self.outcome.to_json(),
            "time_cost": self.time_cost,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_json(cls, d: Any) -> "PatchValidationResult":
        return PatchValidationResult(
            Outcome.from_json(d["outcome"]),
            float(d["time_cost"]),
            d["stdout"],
            d["stderr"],
        )


@dataclass
class ValidationCache(JsonSerializable):
    """Stores already validated patches"""

    # bug_id -> concat_hunk_str -> result
    result_dict: dict[str, dict[str, PatchValidationResult]]

    def to_json(self) -> Any:
        return {
            bug_id: {
                concat_hunk_str: result.to_json()
                for concat_hunk_str, result in results.items()
            }
            for bug_id, results in self.result_dict.items()
        }

    @classmethod
    def from_json(cls, d: dict[str, dict[str, Any]]) -> "ValidationCache":
        return ValidationCache(
            {
                bug_id: {
                    concat_hunk_str: PatchValidationResult.from_json(result)
                    for concat_hunk_str, result in results.items()
                }
                for bug_id, results in d.items()
            }
        )


@dataclass
class ValidationResult(JsonSpecificDirectoryDumpable):
    # bug_id -> patch_id -> (validation_config_index, result)
    validation_configs: list[ValidationConfig]
    result_dict: dict[str, dict[int, tuple[int, PatchValidationResult]]]

    def to_json(self) -> Any:
        return {
            "validation_configs": [
                config.to_json() for config in self.validation_configs
            ],
            "result_dict": {
                bug_id: {
                    int(patch_idx): (config_idx, patch_result.to_json())
                    for patch_idx, (config_idx, patch_result) in patch_results.items()
                }
                for bug_id, patch_results in self.result_dict.items()
            },
        }

    @classmethod
    def name(cls) -> str:
        return VALIDATION_FNAME

    @classmethod
    def from_json(cls, d: dict) -> "ValidationResult":
        return ValidationResult(
            [ValidationConfig.from_json(config) for config in d["validation_configs"]],
            {
                bug_id: {
                    int(patch_idx): (
                        config_idx,
                        PatchValidationResult.from_json(patch_result),
                    )
                    for patch_idx, (config_idx, patch_result) in patch_results.items()
                }
                for bug_id, patch_results in d["result_dict"].items()
            },
        )

    @classmethod
    def load(cls, path: Path) -> "ValidationResult":
        if (path / cls.name()).exists():
            val_result = ValidationResult.from_json_file(path / cls.name())
        else:
            val_result = ValidationResult([], {})
        if (path / "val_results").exists():
            for dir in (path / "val_results").iterdir():
                bug_id = dir.with_suffix("").with_suffix("").name
                d: dict[str, Any] = json.loads(dir.read_text())
                data = {
                    int(patch_idx): (-1, PatchValidationResult.from_json(patch_result))
                    for patch_idx, patch_result in d.items()
                }
                val_result.result_dict.setdefault(bug_id, {}).update(data)
        return val_result

    @classmethod
    def try_load(cls, path: Path) -> "ValidationResult | None":
        if not (path / "val_results").exists() and not (path / cls.name()).exists():
            return None
        return ValidationResult.load(path)


#     def to_json(self) -> Any:
#         return {
#             "results": [result.to_json() for result in self.results],
#         }

#     @classmethod
#     def from_json(cls, d: dict[str, list]) -> "ValidationResults":
#         return ValidationResults(
#             [ValidationConfig.from_json(config) for config in d["validation_configs"]],
#             [ValidationResult.from_json(result) for result in d["results"]],
#         )


# @dataclass
# class ValidationResults(JsonSpecificDirectoryDumpable):
#     validation_configs: list[ValidationConfig]
#     results: list[ValidationResult]

#     @classmethod
#     def name(cls) -> str:
#         return VALIDATION_FNAME

#     def to_json(self) -> Any:
#         return {
#             "validation_configs": [
#                 config.to_json() for config in self.validation_configs
#             ],
#             "results": [result.to_json() for result in self.results],
#         }

#     @classmethod
#     def from_json(cls, d: dict[str, list]) -> "ValidationResults":
#         return ValidationResults(
#             [ValidationConfig.from_json(config) for config in d["validation_configs"]],
#             [ValidationResult.from_json(result) for result in d["results"]],
#         )
