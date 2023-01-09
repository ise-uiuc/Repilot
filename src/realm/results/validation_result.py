from dataclasses import dataclass
from enum import Enum
from typing import Any

from realm.config import ValidationConfig
from realm.utils import JsonSerializable, JsonSpecificDirectoryDumpable

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
class ValidationResult(JsonSpecificDirectoryDumpable):
    # bug_id -> patch_id -> (result, validation_config_index)
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
