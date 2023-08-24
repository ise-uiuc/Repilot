from dataclasses import dataclass
from typing import Any

from repilot import utils


@dataclass(frozen=True)
class GenerationDatapoint(utils.JsonSerializable):
    """Represents the number of total and uniques patches generated given the time consumed"""

    gen_time: float
    n_total: int
    n_unique: int
    n_unfinished: int
    n_pruned: int

    def __add__(self, other: "GenerationDatapoint") -> "GenerationDatapoint":
        return GenerationDatapoint(
            self.gen_time + other.gen_time,
            self.n_total + other.n_total,
            self.n_unique + other.n_unique,
            self.n_unfinished + other.n_unfinished,
            self.n_pruned + other.n_pruned,
        )

    @classmethod
    def zero(cls) -> "GenerationDatapoint":
        return GenerationDatapoint(0.0, 0, 0, 0, 0)

    @property
    def average_time(self) -> float:
        return self.gen_time / self.n_total

    def to_json(self) -> Any:
        return {
            "gen_time": self.gen_time,
            "n_total": self.n_total,
            "n_unique": self.n_unique,
            "n_unfinished": self.n_unfinished,
            "n_pruned": self.n_pruned,
        }

    @classmethod
    def from_json(cls, d: dict) -> "GenerationDatapoint":
        return GenerationDatapoint(
            float(d["gen_time"]),
            int(d["n_total"]),
            int(d["n_unique"]),
            int(d["n_unfinished"]),
            int(d["n_pruned"]),
        )


@dataclass
class ValidationDatapoint(utils.JsonSerializable):
    """Represents the number of parsable, compilable, and plausible patches achieved given the time consumed"""

    n_parse_success: int
    n_comp_success: int
    n_test_success: int
    total_time_consumed: float
    gen_datapoint: GenerationDatapoint

    def compilable_by_parsable(self) -> float:
        return self.n_comp_success / self.n_parse_success

    def plausible_by_parsable(self) -> float:
        return self.n_test_success / self.n_parse_success

    def unique_compilation_rate(self) -> float:
        return utils.safe_div(
            self.n_comp_success,
            self.gen_datapoint.n_unique
            - self.gen_datapoint.n_pruned
            - self.gen_datapoint.n_unfinished,
        )

    def unique_plausible_rate(self) -> float:
        return utils.safe_div(
            self.n_test_success,
            self.gen_datapoint.n_unique
            - self.gen_datapoint.n_pruned
            - self.gen_datapoint.n_unfinished,
        )

    def compilation_rate(self) -> float:
        return self.n_comp_success / self.gen_datapoint.n_total

    def plausible_rate(self) -> float:
        return self.n_test_success / self.gen_datapoint.n_total

    def to_json(self) -> Any:
        return {
            "n_parse_success": self.n_parse_success,
            "n_comp_success": self.n_comp_success,
            "n_test_success": self.n_test_success,
            "total_time_consumed": self.total_time_consumed,
            "gen_datapoint": self.gen_datapoint.to_json(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "ValidationDatapoint":
        return ValidationDatapoint(
            int(d["n_parse_success"]),
            int(d["n_comp_success"]),
            int(d["n_test_success"]),
            float(d["total_time_consumed"]),
            GenerationDatapoint.from_json(d["gen_datapoint"]),
        )

    def __add__(self, other: "ValidationDatapoint") -> "ValidationDatapoint":
        return ValidationDatapoint(
            self.n_parse_success + other.n_parse_success,
            self.n_comp_success + other.n_comp_success,
            self.n_test_success + other.n_test_success,
            self.total_time_consumed + other.total_time_consumed,
            self.gen_datapoint + other.gen_datapoint,
        )

    @classmethod
    def zero(cls) -> "ValidationDatapoint":
        return ValidationDatapoint(0, 0, 0, 0.0, GenerationDatapoint.zero())
