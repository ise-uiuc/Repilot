import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from . import utils
from .config import MetaConfig
from .d4j import Defects4J
from .results import RepairResult, RepairTransformedResult, ValidationResult

Result = TypeVar("Result")


@dataclass
class Report(utils.IORetrospective):
    """A integrated report type that represents all the information recorded and to be dumped"""

    root: Path
    config: MetaConfig
    _repair_result: RepairResult | None | tuple[()]
    _transformed_result: RepairTransformedResult | None | tuple[()]
    _validation_result: ValidationResult | None | tuple[()]

    @property
    def repair_result(self) -> RepairResult | None:
        if self._repair_result == ():
            print(f"[{self.root.name}] Loading raw generation data...")
            self._repair_result = RepairResult.load(self.root)
            print("Done")
        assert not isinstance(self._repair_result, tuple)
        return self._repair_result

    @repair_result.setter
    def repair_result(self, value: RepairResult | None):
        self._repair_result = value

    @property
    def transformed_result(self) -> RepairTransformedResult | None:
        if self._transformed_result == ():
            print(f"[{self.root.name}] Loading transformed raw generation data...")
            self._transformed_result = RepairTransformedResult.try_load(self.root)
            print("Done")
        assert not isinstance(self._transformed_result, tuple)
        return self._transformed_result

    @transformed_result.setter
    def transformed_result(self, value: RepairTransformedResult | None):
        self._transformed_result = value

    @property
    def validation_result(self) -> ValidationResult | None:
        if self._validation_result == ():
            print(f"[{self.root.name}] Loading validation raw data...")
            self._validation_result = ValidationResult.try_load(self.root)
            print("Done")
        assert not isinstance(self._validation_result, tuple)
        return self._validation_result

    @validation_result.setter
    def validation_result(self, value: ValidationResult | None):
        self._validation_result = value

    def get_d4j(self) -> Defects4J:
        # TODO (low priority): can be optimized
        return self.config.d4j()

    def dump_metadata(self, path: Path):
        assert isinstance(path, Path)
        if not (MetaConfig.json_save_path(path)).exists():
            self.config.dump(path)
        # if not (sys_args_path := path / f"sys_args_{os.getpid()}.txt").exists():
        #     with open(sys_args_path, "w") as f:
        #         json.dump(sys.argv, f)
        #         f.write("\n")
        #         json.dump(dict(os.environ), f, indent=2)
        if not (meta_path := path / "meta.txt").exists():
            with open(meta_path, "w") as f:
                utils.log_git_repo(f, "Repair tool", Path("."))
                utils.log_git_repo(f, "Defects4J", self.config.d4j_home)
                utils.log_git_repo(f, "Language server", self.config.jdt_ls_repo)
                f.write(f"Defects4J checkout path: {self.config.d4j_checkout_root}\n")

    def save(self):
        self.dump(self.root)

    def save_validation_result(self):
        if self.validation_result is not None:
            self.validation_result.dump(self.root)

    def dump(self, path: Path):
        self.dump_metadata(path)
        if self.repair_result is not None:
            self.repair_result.dump(path)
        if self.transformed_result is not None:
            self.transformed_result.dump(path)
        if self.validation_result is not None:
            self.validation_result.dump(path)
        # for idx, config in enumerate(self.validation_configs):
        #     if not (config_path := self.root / f"{idx}{VAL_CONFIG_SUFFIX}").exists():
        #         config.dump(config_path)

    @classmethod
    def load(cls, path: Path) -> "Report":
        config = MetaConfig.load(path)
        return cls.load_from_meta_config(path, config)

    @classmethod
    def load_from_meta_config(cls, path: Path, meta_config: MetaConfig) -> "Report":
        repair_result = ()  # RepairResult.load(path)
        transformed_result = ()  # RepairTransformedResult.try_load(path)
        validation_result = ()  # ValidationResult.try_load(path)
        return Report(
            path,
            meta_config,
            repair_result,
            transformed_result,
            validation_result,
        )
