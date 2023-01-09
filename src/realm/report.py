import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import utils
from .config import MetaConfig
from .d4j import Defects4J
from .results import RepairResult, RepairTransformedResult, ValidationResult


@dataclass
class Report(utils.IORetrospective):
    """A integrated report type that represents all the information recorded and to be dumped"""

    root: Path
    config: MetaConfig
    repair_result: RepairResult | None
    transformed_result: RepairTransformedResult | None
    validation_result: ValidationResult | None

    def get_d4j(self) -> Defects4J:
        # TODO (low priority): can be optimized
        return self.config.d4j()

    def dump_metadata(self, path: Path):
        assert isinstance(path, Path)
        if not (MetaConfig.json_save_path(path)).exists():
            self.config.dump(path)
        if not (sys_args_path := path / f"sys_args_{os.getpid()}.txt").exists():
            with open(sys_args_path, "w") as f:
                json.dump(sys.argv, f)
        if not (meta_path := path / "meta.txt").exists():
            with open(meta_path, "w") as f:
                utils.log_git_repo(f, "Repair tool", Path("."))
                utils.log_git_repo(f, "Defects4J", self.config.d4j_home)
                utils.log_git_repo(f, "Language server", self.config.jdt_ls_repo)
                f.write(f"Defects4J checkout path: {self.config.d4j_checkout_root}\n")

    def save(self):
        self.dump(self.root)

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
        meta_config = MetaConfig.load(path)
        repair_result = RepairResult.load(path)
        transformed_result = RepairTransformedResult.try_load(path)
        validation_result = ValidationResult.try_load(path)
        return Report(
            path,
            meta_config,
            repair_result,
            transformed_result,
            validation_result,
        )
