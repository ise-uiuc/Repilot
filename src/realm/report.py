import json
import sys
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterator, cast

from joblib import Parallel, delayed

from . import utils
from .config import MetaConfig, ValidationConfig
from .generation_defs import AvgSynthesisResult
from .lsp import TextFile
from .d4j import Defects4J
from .results import (
    AvgFilePatch,
    AvgPatch,
    BuggyHunk,
    HunkRepairResult,
    Outcome,
    PatchValidationResult,
    RepairAnalysisResult,
    RepairAnalysisResults,
    RepairResult,
    TaggedResult,
    ValidationResult,
    ValidationResults,
)

META_CONFIG_FNAME = "meta_config.json"
VAL_CONFIG_SUFFIX = "_validation_config.json"


@dataclass
class Report(utils.IORetrospective):
    """A integrated report type that represents all the information recorded and to be dumped"""

    root: Path
    config: MetaConfig
    repair_result: RepairResult
    analysis_result: RepairAnalysisResults | None
    validation_result: ValidationResults | None
    validation_configs: list[ValidationConfig]

    def get_d4j(self) -> Defects4J:
        # TODO (low priority): can be optimized
        return Defects4J(
            self.config.d4j_home, self.config.d4j_checkout_root, self.config.java8_home
        )

    def dump_metadata(self, path: Path):
        assert isinstance(path, Path)
        if not (config_path := path / META_CONFIG_FNAME).exists():
            self.config.dump(config_path)
        if not (sys_args_path := path / "sys_args.txt").exists():
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
        self.repair_result.dump(path)
        if self.analysis_result is not None:
            self.analysis_result.dump(path)
        if self.validation_result is not None:
            self.validation_result.dump(path)
        for idx, config in enumerate(self.validation_configs):
            if not (config_path := self.root / f"{idx}{VAL_CONFIG_SUFFIX}").exists():
                config.dump(config_path)

    @classmethod
    def load(cls, path: Path) -> "Report":
        meta_config = MetaConfig.load(path / META_CONFIG_FNAME)
        repair_result = RepairResult.load(path)
        analysis_result = RepairAnalysisResults.try_load(path)
        validation_result = ValidationResults.try_load(path)
        p_validation_configs = list(
            filter(lambda p: p.name.endswith(VAL_CONFIG_SUFFIX), path.iterdir())
        )
        p_validation_configs.sort(key=lambda p: int(p.name[: len(VAL_CONFIG_SUFFIX)]))
        validation_configs = list(map(ValidationConfig.load, p_validation_configs))
        return Report(
            path,
            meta_config,
            repair_result,
            analysis_result,
            validation_result,
            validation_configs,
        )
