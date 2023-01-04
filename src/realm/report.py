import difflib
import io
import json
from pathlib import Path
from typing import NamedTuple

import git

from .config import MetaConfig, SynthesisConfig
from .d4j import Defects4J
from .generation_defs import (
    PrunedHalfway,
    SynthesisResultBatch,
    SynthesisSuccessful,
    Unfinished,
)


class TaggedResult(NamedTuple):
    synthesis_result_batch: SynthesisResultBatch
    buggy_file_path: Path
    is_dumpped: bool


RepairResult = dict[str, dict[tuple[int, int], list[TaggedResult]]]


class Reporter:
    def __init__(
        self,
        root: Path,
        config: MetaConfig,
        result_dict: RepairResult,
        synthesis_times: int,
    ) -> None:
        self.root = root
        assert self.root.exists()
        self.config = config
        self.result_dict = result_dict
        self.synthesis_times = synthesis_times

    @staticmethod
    def create(root: Path, config: MetaConfig) -> "Reporter":
        root.mkdir()
        print(f"Metadata will be saved in {root}")
        reporter = Reporter(root, config, {}, 0)
        reporter.dump_metadata()
        return reporter

    @staticmethod
    def load(root: Path):
        raise NotImplementedError

    def dump_metadata(self):
        with open(self.root / "meta_config.json", "w") as f:
            json.dump(self.config.to_json(), f, indent=2)
        with open(self.root / "gen_meta.txt", "w") as f:
            log_git_repo(f, "Repair tool", Path("."))
            log_git_repo(f, "Defects4J", self.config.d4j_home)
            log_git_repo(f, "Language server", self.config.jdt_ls_repo)
            f.write(f"Defects4J checkout path: {self.config.d4j_checkout_root}\n")

    def dump_synthesis_config(self, config: SynthesisConfig):
        with open(
            self.root / f"synthesis_config_{self.synthesis_times}.json", "w"
        ) as f:
            json.dump(config.to_json(), f, indent=2)

    def report_timeout_error(self):
        path = self.root / "timeout.times"
        if path.exists():
            with open(path) as f:
                times = int(f.read().strip()) + 1
        else:
            times = 1
        with open(path, "w") as f:
            f.write(str(times))

    def add(
        self,
        bug_id: str,
        hunk_id: tuple[int, int],
        result: SynthesisResultBatch,
        buggy_file_path: Path,
    ) -> None:
        hunk_dict = self.result_dict.setdefault(bug_id, {})
        tagged_results = hunk_dict.setdefault(hunk_id, [])
        tagged_results.append(TaggedResult(result, buggy_file_path, False))

    def save(self):
        for bug_id, hunk_dict in self.result_dict.items():
            proj, id_str = Defects4J.split_bug_id(bug_id)
            for _, tagged_results in hunk_dict.items():
                idx = 0
                for tagged_result in tagged_results:
                    for result in tagged_result.synthesis_result_batch.results:
                        idx += 1
                        save_dir = self.root / proj / id_str / str(idx)
                        save_dir.mkdir(exist_ok=True, parents=True)
                        if isinstance(result, Unfinished):
                            (save_dir / str(Unfinished)).touch(exist_ok=True)
                        elif isinstance(result, PrunedHalfway):
                            (save_dir / str(PrunedHalfway)).touch(exist_ok=True)
                        else:
                            assert isinstance(result, SynthesisSuccessful)
                            debug_dir = save_dir / "debug"
                            debug_dir.mkdir(exist_ok=True, parents=False)
                            buggy_file_path = tagged_result.buggy_file_path
                            with open(buggy_file_path) as f:
                                buggy_file_lines = f.readlines()
                                assert isinstance(result, SynthesisSuccessful)
                                unified_diff = difflib.unified_diff(
                                    buggy_file_lines,
                                    result.patch.content.splitlines(keepends=True),
                                    fromfile="bug",
                                    tofile="patch",
                                )
                            with open(
                                debug_dir / buggy_file_path.with_suffix(".hunk").name,
                                "w",
                            ) as f:
                                f.write(result.hunk)
                            with open(
                                debug_dir / buggy_file_path.with_suffix(".diff").name,
                                "w",
                            ) as f:
                                f.writelines(unified_diff)


RULE = "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"


def log_git_repo(f: io.TextIOBase, tag: str, repo_path: Path, new_line: bool = True):
    repo = git.Repo(repo_path)
    f.writelines(
        [
            f"[{tag}] Git hash: {repo.head.object.hexsha}\n",
            f"[{tag}] Git status: ",
            RULE,
            repo.git.status(),
            RULE,
            f"[{tag}] Git diff:",
            repo.git.diff(),
            RULE,
        ]
    )
    if new_line:
        f.write("\n")
