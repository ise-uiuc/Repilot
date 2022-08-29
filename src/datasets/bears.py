import os
import git
from typing import Callable, List, Tuple, Optional
from dataset import RepoBasedBug, RepairBenchmark, ReproduceError
from parse_quixbugs import get_unified_diff
import utils
from tqdm import tqdm
import json
import utils
import subprocess
import argparse
from unidiff import PatchSet, Hunk

GIT_REPO = 'https://github.com/UniverseFly/bears-benchmark'


class BearsBug(RepoBasedBug):
    def __init__(self, metadata: dict, bug_id: str) -> None:
        self.metadata = metadata
        self.bug_id = bug_id
        self._repo: Optional[git.Repo] = None
        assert bug_id.startswith('Bears')

    @property
    def repo(self) -> git.Repo:
        if self._repo is None:
            self._repo = git.Repo(self.working_dir)
        return self._repo

    @property
    def extension(self) -> str:
        return '.java'

    @property
    def repo_data(self) -> dict:
        return self.metadata['bugs'][self.bug_id]

    @property
    def working_dir(self):
        return self.repo_data['checkout_path']

    @property
    def id(self) -> str:
        return self.bug_id

    def get_patch_set(self) -> PatchSet:
        return PatchSet(self.repo_data['diff'])

    def validate(self, timeout: float, clean_test: bool):
        try:
            cmd = f'python scripts/compile_bug.py --bugId {self.bug_id} --workspace {self.metadata["workspace"]}\n'
            cmd += f'python scripts/run_tests_bug.py --bugId {self.bug_id} --workspace {self.metadata["workspace"]}'
            result = subprocess.run(
                cmd,
                cwd=self.metadata['repo'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(e.stdout, e.stderr)
        if not result.returncode == 0:
            raise ReproduceError(result.stdout, result.stderr)

    def checkout_buggy(self):
        self.checkout_fixed()
        branch = self.repo.create_head(
            f"buggy-{self.repo_data['branch']}", 'HEAD^^', force=True)
        self.repo.head.reference = branch
        self.repo.head.reset(index=True, working_tree=True)

    def checkout_fixed(self):
        branch = self.repo.create_head(
            self.repo_data['branch'], self.repo_data['branch'], force=True)
        self.repo.head.reference = branch
        self.repo.head.reset(index=True, working_tree=True)


def checkout(repo: str, workspace: str) -> 'Bears':
    repo = os.path.abspath(repo)
    workspace = os.path.abspath(workspace)
    if not os.path.exists(repo):
        git.Repo.clone_from(GIT_REPO, repo)
    if not os.path.exists(workspace):
        os.makedirs(workspace, exist_ok=True)
        subprocess.run(
            f'python scripts/checkout_all.py --workspace {workspace}',
            cwd=repo,
            shell=True
        )
    metadata: dict = {
        'repo': repo,
        'workspace': workspace,
        'bugs': {},
    }
    with open(os.path.join(repo, 'docs', 'data', 'bears-bugs.json')) as f:
        repo_metadata = json.load(f)
    with open(os.path.join(repo, 'scripts', 'data', 'bug_id_and_branch.json')) as f:
        repo_branches = json.load(f)
    assert len(repo_metadata) == len(repo_branches)
    for branch_info, bug_info in zip(tqdm(repo_branches), repo_metadata):
        bug_id = branch_info['bugId']
        if bug_id not in metadata:
            metadata['bugs'][bug_id] = {}
        metadata['bugs'][bug_id]['branch'] = branch_info['bugBranch']
        metadata['bugs'][bug_id]['diff'] = bug_info['diff']
        metadata['bugs'][bug_id]['checkout_path'] = os.path.join(
            workspace, bug_id)
    return Bears(metadata)


class Bears(RepairBenchmark[BearsBug]):
    def __init__(self, metadata: dict) -> None:
        self._logger: Optional[utils.Logger] = None
        self.checkout_meta = metadata

    def with_logger_tag(self, tag: str) -> 'Bears':
        if tag != '':
            tag = '-' + tag
        self._logger = utils.Logger(f'Bears{tag}')
        return self

    @property
    def logger(self) -> Optional[utils.Logger]:
        return self._logger

    @classmethod
    def load_metadata(cls, metadata: dict) -> 'Bears':
        return Bears(metadata)

    @property
    def metadata(self) -> dict:
        return self.checkout_meta

    def get_all_bugs(self) -> List[BearsBug]:
        return [BearsBug(self.checkout_meta, id) for id in self.checkout_meta['bugs']]
