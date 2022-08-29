import os
from typing import List, Optional
import utils
import subprocess
from unidiff import PatchSet
from dataset import RepoBasedBug, RepairBenchmark, ReproduceError
import git
from tqdm import tqdm


GIT_REPO = 'https://github.com/bugs-dot-jar/bugs-dot-jar'
PATCH_PATH = os.path.join('.bugs-dot-jar', 'developer-patch.diff')
PROJECTS = ["accumulo", "camel", "commons-math", "flink",
            "jackrabbit-oak", "logging-log4j2", "maven", "wicket"]


class BugsDotJarBug(RepoBasedBug):
    def __init__(self, metadata: dict, project: str, branch: str) -> None:
        self.metadata = metadata
        self.project = project
        self.branch = branch
        self.repo = git.Repo(self.working_dir)

    @property
    def extension(self) -> str:
        return '.java'

    @property
    def repo_data(self) -> dict:
        return self.metadata[self.project]['branches'][self.branch]

    @property
    def working_dir(self):
        return self.metadata[self.project]['checkout_path']

    @property
    def id(self) -> str:
        return self.project + ' - ' + self.branch

    def get_patch_set(self) -> PatchSet:
        return PatchSet(self.repo_data['diff'])

    def validate(self, timeout: float, clean_test: bool):
        assert self.repo.working_dir is not None
        try:
            result = subprocess.run(
                'mvn clean test' if clean_test else 'mvn test',
                cwd=self.repo.working_dir,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(e.stdout, e.stderr)
        if result.returncode != 0:
            raise ReproduceError(result.stdout, result.stderr)

    def checkout_buggy(self):
        branch = self.repo.create_head(
            self.branch, self.branch, force=True)
        self.repo.head.reference = branch
        self.repo.head.reset(index=True, working_tree=True)

    def checkout_fixed(self):
        self.checkout_buggy()
        assert self.repo.working_dir is not None
        apply_result = subprocess.run(
            f'git apply --whitespace=fix {PATCH_PATH}',
            cwd=self.repo.working_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        assert apply_result.returncode == 0


def checkout(bugs_dot_jar_repo: str) -> 'BugsDotJar':
    if not os.path.exists(bugs_dot_jar_repo):
        git.Repo.clone_from(GIT_REPO, bugs_dot_jar_repo)
    metadata: dict = {}
    for project in tqdm(PROJECTS):
        project_path = os.path.abspath(
            os.path.join(bugs_dot_jar_repo, project))
        metadata[project] = {'checkout_path': project_path, 'branches': {}}
        current_repo = git.Repo(project_path)
        branches = subprocess.run(
            "git branch -a | grep bugs-dot-jar_",
            shell=True,
            cwd=project_path,
            capture_output=True,
            text=True,
        ).stdout.strip().split('\n')
        branches = list(
            set(branch[branch.find('bugs-dot-jar_'):] for branch in branches))
        # self.metadata.extend([(project, branch) for branch in branches])
        for branch_name in tqdm(branches, leave=False):
            assert branch_name.startswith('bugs-dot-jar')
            branch = current_repo.create_head(
                branch_name, branch_name, force=True)
            current_repo.head.reference = branch  # type: ignore # noqa
            current_repo.head.reset(index=True, working_tree=True)
            metadata[project]['branches'][branch_name] = {}
            current_metadata = metadata[project]['branches'][branch_name]
            with open(os.path.join(project_path, PATCH_PATH), 'r') as f:
                current_metadata['diff'] = f.read()
    return BugsDotJar(metadata)


class BugsDotJar(RepairBenchmark[BugsDotJarBug]):
    def __init__(self, metadata: dict) -> None:
        self._logger: Optional[utils.Logger] = None
        self.checkout_meta = metadata
    
    def with_logger_tag(self, tag: str) -> 'BugsDotJar':
        if tag != '':
            tag = '-' + tag
        self._logger = utils.Logger(f'BugsDotJar{tag}')
        return self

    @property
    def logger(self) -> Optional[utils.Logger]:
        return self._logger

    @classmethod
    def load_metadata(cls, metadata: dict) -> 'BugsDotJar':
        return BugsDotJar(metadata)

    @property
    def metadata(self) -> dict:
        return self.checkout_meta

    def get_all_bugs(self) -> List[BugsDotJarBug]:
        return [BugsDotJarBug(self.checkout_meta, project, branch) for project in self.checkout_meta for branch in self.checkout_meta[project]['branches']]
