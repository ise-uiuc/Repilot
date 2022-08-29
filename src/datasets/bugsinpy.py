import os
from typing import List, Optional
import utils
import utils
import subprocess
from unidiff import PatchSet
from dataset import RepoBasedBug, RepairBenchmark, ReproduceError
import git


GIT_REPO = 'https://github.com/bugs-dot-jar/bugs-dot-jar'
PATCH_PATH = os.path.join('.bugs-dot-jar', 'developer-patch.diff')
PROJECTS = ["PySnooper", "ansible", "black", "cookiecutter", "fastapi", "httpie", "keras", "luigi",
            "matplotlib", "pandas", "sanic", "scrapy", "spacy", "thefuck", "tornado", "tqdm", "youtube-dl"]


class BugsInPyBug(RepoBasedBug):
    def __init__(self, metadata: dict, project: str, id: str) -> None:
        self.metadata = metadata
        self.project = project
        self.bug_id = id
        self._repo: Optional[git.Repo] = None
        assert id.isdigit()
    
    @property
    def repo(self) -> git.Repo:
        if self._repo is None:
            self._repo = git.Repo(self.repo_data['checkout_path'])
        return self._repo
    
    @property
    def extension(self) -> str:
        return '.py'

    @property
    def repo_data(self) -> dict:
        return self.metadata[self.project][self.bug_id]

    @property
    def working_dir(self):
        return self.repo_data['checkout_path']

    @property
    def id(self) -> str:
        return self.project + ' - ' + self.bug_id

    def get_patch_set(self) -> PatchSet:
        return PatchSet(self.repo_data['diff'])

    def validate(self, timeout: float, clean_test: bool):
        # self.repo.git.clean('-xdf')
        env_name = 'py' + self.repo_data['python_version']
        path = self.working_dir
        if not clean_test and os.path.exists(os.path.join(path, 'bugsinpy_compile_flag')):
            command = 'bugsinpy-test'
        else:
            prereq_cmd = '. $(conda info --base)/etc/profile.d/conda.sh\n'
            command = prereq_cmd + f'conda activate {env_name}\n'
            if subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
                command = prereq_cmd + \
                    f'conda create -n {env_name} python={env_name[2:]} -y; conda activate {env_name}\n'
            command += 'bugsinpy-compile; bugsinpy-test'
        try:
            result = subprocess.run(
                command, shell=True, cwd=path, capture_output=True, text=True, timeout=timeout)
            assert result.returncode == 0, (result.stdout, result.stderr)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(e.stdout, e.stderr)
        if os.path.exists(os.path.join(path, 'bugsinpy_fail.txt')):
            raise ReproduceError(result.stdout, result.stderr)

    def checkout_buggy(self):
        branch = self.repo.create_head(
            'buggy', self.repo_data['buggy_commit'], force=True)
        self.repo.head.reference = branch
        self.repo.head.reset(index=True, working_tree=True)

    def checkout_fixed(self):
        branch = self.repo.create_head(
            'fixed', self.repo_data['fixed_commit'], force=True)
        self.repo.head.reference = branch
        self.repo.head.reset(index=True, working_tree=True)


def checkout(repo: str, workspace: str) -> 'BugsInPy':
    if not os.path.exists(repo):
        git.Repo.clone_from(GIT_REPO, repo)
    metadata: dict = {}
    for project in PROJECTS:
        metadata[project] = {}
        project_dir = os.path.join(repo, 'projects', project)
        bug_ids = filter(str.isdigit, os.listdir(
            os.path.join(project_dir, 'bugs')))
        for bug_id in bug_ids:
            project_checkout_path = os.path.abspath(
                os.path.join(workspace, f'{project}-{bug_id}'))
            if not os.path.exists(project_checkout_path):
                subprocess.run(
                    f'bugsinpy-checkout -p {project} -v 0 -i {bug_id} -w {project_checkout_path}', shell=True)
            metadata[project][bug_id] = {}
            current_bug = metadata[project][bug_id]
            current_bug['checkout_path'] = os.path.join(
                project_checkout_path, project)
            with open(os.path.join(project_dir, 'bugs', bug_id, 'bug_patch.txt')) as f:
                current_bug['diff'] = f.read()
            with open(os.path.join(project_dir, 'bugs', bug_id, 'bug.info')) as f:
                for line in f.readlines():
                    if line.startswith('python_version'):
                        first_quote = line.index("\"") + 1
                        second_quote = line.index("\"", first_quote)
                        current_bug['python_version'] = line[first_quote:second_quote]
                    if line.startswith('buggy_commit_id'):
                        first_quote = line.index("\"") + 1
                        second_quote = line.index("\"", first_quote)
                        current_bug['buggy_commit'] = line[first_quote:second_quote]
                    elif line.startswith('fixed_commit_id'):
                        first_quote = line.index("\"") + 1
                        second_quote = line.index("\"", first_quote)
                        current_bug['fixed_commit'] = line[first_quote:second_quote]
                assert 'buggy_commit' in current_bug and 'fixed_commit' in current_bug, f.name
    return BugsInPy(metadata)


class BugsInPy(RepairBenchmark[BugsInPyBug]):
    def __init__(self, metadata: dict) -> None:
        self._logger: Optional[utils.Logger] = None
        self.checkout_meta = metadata
    
    def with_logger_tag(self, tag: str) -> 'BugsInPy':
        if tag != '':
            tag = '-' + tag
        self._logger = utils.Logger(f'BugsInPy{tag}')
        return self

    @property
    def logger(self) -> Optional[utils.Logger]:
        return self._logger

    @classmethod
    def load_metadata(cls, metadata: dict) -> 'BugsInPy':
        return BugsInPy(metadata)

    @property
    def metadata(self) -> dict:
        return self.checkout_meta

    def get_all_bugs(self) -> List[BugsInPyBug]:
        return [BugsInPyBug(self.checkout_meta, project, id) for project in self.checkout_meta for id in self.checkout_meta[project]]
