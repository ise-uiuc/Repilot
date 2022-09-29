import subprocess
import multiprocessing as mp
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TypeVar
from realm.utils import chunked

D4J_PATH = Path('/home/yuxiang/Developer/defects4j')
assert D4J_PATH.exists()

Metadata = Dict[str, List[Dict[str, str]]]

T = TypeVar('T')


def get_all_bugs() -> Metadata:
    def impl():
        proj_dir = D4J_PATH / 'framework' / 'projects'
        for path_i in proj_dir.iterdir():
            if not path_i.is_dir():
                continue
            for path_j in path_i.iterdir():
                if path_j.name == 'active-bugs.csv':
                    with open(path_j) as f:
                        dataset = csv.reader(f)
                        keys = next(dataset)
                        kv_list = (zip(keys, values) for values in dataset)
                        bugs = [{k: v for k, v in kv} for kv in kv_list]
                        yield path_i.name, bugs
    return {proj: bugs for proj, bugs in impl()}


# print(get_all_bugs()['Lang'][0])

ROOT = Path('/home/yuxiang/Developer/d4j-checkout')


def get_checkout_meta(proj: str, bug: Dict[str, str]) -> Dict:
    path = ROOT / f'{proj}-{bug["bug.id"]}'
    bug_id = bug['bug.id']
    return {
        'proj': proj,
        'bug_id': bug_id,
        'buggy_commit': bug['revision.id.buggy'],
        'url': bug['report.url'],
        'fixed_commit': bug['revision.id.fixed'],
        'path': str(path.absolute()),
        'cmd': ['defects4j', 'checkout', '-p', proj, '-v', f'{bug_id}f', '-w', str(path.absolute())]
    }


def get_all_checkout_meta(bugs: Metadata) -> List[Dict[str, str]]:
    return [get_checkout_meta(proj, bug)
            for proj, proj_bugs in bugs.items()
            for bug in proj_bugs]


all_bugs = get_all_bugs()
data = get_all_checkout_meta(all_bugs)


def run_process(*cmd: str):
    subprocess.run(cmd)


BATCH_SIZE = 64
if __name__ == '__main__':
    for d_sub in chunked(BATCH_SIZE, data):
        processes: List[mp.Process] = []
        for d in d_sub:
            p = mp.Process(target=run_process, args=d['cmd'])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
