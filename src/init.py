import subprocess
import multiprocessing as mp
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TypeVar

D4J_PATH = Path('/Users/nole/Developer/defects4j')
assert D4J_PATH.exists()

Metadata = Dict[str, List[Dict[str, str]]]

T = TypeVar('T')


def chunked(n: int, data: Iterable[T]) -> Iterator[List[T]]:
    data_iter = iter(data)

    def take(n: int, data_iter: Iterator[T]) -> Iterable[T]:
        for _ in range(n):
            try:
                yield next(data_iter)
            except StopIteration:
                return
    while len(result := list(take(n, data_iter))) > 0:
        yield result


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

ROOT = Path('/Users/nole/Developer/d4j-checkout')


def get_checkout_meta(proj: str, bug: Dict[str, str], fixed: bool) -> Dict:
    path = ROOT / f'{proj}-{bug["bug.id"]}-{"fixed" if fixed else "buggy"}'
    bug_id = bug['bug.id']
    return {
        'proj': proj,
        'bug_id': bug_id,
        'buggy_commit': bug['revision.id.buggy'],
        'url': bug['report.url'],
        'fixed_commit': bug['revision.id.fixed'],
        'fixed': fixed,
        'path': str(path.absolute()),
        'cmd': ['defects4j', 'checkout', '-p', proj, '-v', f'{bug_id}{"f" if fixed else "b"}', '-w', str(path.absolute())]
    }


def get_all_checkout_meta(bugs: Metadata) -> List[Dict[str, str]]:
    return [y for x in ((get_checkout_meta(proj, bug, False),
                         get_checkout_meta(proj, bug, True))
                        for proj, proj_bugs in bugs.items()
                        for bug in proj_bugs) for y in x]


data = get_all_checkout_meta(get_all_bugs())


def checkout(*cmd: str):
    subprocess.run(cmd)


BATCH_SIZE = 10
if __name__ == '__main__':
    for d_sub in chunked(BATCH_SIZE, data):
        processes: List[mp.Process] = []
        for d in d_sub:
            p = mp.Process(target=checkout, args=d['cmd'])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# f'defects4j checkout -p {proj} -v {bug_id}f -w {root / f"{proj}-{bug_id}-{}"}'
