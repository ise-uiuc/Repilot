import csv
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TypeVar

from repilot.config import MetaConfig
from repilot.utils import chunked

META_CONFIG = MetaConfig.from_json_file(Path("meta_config.json"))
D4J_PATH = META_CONFIG.d4j_home
assert D4J_PATH.exists(), D4J_PATH

Metadata = Dict[str, List[Dict[str, str]]]

T = TypeVar("T")


def get_all_bugs() -> Metadata:
    def impl():
        proj_dir = D4J_PATH / "framework" / "projects"
        for path_i in proj_dir.iterdir():
            if not path_i.is_dir():
                continue
            for path_j in path_i.iterdir():
                if path_j.name == "active-bugs.csv":
                    with open(path_j) as f:
                        dataset = csv.reader(f)
                        keys = next(dataset)
                        kv_list = (zip(keys, values) for values in dataset)
                        bugs = [{k: v for k, v in kv} for kv in kv_list]
                        yield path_i.name, bugs

    return {proj: bugs for proj, bugs in impl()}


# print(get_all_bugs()['Lang'][0])

ROOT = META_CONFIG.d4j_checkout_root
ROOT.mkdir(exist_ok=True)


def get_checkout_meta(proj: str, bug: Dict[str, str]) -> Dict:
    path = ROOT / f'{proj}-{bug["bug.id"]}'
    bug_id = bug["bug.id"]
    d4j_exec = META_CONFIG.d4j_home / "framework" / "bin" / "defects4j"
    return {
        "proj": proj,
        "bug_id": bug_id,
        "buggy_commit": bug["revision.id.buggy"],
        "url": bug["report.url"],
        "fixed_commit": bug["revision.id.fixed"],
        "path": str(path.absolute()),
        "cmd": [
            str(d4j_exec.absolute()),
            "checkout",
            "-p",
            proj,
            "-v",
            f"{bug_id}f",
            "-w",
            str(path.absolute()),
        ],
    }


def get_all_checkout_meta(bugs: Metadata) -> List[Dict[str, str]]:
    return [
        get_checkout_meta(proj, bug)
        for proj, proj_bugs in bugs.items()
        for bug in proj_bugs
    ]


all_bugs = get_all_bugs()
data = get_all_checkout_meta(all_bugs)


def run_process(*cmd: str):
    subprocess.run(cmd)


BATCH_SIZE = 64
if __name__ == "__main__":
    for d_sub in chunked(BATCH_SIZE, data):
        processes: List[mp.Process] = []
        for d in d_sub:
            p = mp.Process(target=run_process, args=d["cmd"])
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
