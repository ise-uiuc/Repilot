from functools import partial
from os import PathLike
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple


class BuggyFile(NamedTuple):
    path: Path
    buggy_lines: List[int]


class Defects4J:
    def __init__(self, metadata: List[dict], buggy_location_path: PathLike) -> None:
        self.metadata = metadata
        self.buggy_location_path = Path(buggy_location_path)
        assert self.buggy_location_path.exists()

    def buggy_files(self, bug: dict) -> List[BuggyFile]:
        with open(self.buggy_location_path / f"{bug['proj']}-{bug['bug_id']}.buggy.lines", 'rb') as f:
            diff_lines = f.readlines()

        def parse_line(line: bytes) -> Optional[Tuple[Path, int]]:
            first = line.index(b'#')
            second = line.index(b'#', first + 1)
            if b'FAULT_OF_OMISSION' in line[second + 1:]:
                return None
            file = Path(bug['path']) / line[:first].decode()
            return file, int(line[first + 1: second].decode()) - 1
        buggy_lines = filter(lambda x : x is not None, map(parse_line, diff_lines))

        buggy_file_lines: Dict[str, List[int]] = {}
        for file, line in buggy_lines: # type: ignore
            if not file.name in buggy_file_lines:
                buggy_file_lines[file.name] = []
            buggy_file_lines[file.name].append(line)
        return list(map(lambda x: BuggyFile(Path(x[0]), x[1]), buggy_file_lines.items()))

    def all_bugs(self) -> list:
        return [(bug['path'], self.buggy_files(bug)) for bug in self.metadata]
