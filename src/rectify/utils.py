import difflib
import io
import itertools
import json
import os
import pickle
import re
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, TypeVar, cast

import git
import torch
from unidiff.patch import Line

T = TypeVar("T")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INCODER = os.getenv("INCODER") is not None


def take_while(
    pred: Callable[[T], bool], iterable: Iterable[T]
) -> tuple[list[T], Iterator[T]]:
    iterator = iter(iterable)
    elements: list[T] = []
    return_iter = iterator
    for elem in iterator:
        if pred(elem):
            elements.append(elem)
        else:
            return_iter = itertools.chain([elem], iterator)
            break
    return (elements, return_iter)


def take_while_two(
    pred_first: Callable[[T], bool], pred: Callable[[T, T], bool], iterable: Iterable[T]
) -> tuple[list[T], Iterator[T]]:
    """take_while that looks at two elements at a time"""
    iterator = iter(iterable)
    try:
        first_elem = next(iterator)
        if not pred_first(first_elem):
            return [], itertools.chain([first_elem], iterator)
    except StopIteration:
        return [], iter([])
    elements: list[T] = [first_elem]
    return_iter = iterator
    for elem in iterator:
        if pred(first_elem, elem):
            elements.append(elem)
            first_elem = elem
        else:
            return_iter = itertools.chain([elem], iterator)
            break
    return (elements, return_iter)


COUNTER = itertools.count(0)


def line_consecutive(lhs: Line, rhs: Line) -> bool:
    if lhs.is_added and rhs.is_added:
        return lhs.target_line_no + 1 == rhs.target_line_no
    if lhs.is_removed and rhs.is_removed:
        return lhs.source_line_no + 1 == rhs.source_line_no
    return False


def chunked(n: int, data: Iterable[T]) -> Iterator[list[T]]:
    data_iter = iter(data)

    def take(n: int, data_iter: Iterator[T]) -> Iterable[T]:
        for _ in range(n):
            try:
                yield next(data_iter)
            except StopIteration:
                return

    while len(result := list(take(n, data_iter))) > 0:
        yield result


def load_and_cache_data(path: Path, default_data: Callable[[], T]) -> T:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        data = default_data()
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return data


# TODO: construct a Trie for all the vocabulary
class TrieNode:
    # This class represents a single node in a Trie. It has a dictionary of children
    # nodes and a flag to indicate if it is the end of a word
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    # This method takes a word and inserts it into the Trie
    # by creating a new TrieNode for each character in the word
    # and setting the is_end_of_word flag for the last TrieNode to True
    def insert(self, word: str) -> None:
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_end_of_word = True

    def is_prefix_of(self, word: str) -> bool:
        """Returns if there exists a string in the trie that is a prefix of `word`"""
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return curr.is_end_of_word
            curr = curr.children[ch]
        return curr.is_end_of_word


# This function finds the common prefix shared by all the strings in the given list
# by inserting each string into a Trie and traversing the Trie starting from the root
# If a TrieNode is reached that is the end of a word and has more than one child
# then we return the common prefix up to that point
# Otherwise, if the current TrieNode has only one child we continue to traverse
# the Trie using the child node and add the child's character to the common prefix
# If we reach a TrieNode that has 0 or more than 1 children then we return the common
# prefix (if it's not empty) or None if the prefix is empty
def common_prefix(strings: list[str]) -> Optional[str]:
    trie = Trie()
    for s in strings:
        if s == "":
            return None
        trie.insert(s)

    common = ""
    curr = trie.root
    while curr:
        if curr.is_end_of_word:
            return common if len(common) > 0 else None
        if len(curr.children) == 1:
            ch = list(curr.children.keys())[0]
            common += ch
            curr = curr.children[ch]
        else:
            break

    return common if len(common) > 0 else None


class Meaningless:
    def __call__(self, *args, **kwds):
        return self

    def __getattribute__(self, __name: str):
        return self


class ConnectionWrapper:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def send(self, obj: Any) -> None:
        self.conn.send(obj)

    def send_call(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.conn.send((method, args, kwargs))

    def recv(self) -> Any:
        return self.conn.recv()


class IORetrospective(Protocol):
    @classmethod
    def load(cls: type[T], path: Path) -> T:
        ...

    def dump(self, path: Path):
        ...


class JsonSerializable(IORetrospective, Protocol):
    def save_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def to_json(self) -> Any:
        ...

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        return cls.from_json_file(path)  # type: ignore # noqa

    def dump(self, path: Path):
        self.save_json(path)

    @classmethod
    def from_json_file(cls: type[T], path: Path) -> T:
        with open(path) as f:
            d = json.load(f)
        return cls.from_json(d)  # type: ignore # noqa

    @classmethod
    def from_json(cls: type[T], d: Any) -> T:
        ...


class JsonSpecificDirectoryDumpable(JsonSerializable, Protocol):
    @classmethod
    def name(cls) -> str:
        ...

    @classmethod
    def json_save_path(cls, path: Path) -> Path:
        return path / cls.name()

    def dump(self, path: Path):
        self.save_json(self.json_save_path(path))

    @classmethod
    def try_load(cls: type[T], path: Path) -> T | None:
        cls_casted = cast(JsonSpecificDirectoryDumpable, cls)
        if not (cls_casted.json_save_path(path)).exists():
            return None
        return cast(T, cls_casted.load(path))

    @classmethod
    def load(cls: type[T], path: Path) -> T:
        cls_casted = cast(JsonSpecificDirectoryDumpable, cls)
        path = cls_casted.json_save_path(path)
        assert path.exists()
        return cast(T, cls_casted.from_json_file(path))


DIFF_RULE = (
    "\n******************************************************************************\n"
)

RULE = (
    "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n"
)

HUNK_RULE = (
    "\n==============================================================================\n"
)

# class BuggyHunkPlaceholder:
#     def __repr__(self) -> str:
#         return "<BUGGY_HUNK_PLACEHOLDER>"

BUGGY_HUNK_PLACEHOLDER = "<BUGGY_HUNK_PLACEHOLDER>"


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
            repo.git.diff(repo.head.commit),
            repo.git.diff(None),
            RULE,
        ]
    )
    if new_line:
        f.write("\n")


U = TypeVar("U")


def bind_optional(x: T | None, f: Callable[[T], U]) -> U | None:
    return None if x is None else f(x)


def disable_tokenizer_parallel():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def select(true: U, false: U, is_true: Callable[[T], bool], value: T) -> U:
    return true if is_true(value) else false


binary: Callable[[Callable[[T], bool], T], int] = partial(select, 1, 0)  # type: ignore # fmt: skip

binary_optional: Callable[[T | None], int] = partial(binary, lambda t: t is not None)  # type: ignore # fmt: skip

binary_bool: Callable[[bool], int] = partial(binary, lambda x: x)  # type: ignore # fmt: skip


def stride(init: float, step: float, times: int) -> Iterable[float]:
    for _ in range(times):
        yield init
        init += step


def longest_common_prefix(strings: list[str]) -> str:
    """Leetcode benchmark fastest way"""
    prefix = ""
    index = 0
    while True:
        try:
            c = strings[0][index]
            if all(string[index] == c for string in strings):
                prefix += c
                index += 1
            else:
                break
        except IndexError:
            break
    return prefix


def remove_whitespace(s: str) -> str:
    return s.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")


def remove_java_comments(text: str) -> str:
    text = re.sub(re.compile("//.*"), "", text)
    text = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", text)
    return text


def diff(lhs: str, rhs: str, lhs_msg: str, rhs_msg: str) -> str:
    unified_diff = difflib.unified_diff(
        lhs.splitlines(), rhs.splitlines(), lhs_msg, rhs_msg
    )
    return "\n".join(unified_diff)


def average_and_adjust(num: int, n: int):
    average = num // n
    parts = [average] * n
    parts[-1] += num - (n * average)
    assert sum(parts) == num
    return parts


def ceil(num: int, n: int):
    return (num + n - 1) // n


def safe_div(dividend: int | float, divisor: int | float) -> float:
    return dividend / divisor if divisor != 0 else 0.0
