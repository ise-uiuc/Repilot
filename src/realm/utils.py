import itertools
import pickle
import json
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Optional,
    Any,
    Protocol,
    Type,
    ClassVar,
    cast,
)
from unidiff.patch import Line
from pathlib import Path
from multiprocessing.connection import Connection
from dataclasses import dataclass
import torch

T = TypeVar("T")

DEVICE = "cuda" if torch.cuda.is_available else "cpu"


def take_while(
    pred: Callable[[T], bool], iterable: Iterable[T]
) -> Tuple[List[T], Iterator[T]]:
    iterator = iter(iterable)
    elements: List[T] = []
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
) -> Tuple[List[T], Iterator[T]]:
    """take_while that looks at two elements at a time"""
    iterator = iter(iterable)
    try:
        first_elem = next(iterator)
        if not pred_first(first_elem):
            raise StopIteration
    except StopIteration:
        return [], iter([])
    elements: List[T] = [first_elem]
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
def common_prefix(strings: List[str]) -> Optional[str]:
    trie = Trie()
    for s in strings:
        if s == "":
            return None
        trie.insert(s)

    common = ""
    curr = trie.root
    while curr:
        if curr.is_end_of_word and len(curr.children) > 1:
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
    def load(cls: Type[T], path: Path) -> T:
        ...

    def dump(self, path: Path):
        ...


class JsonSerializable(IORetrospective):
    def save_json(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def to_json(self) -> Any:
        ...

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        return cls.from_json_file(path)  # type: ignore # noqa

    def dump(self, path: Path):
        self.save_json(path)

    @classmethod
    def from_json_file(cls: Type[T], path: Path) -> T:
        with open(path) as f:
            d = json.load(f)
        return cls.from_json(d)  # type: ignore # noqa

    @classmethod
    def from_json(cls: Type[T], d: Any) -> T:
        ...
