import itertools
import pickle
from typing import Callable, Iterable, Iterator, List, Tuple, TypeVar, Optional
from unidiff.patch import Line
from pathlib import Path
import json

T = TypeVar('T')


def take_while(pred: Callable[[T], bool], iterable: Iterable[T]) -> Tuple[List[T], Iterator[T]]:
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


def take_while_two(pred_first: Callable[[T], bool], pred: Callable[[T, T], bool], iterable: Iterable[T]) -> Tuple[List[T], Iterator[T]]:
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

def load_and_cache_data(path: Path, default_data: T) -> T:
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(default_data, f)
        return default_data


# TODO: construct a Trie for all the vocabulary
class TrieNode:
    # This class represents a single node in a Trie. It has a dictionary of children
    # nodes and a flag to indicate if it is the end of a word
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
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
        if s == '':
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