import itertools
from typing import Callable, Iterable, Iterator, List, Tuple, TypeVar
from unidiff.patch import Line

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
