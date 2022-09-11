import itertools
from typing import Callable, Iterable, Iterator, List, Tuple, TypeVar

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
