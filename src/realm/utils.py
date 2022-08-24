from typing import Any, Callable, Dict
from itertools import count


def compose(f: Callable, g: Callable) -> Callable:
    return lambda *args, **kwargs: f(g(*args, **kwargs))
