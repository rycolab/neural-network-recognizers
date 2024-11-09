from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')

def fixed_point_iteration(
    func: Callable[[T], T],
    equal: Callable[[T, T], bool],
    zero: T
) -> T:
    x = zero
    while True:
        new_x = func(x)
        if equal(new_x, x):
            return new_x
        x = new_x
