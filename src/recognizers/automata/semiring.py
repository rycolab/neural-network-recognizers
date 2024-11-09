from collections.abc import Callable
from typing import Any, Generic, TypeVar

import torch

T = TypeVar('T')

class Semiring(Generic[T]):

    def add(self, a: T, b: T) -> T:
        raise NotImplementedError

    def add_in_place(self, a: T, b: T) -> None:
        raise NotImplementedError

    def add_one_in_place(self, a: T) -> None:
        self.add_in_place(a, self.ones((), a.dtype, a.device))

    def sum(self, a: T, dims: tuple[int, ...]) -> T:
        raise NotImplementedError

    def multiply(self, a: T, b: T) -> T:
        raise NotImplementedError

    def star(self, a: T) -> T:
        raise NotImplementedError

    def zeros(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> T:
        raise NotImplementedError

    def ones(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> T:
        raise NotImplementedError

    def transform_tensors(self, a: T, f: Callable[[torch.Tensor], torch.Tensor]) -> T:
        raise NotImplementedError

    def set_index(self, a: T, index: Any, b: T) -> None:
        raise NotImplementedError

    def equal(self, a: T, b: T) -> bool:
        raise NotImplementedError

class TensorSemiring(Semiring[torch.Tensor]):

    def transform_tensors(self, a: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]):
        return f(a)

    def set_index(self, a: torch.Tensor, index: Any, b: torch.Tensor) -> None:
        a[index] = b

    def equal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return torch.allclose(a, b)
