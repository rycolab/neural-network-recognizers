import torch

from .semiring import TensorSemiring

class BooleanSemiring(TensorSemiring):

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a | b

    def add_in_place(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a |= b

    def add_one_in_place(self, a: torch.Tensor) -> None:
        a[...] = True

    def sum(self, a: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        return torch.any(a, dim=dims)

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a & b

    def star(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def zeros(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.zeros(size, dtype=dtype, device=device)

    def ones(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.ones(size, dtype=dtype, device=device)

    def equal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return torch.equal(a, b)
