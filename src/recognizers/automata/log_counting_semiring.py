import math

import torch

from .semiring import TensorSemiring

class LogCountingSemiring(TensorSemiring):

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(a, b)

    def add_in_place(self, a: torch.Tensor, b: torch.Tensor) -> None:
        torch.logaddexp(a, b, out=a)

    def add_one_in_place(self, a: torch.Tensor) -> None:
        out = a[..., 0]
        torch.logaddexp(out, a.new_zeros(()), out=out)

    def sum(self, a: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        if dims:
            return torch.logsumexp(a, dim=dims)
        else:
            return a

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        A = a.unsqueeze(-2)
        r = torch.arange(b.size(-1), device=b.device)
        i = r[:, None] - r[None, :]
        del r
        B = b[..., i]
        B[..., i < 0] = -math.inf
        del i
        return torch.logsumexp(A + B, dim=-1)

    def star(self, a: torch.Tensor) -> torch.Tensor:
        a_star = torch.empty_like(a)
        # The `torch.flip()` function always creates a copy instead of a view,
        # so we maintain a flipped copy of a_star to avoid flipping it for
        # every i.
        flipped_a_star = torch.empty_like(a)
        a0_star = log_star(a[..., 0])
        size = a.size(-1)
        for i in range(size):
            c = torch.logsumexp(a[..., 1:i+1] + flipped_a_star[..., size-i:], dim=-1)
            if i == 0:
                torch.logaddexp(c, a.new_zeros(()), out=c)
            a_star[..., i] = flipped_a_star[..., size-1-i] = a0_star + c
        return a_star

    def zeros(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        return torch.full(size + (self.size,), -math.inf, dtype=dtype, device=device)

    def ones(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        result = torch.full(size + (self.size,), -math.inf, dtype=dtype, device=device)
        result[..., 0] = 0
        return result

def log_star(a: torch.Tensor) -> torch.Tensor:
    return -torch.log(1 - torch.exp(a))
