import torch

from .semiring import TensorSemiring

class CountingSemiring(TensorSemiring):

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.fft_n = 2 * size - 1

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def add_in_place(self, a: torch.Tensor, b: torch.Tensor) -> None:
        a.add_(b)

    def add_one_in_place(self, a: torch.Tensor) -> None:
        a[..., 0] += 1

    def sum(self, a: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
        if dims:
            return torch.sum(a, dim=dims)
        else:
            return a

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # See https://pytorch.org/audio/stable/_modules/torchaudio/functional/functional.html#fftconvolve
        return torch.fft.irfft(
            torch.fft.rfft(a, n=self.fft_n) * torch.fft.rfft(b, n=self.fft_n),
            n=self.fft_n
        )[..., :self.size]

    def star(self, a: torch.Tensor) -> torch.Tensor:
        a_star = torch.empty_like(a)
        # The `torch.flip()` function always creates a copy instead of a view,
        # so we maintain a flipped copy of a_star to avoid flipping it for
        # every i.
        flipped_a_star = torch.empty_like(a)
        a0_star = 1 / (1 - a[..., 0])
        size = a.size(-1)
        for i in range(size):
            c = torch.sum(a[..., 1:i+1] * flipped_a_star[..., size-i:], dim=-1)
            if i == 0:
                c += 1
            a_star[..., i] = flipped_a_star[..., size-1-i] = a0_star * c
        return a_star

    def zeros(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        return torch.zeros(size + (self.size,), dtype=dtype, device=device)

    def ones(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        result = torch.zeros(size + (self.size,), dtype=dtype, device=device)
        result[..., 0] = 1
        return result
