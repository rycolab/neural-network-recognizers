from typing import Any

import torch
from torch.testing import assert_close

from recognizers.automata.counting_semiring import CountingSemiring

def hot(n: int, i: Any, value: float=1) -> torch.Tensor:
    result = torch.zeros(n)
    result[i] = value
    return result

def reference_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 1 and b.ndim == 1 and a.size() == b.size()
    size = a.size(0)
    c = torch.zeros(size)
    for j in range(size):
        c[j] = sum(a[n] * b[j - n] for n in range(0, j+1))
    return c

def real_star(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 - x)

def reference_star(a: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 1
    size = a.size(0)
    r = torch.zeros(size)
    for i in range(size):
        c = int(i == 0) + sum(a[n] * r[i - n] for n in range(1, i+1))
        r[i] = real_star(a[0]) * c
    return r

def test_add_in_place() -> None:
    semiring = CountingSemiring(10)
    a = hot(10, 0)
    semiring.add_in_place(a, hot(10, 1))
    assert_close(a, hot(10, [0, 1]))

def test_random_add_in_place() -> None:
    generator = torch.manual_seed(123)
    semiring = CountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        b = torch.rand(10, generator=generator)
        c = a.clone()
        semiring.add_in_place(c, b)
        assert_close(c, a + b)

def test_multiply() -> None:
    semiring = CountingSemiring(10)
    assert_close(semiring.multiply(hot(10, 0), hot(10, 1)), hot(10, 1))
    assert_close(semiring.multiply(hot(10, 0), hot(10, 0)), hot(10, 0))
    assert_close(semiring.multiply(hot(10, 1), hot(10, 1)), hot(10, 2))
    assert_close(semiring.multiply(hot(10, 3), hot(10, 4)), hot(10, 7))
    assert_close(semiring.multiply(hot(10, 4), hot(10, 7)), torch.zeros(10))
    assert_close(semiring.multiply(torch.zeros(10), torch.zeros(10)), torch.zeros(10))
    assert_close(semiring.multiply(hot(10, 0, 0.5), hot(10, 0, 0.5)), hot(10, 0, 0.25))
    assert_close(semiring.multiply(hot(10, 2, 0.5), hot(10, 4, 0.5)), hot(10, 6, 0.25))

def test_random_multiply() -> None:
    generator = torch.manual_seed(123)
    semiring = CountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        b = torch.rand(10, generator=generator)
        assert_close(semiring.multiply(a, b), reference_multiply(a, b))

def test_multi_dim_multiply() -> None:
    generator = torch.manual_seed(123)
    semiring = CountingSemiring(10)
    a = torch.rand((29, 1, 7, 10), generator=generator)
    b = torch.rand((1, 13, 7, 10), generator=generator)
    r = semiring.multiply(a, b)
    assert r.size() == (29, 13, 7, 10)
    expected_r = torch.empty(29, 13, 7, 10)
    for i in range(29):
        for j in range(13):
            for k in range(7):
                expected_r[i, j, k] = reference_multiply(a[i, 0, k], b[0, j, k])
    assert_close(r, expected_r)

def test_reference_star_is_mathematically_correct() -> None:
    generator = torch.manual_seed(123)
    one = hot(10, 0)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        a_star = reference_star(a)
        assert_close(a_star, reference_multiply(a, a_star) + one)

def test_star() -> None:
    semiring = CountingSemiring(10)
    for a in [
        hot(10, 0, 0.5),
        hot(10, 1, 0.5),
        hot(10, [2, 4], 0.2)
    ]:
        assert_close(semiring.star(a), reference_star(a))

def test_random_star() -> None:
    generator = torch.manual_seed(123)
    semiring = CountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        assert_close(semiring.star(a), reference_star(a))
