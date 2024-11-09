import torch
from torch.testing import assert_close

from recognizers.automata.log_counting_semiring import LogCountingSemiring

from test_counting_semiring import hot, reference_multiply, reference_star

def test_add_in_place() -> None:
    semiring = LogCountingSemiring(10)
    a = torch.log(hot(10, 0))
    semiring.add_in_place(a, torch.log(hot(10, 1)))
    assert_close(a, torch.log(hot(10, [0, 1])))

def test_random_add_in_place() -> None:
    generator = torch.manual_seed(123)
    semiring = LogCountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        b = torch.rand(10, generator=generator)
        c = torch.log(a)
        semiring.add_in_place(c, torch.log(b))
        assert_close(c, torch.log(a + b))

def test_multiply() -> None:
    semiring = LogCountingSemiring(10)
    assert_close(semiring.multiply(torch.log(hot(10, 0)), torch.log(hot(10, 1))), torch.log(hot(10, 1)))
    assert_close(semiring.multiply(torch.log(hot(10, 0)), torch.log(hot(10, 0))), torch.log(hot(10, 0)))
    assert_close(semiring.multiply(torch.log(hot(10, 1)), torch.log(hot(10, 1))), torch.log(hot(10, 2)))
    assert_close(semiring.multiply(torch.log(hot(10, 3)), torch.log(hot(10, 4))), torch.log(hot(10, 7)))
    assert_close(semiring.multiply(torch.log(hot(10, 4)), torch.log(hot(10, 7))), torch.log(torch.zeros(10)))
    assert_close(semiring.multiply(torch.log(torch.zeros(10)), torch.log(torch.zeros(10))), torch.log(torch.zeros(10)))
    assert_close(semiring.multiply(torch.log(hot(10, 0, 0.5)), torch.log(hot(10, 0, 0.5))), torch.log(hot(10, 0, 0.25)))
    assert_close(semiring.multiply(torch.log(hot(10, 2, 0.5)), torch.log(hot(10, 4, 0.5))), torch.log(hot(10, 6, 0.25)))

def test_random_multiply() -> None:
    generator = torch.manual_seed(123)
    semiring = LogCountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        b = torch.rand(10, generator=generator)
        assert_close(semiring.multiply(torch.log(a), torch.log(b)), torch.log(reference_multiply(a, b)))

def test_multi_dim_multiply() -> None:
    generator = torch.manual_seed(123)
    semiring = LogCountingSemiring(10)
    a = torch.rand((29, 1, 7, 10), generator=generator)
    b = torch.rand((1, 13, 7, 10), generator=generator)
    r = semiring.multiply(torch.log(a), torch.log(b))
    assert r.size() == (29, 13, 7, 10)
    expected_r = torch.empty(29, 13, 7, 10)
    for i in range(29):
        for j in range(13):
            for k in range(7):
                expected_r[i, j, k] = reference_multiply(a[i, 0, k], b[0, j, k])
    assert_close(r, torch.log(expected_r))

def test_star() -> None:
    semiring = LogCountingSemiring(10)
    for a in [
        hot(10, 0, 0.5),
        hot(10, 1, 0.5),
        hot(10, [2, 4], 0.2)
    ]:
        assert_close(semiring.star(torch.log(a)), torch.log(reference_star(a)))

def test_random_star() -> None:
    generator = torch.manual_seed(123)
    semiring = LogCountingSemiring(10)
    for _ in range(50):
        a = torch.rand(10, generator=generator)
        assert_close(semiring.star(torch.log(a)), torch.log(reference_star(a)))
