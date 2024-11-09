from typing import TypeVar

import torch

from .semiring import Semiring

T = TypeVar('T')

def lehmann(A: T, semiring: Semiring) -> None:
    num_states = A.size(0)
    if A.size(1) != num_states:
        raise ValueError
    for k in range(num_states):
        semiring.add_in_place(
            A,
            semiring.multiply(
                semiring.multiply(
                    semiring.transform_tensors(A, lambda x: x[:, k, None]),
                    semiring.transform_tensors(
                        semiring.star(semiring.transform_tensors(A, lambda x: x[k, k])),
                        lambda x: x[None, None]
                    )
                ),
                semiring.transform_tensors(A, lambda x: x[k, None, :])
            )
        )
    semiring.add_one_in_place(
        semiring.transform_tensors(A, lambda x: move_last_dim(torch.diagonal(x)))
    )

def floyd_warshall(A: T, semiring: Semiring) -> None:
    num_states = A.size(0)
    if A.size(1) != num_states:
        raise ValueError
    for k in range(num_states):
        semiring.add_in_place(
            A,
            semiring.multiply(
                semiring.transform_tensors(A, lambda x: x[:, k, None]),
                semiring.transform_tensors(A, lambda x: x[k, None, :])
            )
        )
    semiring.add_one_in_place(
        semiring.transform_tensors(A, lambda x: move_last_dim(torch.diagonal(x)))
    )

def move_last_dim(x):
    if x.ndim > 1:
        last_dim = x.ndim - 1
        permutation = (last_dim, *range(last_dim))
        x = x.permute(*permutation)
    return x
