import dataclasses
from collections.abc import Iterable
from typing import TypeVar

import torch

from .finite_automaton import WeightedFiniteAutomaton
from .lehmann import lehmann, floyd_warshall

T = TypeVar('T')

def inner(
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device,
    no_star: bool=False
) -> T:
    semiring = M.semiring()
    num_states = M.num_states()
    A = semiring.zeros(size=(num_states, num_states), dtype=dtype, device=device)
    for t, weight in M.transition_weights():
        # Coalesce weights over different symbols.
        semiring.add_in_place(
            semiring.transform_tensors(A, lambda x: x[t.state_from, t.state_to]),
            weight
        )
    if no_star:
        floyd_warshall(A, semiring)
    else:
        lehmann(A, semiring)
    return A

def backward_from_inner(
    M: WeightedFiniteAutomaton[T],
    A: T,
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = M.semiring()
    num_states = M.num_states()
    accept_weights = semiring.zeros((num_states,), dtype, device)
    for f, w in M.accept_weights():
        semiring.set_index(accept_weights, f, w)
    return semiring.sum(
        semiring.multiply(
            A,
            semiring.transform_tensors(accept_weights, lambda x: x[None])
        ),
        dims=(1,)
    )

def backward(
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device,
    no_star: bool=False
) -> T:
    return backward_from_inner(M, inner(M, dtype, device, no_star), dtype, device)

def allsum(
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device,
    no_star: bool=False
) -> T:
    semiring = M.semiring()
    A = inner(M, dtype, device, no_star)
    q = M.initial_state()
    # Only compute the backward weight of the initial state.
    b = backward_from_inner(M, semiring.transform_tensors(A, lambda x: x[q:q+1]), dtype, device)
    return semiring.transform_tensors(b, lambda x: x[0])
