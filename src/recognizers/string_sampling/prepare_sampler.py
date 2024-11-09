import argparse
import math

import torch

from recognizers.automata.automaton import State, Transition
from recognizers.automata.finite_automaton import (
    FiniteAutomaton,
    FiniteAutomatonTransition,
    WeightedFiniteAutomaton,
    WeightedFiniteAutomatonContainer,
)
from recognizers.automata.reserved import ReservedSymbol
from recognizers.automata.log_counting_semiring import LogCountingSemiring
from recognizers.string_sampling.finite_automaton_weight_pushing import (
    push_finite_automaton_weights
)

def get_transition_weight(
    transition: Transition,
    log_probability: float,
    weight_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    weight = torch.full((weight_size,), -math.inf, dtype=dtype, device=device)
    index = int(transition.symbol != ReservedSymbol.EPSILON)
    weight[index] = log_probability
    return weight

def get_accept_weight(
    log_probability: float,
    weight_size: int,
    dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    weight = torch.full((weight_size,), -math.inf, dtype=dtype, device=device)
    weight[0] = log_probability
    return weight

def lift_finite_automaton(
    M: FiniteAutomaton,
    max_count: int,
    dtype: torch.dtype,
    device: torch.device
) -> WeightedFiniteAutomaton[torch.Tensor]:
    num_states = M.num_states()
    weight_size = max_count + 1
    result = WeightedFiniteAutomatonContainer[torch.Tensor](
        num_states=num_states,
        alphabet_size=M.alphabet_size(),
        initial_state=M.initial_state(),
        semiring=LogCountingSemiring(weight_size)
    )
    grouped_transitions: list[list[FiniteAutomatonTransition]] = [[] for _ in range(num_states)]
    for t in M.transitions():
        grouped_transitions[t.state_from].append(t)
    for state_from, transitions in enumerate(grouped_transitions):
        is_accept_state = M.is_accept_state(State(state_from))
        num_actions = len(transitions) + int(is_accept_state)
        if num_actions > 0:
            log_prob = -math.log(num_actions)
            for t in transitions:
                result.set_transition_weight(
                    t,
                    get_transition_weight(t, log_prob, weight_size, dtype, device)
                )
            if is_accept_state:
                result.set_accept_weight(
                    State(state_from),
                    get_accept_weight(log_prob, weight_size, dtype, device)
                )
    return result

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--device', type=torch.device, required=True)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = args.device

    data = torch.load(args.input)
    automaton = data.pop('automaton')
    match automaton:
        case FiniteAutomaton():
            prepared_automaton = push_finite_automaton_weights(
                lift_finite_automaton(
                    automaton,
                    args.max_length,
                    dtype,
                    device
                ),
                dtype,
                device
            )
        case _:
            raise ValueError
    data['sampler'] = prepared_automaton
    torch.save(data, args.output)

if __name__ == '__main__':
    main()
