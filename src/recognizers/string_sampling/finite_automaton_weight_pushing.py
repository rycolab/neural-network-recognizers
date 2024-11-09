import dataclasses
import math
import random
from collections.abc import Iterable
from typing import Literal

import torch

from recognizers.automata.reserved import ReservedSymbol
from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    FiniteAutomatonTransition,
    FiniteAutomaton,
    WeightedFiniteAutomaton
)
from recognizers.automata.finite_automaton_allsum import backward
from recognizers.automata.boolean_semiring import BooleanSemiring
from recognizers.automata.lehmann import lehmann
from recognizers.automata.fixed_point_iteration import fixed_point_iteration
from .weighted_language import (
    String,
    ValidNextSymbolSet,
    ValidNextSymbolList
)

@dataclasses.dataclass
class OutgoingTransition:
    symbol: Symbol | Literal[ReservedSymbol.EPSILON]
    state_to: State
    log_weight: torch.Tensor
    """The log of the original weights, which can be used to get the log weight
    of a run. The size of this tensor is ``(max_count+1,)``."""

@dataclasses.dataclass
class Actions:
    cumulative_weights: torch.Tensor
    """The cumulative transition/accept weights. This tensor has size
    ``(len(transitions)+1, max_count+1)``. The weights are cumulative across
    dimension 0. The last weight in dimension 0 corresponds to the weight of
    accepting."""
    transitions: list[OutgoingTransition]
    """The list of outgoing transitions, listed in the same order as
    ``cumulative_weights``."""
    accept_log_weight: torch.Tensor
    """The log of the original accept weights. The size of this tensor is
    ``(max_count+1,)``."""
    next_symbols: ValidNextSymbolSet
    """The set of valid next symbols for this state."""

@dataclasses.dataclass
class NormalizedCountingFiniteAutomaton:

    actions: list[Actions]
    alphabet_size: int
    initial_state: State
    transitions: dict[tuple[State, Symbol], State]
    accept_states: set[State]
    total_length_weights: torch.Tensor
    max_length: int

    @staticmethod
    def from_parts(
        num_states: int,
        alphabet_size: int,
        initial_state: State,
        transition_weights: Iterable[tuple[FiniteAutomatonTransition, torch.Tensor]],
        accept_weights: Iterable[tuple[State, torch.Tensor]],
        allsum: torch.Tensor,
        zero: torch.Tensor,
        next_symbols: list[ValidNextSymbolSet]
    ) -> 'NormalizedCountingFiniteAutomaton':
        # It's assumed that all weights are given in log space.
        actions = [
            Actions(None, [], None, next_symbols[q]) # type: ignore
            for q in range(num_states)
        ]
        accept_weights_dict = dict(accept_weights)
        transitions_by_key = {}
        for t, weight in transition_weights:
            actions[t.state_from].transitions.append(OutgoingTransition(
                symbol=t.symbol,
                state_to=t.state_to,
                log_weight=weight
            ))
            key = (t.state_from, t.symbol)
            if key not in transitions_by_key:
                transitions_by_key[key] = []
            transitions_by_key[key].append(t.state_to)
        # Check whether the automaton is deterministic.
        is_deterministic = all(
            a != ReservedSymbol.EPSILON and len(rs) == 1
            for (q, a), rs in transitions_by_key.items()
        )
        if not is_deterministic:
            raise NotImplementedError(
                'this finite automaton is nondeterministic, which is not '
                'supported'
            )
        transitions_dict = { k : r for k, (r,) in transitions_by_key.items() }
        for state, a in enumerate(actions):
            action_weights = [t.log_weight for t in a.transitions]
            accept_weight = accept_weights_dict.get(State(state), zero)
            action_weights.append(accept_weight)
            # Precompute the cumulative sums of the probabilities so they can
            # be used during sampling.
            a.cumulative_weights = torch.cumsum(
                # Convert the log weights to normalized probabilities with
                # softmax.
                # Note that if all of the log probabilities are -inf, this will
                # return nan. This is ok -- it indicates that there is no
                # probability distribution at that index, because it is
                # impossible to sample something from that index.
                torch.softmax(
                    torch.stack(action_weights, dim=0),
                    dim=0
                ),
                dim=0
            )
            a.accept_log_weight = accept_weight
        return NormalizedCountingFiniteAutomaton(
            actions=actions,
            alphabet_size=alphabet_size,
            initial_state=initial_state,
            transitions=transitions_dict,
            accept_states=set(accept_weights_dict.keys()),
            total_length_weights=allsum,
            max_length=allsum.size(-1) - 1
        )

    def actions_at_state(self, state: State) -> Actions:
        return self.actions[state]

    def is_accept_state(self, state: State) -> bool:
        return state in self.accept_states

    def valid_lengths(self, length_range: tuple[int, int]) -> list[int]:
        lo, hi = length_range
        cum_weights = self.actions_at_state(self.initial_state).cumulative_weights
        is_valid = (cum_weights[0, lo:hi+1] > -math.inf).tolist()
        return [
            l
            for l, l_is_valid in zip(
                range(length_range[0], length_range[1] + 1),
                is_valid,
                strict=True
            )
            if l_is_valid
        ]

    def accepts(self, string: Iterable[Symbol]) -> bool:
        # NOTE This assumes the automaton is deterministic.
        state = self.initial_state
        for symbol in string:
            state = self.transitions.get((state, symbol))
            if state is None:
                return False
        return state in self.accept_states

    def sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolList | None]:

        sampled_string = []
        if include_log_probability:
            log_probability = 0.0
        else:
            log_probability = None
        if include_next_symbols:
            next_symbols = []
        else:
            next_symbols = None

        state = self.initial_state
        for length_counter in range(length, -1, -1):
            actions = self.actions_at_state(state)
            if include_next_symbols:
                # The set of next symbols is precomputed for each state.
                next_symbols.append(actions.next_symbols)
            # Randomly sample the next action.
            cum_weights = actions.cumulative_weights[:, length_counter].tolist()
            index, = generator.choices(
                range(len(cum_weights)),
                cum_weights=cum_weights
            )
            if index < len(actions.transitions):
                transition = actions.transitions[index]
                if transition.symbol == ReservedSymbol.EPSILON:
                    raise ValueError
                sampled_string.append(transition.symbol)
                if include_log_probability:
                    log_probability += transition.log_weight[length_counter].item()
                state = transition.state_to
            else:
                # Accept.
                if include_log_probability:
                    log_probability += actions.accept_log_weight[length_counter].item()
                break
        return tuple(sampled_string), log_probability, next_symbols

    def total_length_weight(self, length: int) -> float:
        return self.total_length_weights[length].item()

def push_finite_automaton_weights(
    M: WeightedFiniteAutomaton[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device
) -> NormalizedCountingFiniteAutomaton:
    semiring = M.semiring()
    initial_state = M.initial_state()
    backward_weights: torch.Tensor = backward(M, dtype, device)
    return NormalizedCountingFiniteAutomaton.from_parts(
        num_states=M.num_states(),
        alphabet_size=M.alphabet_size(),
        initial_state=initial_state,
        transition_weights=(
            (t, semiring.multiply(weight, backward_weights[t.state_to]))
            for t, weight in M.transition_weights()
        ),
        accept_weights=M.accept_weights(),
        allsum=backward_weights[initial_state],
        zero=semiring.zeros((), dtype, device),
        next_symbols=next_symbols_per_state(M, device)
    )

def next_symbols_per_state(
    M: FiniteAutomaton,
    device: torch.device
) -> list[ValidNextSymbolSet]:
    semiring = BooleanSemiring()
    dtype = torch.bool
    num_states = M.num_states()
    alphabet_size = M.alphabet_size()

    # Build tables of scanning and non-scanning transitions.
    non_scanning_transitions = semiring.zeros((num_states, num_states), dtype, device)
    scanning_transitions = semiring.zeros((num_states, num_states, alphabet_size), dtype, device)
    for t in M.transitions():
        if t.symbol == ReservedSymbol.EPSILON:
            non_scanning_transitions[t.state_from, t.state_to] = True
        else:
            scanning_transitions[t.state_from, t.state_to, t.symbol] = True

    # For each pair of states (p, q), compute whether p can reach q without
    # scanning.
    non_scanning_inner = non_scanning_transitions.clone()
    lehmann(non_scanning_inner, semiring)

    # For each pair of states (p, q) and symbol a, compute whether p can reach
    # q with a run that scans a as its first symbol.
    def scanning_inner_step(scanning_inner):
        any_inner = semiring.sum(scanning_inner, dims=(2,))
        semiring.add_in_place(any_inner, non_scanning_inner)
        return semiring.add(
            # p q a
            semiring.sum(
                # p q a r
                semiring.multiply(
                    # p r a -> p 1 a r
                    scanning_transitions.transpose(1, 2)[:, None],
                    # r q -> 1 q 1 r
                    any_inner.transpose(0, 1)[None, :, None]
                ),
                dims=(3,)
            ),
            # p q a
            semiring.sum(
                # p q a r
                semiring.multiply(
                    # p r -> p 1 1 r
                    non_scanning_inner[:, None, None],
                    # r q a -> 1 q a r
                    scanning_inner.permute(1, 2, 0)[None]
                ),
                dims=(3,)
            )
        )

    scanning_inner = fixed_point_iteration(
        scanning_inner_step,
        equal=semiring.equal,
        zero=scanning_transitions
    )
    is_accept = torch.tensor(
        [M.is_accept_state(q) for q in range(num_states)],
        dtype=dtype,
        device=device
    )
    symbol_backward = semiring.sum(scanning_inner[:, is_accept], dims=(1,))
    eos_backward = semiring.sum(non_scanning_inner[:, is_accept], dims=(1,))
    next_symbols = [
        { a for a, is_included in enumerate(q_backward) if is_included }
        for q, q_backward in enumerate(symbol_backward.tolist())
    ]
    for q, has_eos in enumerate(eos_backward):
        if has_eos:
            next_symbols[q].add(ReservedSymbol.EOS)
    return next_symbols
