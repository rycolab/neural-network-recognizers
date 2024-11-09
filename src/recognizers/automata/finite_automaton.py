from collections.abc import Iterable
from typing import TypeVar

from .automaton import (
    State,
    Transition,
    Automaton,
    AutomatonContainer,
    WeightedAutomaton,
    WeightedAutomatonContainer
)
from .semiring import Semiring

Weight = TypeVar('Weight')

class FiniteAutomatonTransition(Transition):
    pass

class FiniteAutomaton(Automaton):

    def transitions(self) -> Iterable[FiniteAutomatonTransition]:
        return super().transitions() # type: ignore

class FiniteAutomatonContainer(FiniteAutomaton, AutomatonContainer):

    _accept_states: dict[State, None]

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int,
        initial_state: State=State(0)
    ):
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            initial_state=initial_state
        )
        self._accept_states = {}

    def add_transition(self, transition: FiniteAutomatonTransition) -> None:
        self._add_transition(transition)

    def is_accept_state(self, state: State) -> bool:
        return state in self._accept_states

    def add_accept_state(self, state: State) -> None:
        self._accept_states[state] = None

class WeightedFiniteAutomaton(FiniteAutomaton, WeightedAutomaton[Weight]):

    def transition_weights(self) -> Iterable[tuple[FiniteAutomatonTransition, Weight]]:
        return super().transition_weights() # type: ignore

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        raise NotImplementedError

class WeightedFiniteAutomatonContainer(WeightedFiniteAutomaton[Weight], WeightedAutomatonContainer[Weight]):

    _accept_weights: dict[State, Weight]

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int,
        initial_state: State=State(0),
        semiring: Semiring[Weight]
    ):
        super().__init__(
            num_states=num_states,
            alphabet_size=alphabet_size,
            initial_state=initial_state,
            semiring=semiring
        )
        self._accept_weights = {}

    def set_transition_weight(self,
        transition: FiniteAutomatonTransition,
        weight: Weight
    ) -> None:
        self._set_transition_weight(transition, weight)

    def is_accept_state(self, state: State) -> bool:
        return state in self._accept_weights

    def accept_weights(self) -> Iterable[tuple[State, Weight]]:
        return self._accept_weights.items()

    def set_accept_weight(self, state: State, weight: Weight) -> None:
        self._accept_weights[state] = weight
