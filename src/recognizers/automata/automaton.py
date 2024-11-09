import dataclasses
from collections.abc import Iterable
from typing import Any, Generic, Literal, NewType, TypeVar

from .reserved import ReservedSymbol
from .semiring import Semiring

State = NewType('State', int)
Symbol = NewType('Symbol', int)
SymbolOrEpsilon = Symbol | Literal[ReservedSymbol.EPSILON]
Weight = TypeVar('Weight')

@dataclasses.dataclass(frozen=True)
class Transition:
    state_from: State
    symbol: SymbolOrEpsilon
    state_to: State

class Automaton:

    def states(self) -> Iterable[State]:
        raise NotImplementedError

    def num_states(self) -> int:
        raise NotImplementedError

    def alphabet_size(self) -> int:
        raise NotImplementedError

    def transitions(self) -> Iterable[Transition]:
        raise NotImplementedError

    def initial_state(self) -> State:
        raise NotImplementedError

    def is_accept_state(self, state: State) -> bool:
        raise NotImplementedError

class AutomatonContainerBase(Automaton):

    _num_states: int
    _alphabet_size: int
    _initial_state: State
    _transitions: dict[Transition, Any]

    def __init__(self,
        *,
        num_states: int=1,
        alphabet_size: int,
        initial_state: State=State(0)
    ):
        super().__init__()
        if initial_state >= num_states:
            raise ValueError
        self._num_states = num_states
        self._alphabet_size = alphabet_size
        self._initial_state = initial_state
        self._transitions = {}

    def states(self) -> Iterable[State]:
        return range(self._num_states) # type: ignore

    def num_states(self) -> int:
        return self._num_states

    def new_state(self) -> State:
        state = self._num_states
        self._num_states += 1
        return State(state)

    def alphabet_size(self) -> int:
        return self._alphabet_size

    def transitions(self) -> Iterable[Transition]:
        return self._transitions.keys()

    def initial_state(self) -> State:
        return self._initial_state

class AutomatonContainer(AutomatonContainerBase):

    _transitions: dict[Transition, None]

    def _add_transition(self, transition: Transition) -> None:
        self._transitions[transition] = None

class WeightedAutomaton(Automaton, Generic[Weight]):

    def semiring(self) -> Semiring[Weight]:
        raise NotImplementedError

    def transition_weights(self) -> Iterable[tuple[Transition, Weight]]:
        raise NotImplementedError

class WeightedAutomatonContainer(WeightedAutomaton[Weight], AutomatonContainerBase):

    _transitions: dict[Transition, Weight]
    _semiring: Semiring[Weight]

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
            initial_state=initial_state
        )
        self._semiring = semiring

    def semiring(self) -> Semiring[Weight]:
        return self._semiring

    def transition_weights(self) -> Iterable[tuple[Transition, Weight]]:
        return self._transitions.items()

    def _set_transition_weight(self,
        transition: Transition,
        weight: Weight
    ) -> None:
        self._transitions[transition] = weight
