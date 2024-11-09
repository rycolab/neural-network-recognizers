from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer,
    FiniteAutomatonTransition
)

def repeat_01_dfa() -> tuple[FiniteAutomatonContainer, None]:
    q0, q1 = map(State, range(2))
    a, b = map(Symbol, range(2))
    M = FiniteAutomatonContainer(
        num_states=2,
        alphabet_size=2,
        initial_state=q0
    )
    M.add_transition(FiniteAutomatonTransition(q0, a, q1))
    M.add_transition(FiniteAutomatonTransition(q1, b, q0))
    M.add_accept_state(q0)
    return M, None
