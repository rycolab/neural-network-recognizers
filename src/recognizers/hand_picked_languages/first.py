from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer,
    FiniteAutomatonTransition
)

def first_dfa() -> tuple[FiniteAutomatonContainer, None]:
    q0, q1 = map(State, range(2))
    s0, s1 = map(Symbol, range(2))
    M = FiniteAutomatonContainer(
        num_states=2,
        alphabet_size=2,
        initial_state=q0
    )
    M.add_transition(FiniteAutomatonTransition(q0, s1, q1))
    M.add_transition(FiniteAutomatonTransition(q1, s0, q1))
    M.add_transition(FiniteAutomatonTransition(q1, s1, q1))
    M.add_accept_state(q1)
    return M, None
