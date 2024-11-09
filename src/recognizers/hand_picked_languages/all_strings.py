from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer,
    FiniteAutomatonTransition
)

def all_strings_dfa() -> tuple[FiniteAutomatonContainer, None]:
    q0 = State(0)
    a, b = map(Symbol, range(2))
    M = FiniteAutomatonContainer(
        num_states=1,
        alphabet_size=2,
        initial_state=q0
    )
    M.add_transition(FiniteAutomatonTransition(q0, a, q0))
    M.add_transition(FiniteAutomatonTransition(q0, b, q0))
    M.add_accept_state(q0)
    return M, None
