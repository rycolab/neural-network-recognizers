from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)

def empty_set_dfa() -> tuple[FiniteAutomatonContainer, None]:
    return FiniteAutomatonContainer(num_states=1, alphabet_size=2), None
