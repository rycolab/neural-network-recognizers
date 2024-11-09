from rayuela.fsa.fsa import FSA

from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer,
    FiniteAutomatonTransition
)

def from_rayuela_fsa(A: FSA) -> tuple[FiniteAutomatonContainer, list[str]]:
    sorted_symbols = sorted(A.Sigma, key=lambda x: x.value)
    sorted_states = sorted(A.Q, key=lambda x: x.idx)
    symbol_to_int = { a.value: Symbol(i) for i, a in enumerate(sorted_symbols) }
    state_to_int = { q.idx: State(i) for i, q in enumerate(sorted_states) }
    (q0, _), = A.I
    M = FiniteAutomatonContainer(
        num_states=len(state_to_int),
        alphabet_size=len(symbol_to_int),
        initial_state=state_to_int[q0.idx]
    )
    for q in sorted_states:
        for a, t, _ in sorted(A.arcs(q), key=lambda x: (x[0].value, x[1].idx)):
            M.add_transition(FiniteAutomatonTransition(
state_to_int[q.idx],
                symbol_to_int[a.value],
                state_to_int[t.idx]
            ))
    for q, _ in sorted(A.F, key=lambda x: x[0].idx):
        M.add_accept_state(state_to_int[q.idx])
    alphabet = [str(a.value) for a in sorted_symbols]
    if len(set(alphabet)) != len(alphabet):
        raise ValueError('alphabet strings are not unique')
    return M, alphabet
