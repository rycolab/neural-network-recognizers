import torch

from rayuela.base.semiring import Tropical as RayuelaTropical
from rayuela.base.state import State as RayuelaState
from rayuela.base.symbol import Sym as RayuelaSym, ε as RAYUELA_EPSILON
from rayuela.fsa.fsa import FSA

from recognizers.automata.automaton import State, Symbol
from recognizers.automata.finite_automaton import (
    WeightedFiniteAutomatonContainer,
    FiniteAutomatonTransition,
    WeightedFiniteAutomaton
)
from recognizers.automata.tropical_semiring import TropicalSemiring
from recognizers.automata.finite_automaton_allsum import allsum
from recognizers.string_sampling.weighted_language import String

def compute_edit_distance(
    tropical_fsa: FSA,
    s: String,
    dtype: torch.dtype,
    device: torch.device
) -> int:
    str_fsa = string_edit_distance_fsa(s, tropical_fsa.Sigma)
    i = str_fsa.intersect(tropical_fsa)
    i_container = to_tropical_wfa(i, dtype, device)
    edit_distance = allsum(i_container, dtype, device, no_star=True)
    return int(edit_distance.item())

def string_edit_distance_fsa(s, Sigma):
    R = RayuelaTropical
    F = FSA(R=R)
    str_to_sym = [RayuelaSym(x) for x in s]

    for i in range(len(s)):
        for a in Sigma:
            F.add_arc(RayuelaState(i), a, RayuelaState(i), R(1.0))
            if a != str_to_sym[i]:
                F.add_arc(RayuelaState(i), a, RayuelaState(i+1), R(1.0))
        F.add_arc(RayuelaState(i), str_to_sym[i], RayuelaState(i + 1), R(0.0))
        F.add_arc(RayuelaState(i), RAYUELA_EPSILON, RayuelaState(i + 1), R(1.0))

    for a in Sigma:
        F.add_arc(RayuelaState(len(s)), a, RayuelaState(len(s)), R(1.0))

    F.set_I(RayuelaState(0), F.R.one)
    F.set_F(RayuelaState(len(s)), F.R.one)

    return F

def to_tropical_wfa(
    A: FSA,
    dtype: torch.dtype,
    device: torch.device
) -> WeightedFiniteAutomaton[torch.Tensor]:
    symbol_to_int = {a.value: Symbol(i) for i, a in enumerate(A.Sigma)}
    state_to_int = {q.idx: State(i) for i, q in enumerate(A.Q)}
    (q0, _), = A.I
    initial_state = state_to_int[q0.idx]
    M = WeightedFiniteAutomatonContainer(
        num_states=len(state_to_int),
        alphabet_size=len(symbol_to_int),
        initial_state=initial_state,
        semiring=TropicalSemiring()
    )
    for q in A.Q:
        for a, t, w in A.arcs(q):
            transition = FiniteAutomatonTransition(
                state_to_int[q.idx],
                symbol_to_int[a.value],
                state_to_int[t.idx]
            )
            M.set_transition_weight(transition, torch.tensor(w.value, dtype=dtype, device=device))
    for q, w in A.F:
        M.set_accept_weight(state_to_int[q.idx], torch.tensor(w.value, dtype=dtype, device=device))
    return M
