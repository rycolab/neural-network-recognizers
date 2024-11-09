from rayuela.base.semiring import Real
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from .rayuela_util import from_rayuela_fsa

def cycle_navigation() -> FSA:
    A = FSA(R=Real)

    Q = [State(i) for i in range(5)]
    qf = State(5)
    A.set_I(Q[0], Real.one)

    for i in range(5):
        A.add_arc(Q[i - 1], Sym(">"), Q[i], Real(1 / 4))  # Right
        A.add_arc(Q[i], Sym("<"), Q[i - 1], Real(1 / 4))  # Left

    for i in range(5):
        A.add_arc(Q[i], Sym("="), Q[i], Real(1 / 4))  # Stay
        A.add_arc(Q[i], Sym(str(i)), qf, Real(1 / 4))  # Finish

    A.set_F(qf, Real.one)

    return A

def cycle_navigation_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    return from_rayuela_fsa(cycle_navigation_rayuela_fsa())
