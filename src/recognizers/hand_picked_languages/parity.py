from rayuela.base.semiring import Real
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from .rayuela_util import from_rayuela_fsa

def parity_rayuela_fsa() -> FSA:
    """The language of binary strings with an odd number of 1s."""
    A = FSA(R=Real)

    q_even = State(0)
    q_odd = State(1)
    A.set_I(q_even, Real.one)

    A.add_arc(q_even, Sym("0"), q_even, Real(1 / 3))
    A.add_arc(q_even, Sym("1"), q_odd, Real(1 / 3))
    A.add_arc(q_odd, Sym("0"), q_odd, Real(1 / 2))
    A.add_arc(q_odd, Sym("1"), q_even, Real(1 / 2))

    A.set_F(q_odd, Real(1 / 3))

    return A

def parity_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    return from_rayuela_fsa(parity_rayuela_fsa())
