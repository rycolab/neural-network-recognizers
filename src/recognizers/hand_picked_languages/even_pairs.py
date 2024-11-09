from rayuela.base.semiring import Real
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from .rayuela_util import from_rayuela_fsa

def even_pairs_rayuela_fsa() -> FSA:
    """Returns the WFSA specifying the language of even pairs."""
    A = FSA(R=Real)

    q_0 = State(0)
    q_a = State(1)
    q_b = State(2)
    q_ar = State(3)
    q_br = State(4)
    A.set_I(q_0, Real.one)

    A.add_arc(q_0, Sym("0"), q_a, Real(1 / 3))
    A.add_arc(q_0, Sym("1"), q_b, Real(1 / 3))

    A.add_arc(q_a, Sym("0"), q_a, Real(1 / 3))
    A.add_arc(q_a, Sym("1"), q_ar, Real(1 / 3))
    A.add_arc(q_ar, Sym("0"), q_a, Real(1 / 2))
    A.add_arc(q_ar, Sym("1"), q_ar, Real(1 / 2))

    A.add_arc(q_b, Sym("1"), q_b, Real(1 / 3))
    A.add_arc(q_b, Sym("0"), q_br, Real(1 / 3))
    A.add_arc(q_br, Sym("1"), q_b, Real(1 / 2))
    A.add_arc(q_br, Sym("0"), q_br, Real(1 / 2))

    A.set_F(q_0, Real(1 / 3))
    A.set_F(q_a, Real(1 / 3))
    A.set_F(q_b, Real(1 / 3))

    return A

def even_pairs_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    return from_rayuela_fsa(even_pairs_rayuela_fsa())
