from itertools import product

from rayuela.base.semiring import Real, Semiring
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from .rayuela_util import from_rayuela_fsa

def dyck_k_m_rayuela_fsa(k: int, m: int, R: Semiring = Real) -> FSA:
    A = FSA(R=R)

    str2state = {"": 0}
    for n in range(m):
        for ks in product(range(k), repeat=n + 1):
            str2state[" ".join([f"({a}" for a in ks])] = len(str2state)
    state2str = {v: k for k, v in str2state.items()}

    def _f(m_: int, s: State) -> None:
        if m_ > m:
            return

        if m_ < m:
            for a in range(k):
                opening_bracket = Sym(f"({a}")
                closing_bracket = Sym(f"){a}")
                s_str = state2str[s.idx]
                if m_ == 0:
                    t_str = f"({a}"
                elif m_ < m:
                    t_str = s_str + " " + f"({a}"
                t = State(str2state[t_str])
                A.add_arc(s, opening_bracket, t, R.one)

                _f(m_ + 1, t)

        if s.idx == 0:
            return

        s_str = state2str[s.idx]
        s_list = s_str.split(" ")
        b = int(s_list[-1][1:])

        closing_bracket = Sym(f"){b}")
        t_str = " ".join(s_list[:-1])
        t = State(str2state[t_str])
        A.add_arc(s, closing_bracket, t, R.one)

    q_0 = State(str2state[""])
    A.set_I(q_0, R.one)
    A.set_F(q_0, R.one)

    _f(m_=0, s=q_0)

    return A

def dyck_k_m_dfa(k: int, m: int) -> tuple[FiniteAutomatonContainer, list[str]]:
    return from_rayuela_fsa(dyck_k_m_rayuela_fsa(k, m))
