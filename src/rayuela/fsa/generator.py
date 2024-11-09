""" The generator module for various pre-defined weighted finite-state languages.
"""

from itertools import product

from rayuela.base.semiring import Boolean, Real, Semiring
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA


def dyck_k_m(k: int, m: int, R: Semiring = Real) -> FSA:
    A = FSA(R=R)

    str2sym = {}
    for a in range(k):
        str2sym[f"({a}"] = a
        str2sym[f"){a}"] = k + a

    str2state = {"": 0}
    for n in range(m):
        for i, ks in enumerate(product(range(k), repeat=n + 1)):
            str2state[" ".join([f"({a}" for a in ks])] = i + 1
    state2str = {v: k for k, v in str2state.items()}
    print(str2state)

    def _f(m_: int, s: State) -> None:
        if m_ == m:
            return

        for a in range(k):
            opening_bracket = Sym(str2sym[f"({a}"])
            closing_bracket = Sym(str2sym[f"){a}"])
            t = State(
                str2state[state2str[s.idx] + " " + f"({a}" if m_ < m else f"({a}"]
            )
            A.add_arc(s, opening_bracket, t, R.one)
            A.add_arc(t, closing_bracket, s, R.one)
            _f(m_ + 1, t)

    q_0 = State(str2state[""])
    A.set_I(q_0, R.one)
    A.set_F(q_0, R.one)

    _f(m_=0, s=q_0)

    return A


def even_pairs() -> FSA:
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


def parity() -> FSA:
    """Returns the WFSA specifying the language of strings with an even number of bs."""
    A = FSA(R=Real)

    q_0 = State(0)
    q_b = State(1)
    A.set_I(q_0, Real.one)

    A.add_arc(q_0, Sym("0"), q_0, Real(1 / 3))
    A.add_arc(q_0, Sym("1"), q_b, Real(1 / 3))
    A.add_arc(q_b, Sym("0"), q_b, Real(1 / 2))
    A.add_arc(q_b, Sym("1"), q_0, Real(1 / 2))

    A.set_F(q_0, Real(1 / 3))

    return A


def cycle_navigation() -> FSA:
    A = FSA(R=Real)

    Q = [State(i) for i in range(5)]
    qf = State(5)
    A.set_I(Q[0], Real.one)

    for i in range(5):
        A.add_arc(Q[i - 1], Sym("5"), Q[i], Real(1 / 4))  # Right
        A.add_arc(Q[i], Sym("6"), Q[i - 1], Real(1 / 4))  # Left

    for i in range(5):
        A.add_arc(Q[i], Sym("7"), Q[i], Real(1 / 4))  # Stay
        A.add_arc(Q[i], Sym(str(i)), qf, Real(1 / 4))  # Finish

    A.set_F(qf, Real.one)

    return A

def modular_arithmetic() -> FSA:
    # Note: This has a limitation: it may incorrectly consider
    # strings that are too long as not part of the language due to the probability
    # becoming epsilon.
    A = FSA(R=Real)

    base_states = {i: State(i) for i in range(5)}
    q_final = State(100)

    A.set_I(base_states[0], Real.one)

    # intermediate states for each number and operation
    intermediate_states = {}
    for i in range(5):
        for op in ['+', '-', '*']:
            intermediate_states[f'{i}{op}'] = State(5 * i + ord(op))

    # output check states for verifying the result
    check_states = {i: State(50 + i) for i in range(5)}

    for i in range(5):
        A.add_arc(base_states[i], Sym('+'), intermediate_states[f'{i}+'], Real(1 / 4))
        A.add_arc(base_states[i], Sym('-'), intermediate_states[f'{i}-'], Real(1 / 4))
        A.add_arc(base_states[i], Sym('*'), intermediate_states[f'{i}*'], Real(1 / 4))

    for key, op_state in intermediate_states.items():
        current_number, op = int(key[0]), key[1]
        for x in range(5):  
            next_state = (current_number + x) % 5 if op == '+' else (current_number - x) % 5 if op == '-' else (current_number * x) % 5
            A.add_arc(op_state, Sym(str(x)), base_states[next_state], Real(1 / 5))

    for i in range(5):
        A.add_arc(base_states[i], Sym('#'), check_states[i], Real(1 / 4))

    for i in range(5):
        A.add_arc(check_states[i], Sym(str(i)), q_final, Real(1))

    A.set_F(q_final, Real.one)

    return A
