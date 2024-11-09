from rayuela.base.semiring import Real
from rayuela.base.state import State
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA

from recognizers.automata.finite_automaton import (
    FiniteAutomatonContainer
)
from .rayuela_util import from_rayuela_fsa

def modular_arithmetic_simple_rayuela_fsa() -> FSA:
    A = FSA(R=Real)

    num_states = 0

    def new_state(label=None):
        nonlocal num_states
        result = State(num_states, label=label)
        num_states += 1
        return result

    q0 = new_state("q0")
    A.set_I(q0, Real.one)

    base_states = {i: new_state(str(i)) for i in range(5)}
    q_final = new_state("q_final")

    # Read the first number.
    for i in range(5):
        A.add_arc(q0, Sym(str(i)), base_states[i], Real(1 / 5))

    # intermediate states for each number and operation
    intermediate_states = {}
    for i in range(5):
        for op in ["+", "-", "*"]:
            intermediate_states[f"{i}{op}"] = new_state(f"{i}{op}")

    # output check states for verifying the result
    check_states = {i: new_state(f"={i}") for i in range(5)}

    for i in range(5):
        A.add_arc(base_states[i], Sym("+"), intermediate_states[f"{i}+"], Real(1 / 4))
        A.add_arc(base_states[i], Sym("-"), intermediate_states[f"{i}-"], Real(1 / 4))
        A.add_arc(base_states[i], Sym("*"), intermediate_states[f"{i}*"], Real(1 / 4))

    for key, op_state in intermediate_states.items():
        current_number, op = int(key[0]), key[1]
        for x in range(5):
            match op:
                case "+":
                    result = current_number + x
                case "-":
                    result = current_number - x
                case "*":
                    result = current_number * x
                case _:
                    raise ValueError
            next_state = result % 5
            A.add_arc(op_state, Sym(str(x)), base_states[next_state], Real(1 / 5))

    for i in range(5):
        A.add_arc(base_states[i], Sym("="), check_states[i], Real(1 / 4))

    for i in range(5):
        A.add_arc(check_states[i], Sym(str(i)), q_final, Real(1))

    A.set_F(q_final, Real.one)

    return A

def modular_arithmetic_simple_dfa() -> tuple[FiniteAutomatonContainer, list[str]]:
    return from_rayuela_fsa(modular_arithmetic_simple_rayuela_fsa())
