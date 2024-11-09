from typing import Sequence, Type

import rayuela
from rayuela.base.semiring import Semiring


class Expr(Semiring):
    def __add__(self, other):
        if self == Expr.zero:
            return other
        if other == Expr.zero:
            return self
        return Union(self, other)

    def __mul__(self, other):
        if self == Expr.zero:
            return Expr.zero
        elif other == Expr.zero:
            return Expr.zero
        elif self == Expr.one:
            return other
        elif other == Expr.one:
            return self
        else:
            return Concatenation(self, other)

    def __eq__(self, other):
        return isinstance(other, Expr) and self.value == other.value

    def __str__(self):
        return str(self.value) if self.value not in ["ε", "EOS", "BOS"] else ""

    def __repr__(self):
        return repr(self.value)

    def __hash__(self):
        return hash(self.value)

    def star(self):
        # Added by Anej: I think this is correct; at least it makes the regexes nicer.
        if self == Expr.zero:
            return Expr.one
        else:
            return Star(self)

    def fsa(self, R: Type[Semiring]) -> "rayuela.fsa.fsa.FSA":
        raise NotImplementedError()


class Sym(Expr):
    def __len__(self):
        return 0 if self.value == "ε" else 1

    def __eq__(self, other):
        return isinstance(other, Sym) and self.value == other.value

    def __invert__(self):
        return self

    def __hash__(self):
        return hash(self.value)

    def fsa(self, R: Type[Semiring]) -> "rayuela.fsa.fsa.FSA":
        from rayuela.fsa.fsa import FSA
        from rayuela.base.state import State

        F = FSA(R)
        w = F.R.one
        F.add_arc(State(1), self.value, State(2), w)
        F.set_I(State(1), F.R.one)
        F.set_F(State(2), F.R.one)
        return F


Expr.zero = Sym("∞")
Expr.one = Sym("ε")


class Concatenation(Expr):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__((x, y))

    def fsa(self, R):
        return self.x.fsa(R).concatenate(self.y.fsa(R))

    def __repr__(self):
        return f"{repr(self.x)}⋅{repr(self.y)}"

    def __str__(self):
        return f"{self.x}⋅{self.y}"

    def __hash__(self):
        return hash(self.value)


class Union(Expr):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        super().__init__((x, y))

    def fsa(self, R):
        return self.x.fsa(R).union(self.y.fsa(R))

    def __repr__(self):
        return f"({repr(self.x)}|{repr(self.y)})"

    def __str__(self):
        return f"({self.x}|{self.y})"

    def __hash__(self):
        return hash(self.value)


class Star(Expr):
    def __init__(self, x):
        self.x = x
        super().__init__(x)

    def fsa(self, R):
        return self.x.fsa(R).kleene_closure()

    def __repr__(self):
        return f"({repr(self.x)})*"

    def __str__(self):
        return f"({self.x})*"

    def __hash__(self):
        return hash(self.value)


# Some commonly used (special) symbol
ε = Expr.one
ε_1 = Sym("ε_1")
ε_2 = Sym("ε_2")

φ = Sym("φ")
ρ = Sym("ρ")
σ = Sym("σ")

# String sybols
BOS = Sym("BOS")
EOS = Sym("EOS")

# Stack symbols
BOT = Sym("⊥")


def to_sym(s: str) -> Sym:
    """Converts a single character string to a symbol (Sym).

    Args:
        s (str): The input string

    Returns:
        Sym: Sym-ed version of the input string.
    """
    if isinstance(s, Sym):
        return s
    else:
        return Sym(s)


def to_sym_seq(s: str) -> Sequence[Sym]:
    """Converts a string to a sequence of symbols (Sym).

    Args:
        s (str): The input string

    Returns:
        Sequence[Sym]: Sym-ed version of the input string.
    """
    return [to_sym(c) for c in s]
