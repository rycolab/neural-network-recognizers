from collections.abc import Sequence
from typing import List, Set, Union

from rayuela.base.symbol import Sym, to_sym, ε


class Alphabet(Sequence):
    def __init__(self, symbols: Union[List[Sym], Set[Sym], str, "Alphabet"]):
        # We want to have this a set so that the order is deterministic.
        if isinstance(symbols, str):
            symbols = [to_sym(sym) for sym in symbols]  # TODO: Make more robust
        elif isinstance(symbols, set):
            symbols = list(symbols)
        elif isinstance(symbols, Alphabet):
            symbols = symbols.symbols

        assert len(symbols) > 0, "Alphabet must be non-empty."
        assert ε not in symbols, "ε must not be in the alphabet."

        self.symbols = symbols

    def __str__(self):
        return "{" + ", ".join(str(sym) for sym in self.symbols) + "}"

    def __repr__(self):
        return "{" + ", ".join(str(sym) for sym in self.symbols) + "}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, Alphabet) and self.symbols == other.symbols

    def __len__(self):
        return len(self.symbols)

    def __contains__(self, item):
        return item in self.symbols

    def __iter__(self):
        return iter(self.symbols)

    def __getitem__(self, index):
        return self.symbols[index]

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def add(self, symbol: Sym):
        self.symbols.append(symbol)

    def discard(self, symbol: Sym):
        if symbol in self.symbols:
            self.symbols.remove(symbol)

    def union(self, other: Union["Alphabet", Set[Sym], Sequence[Sym]]):
        if not isinstance(other, Alphabet):
            other = to_alphabet(other)
        return Alphabet(self.symbols + other.symbols)

    def difference(self, other: Union["Alphabet", Set[Sym], Sequence[Sym]]):
        if not isinstance(other, Alphabet):
            other = to_alphabet(other)
        return Alphabet(set(self.symbols) - set(other.symbols))


def to_alphabet(
    symbols: Union[List[Union[Sym, str]], Set[Union[Sym, str]], str]
) -> Alphabet:
    """Creates an Alphabet from a list of symbols.

    Args:
        symbols (Union[List[Union[Sym, str]], Set[Union[Sym, str]]]): The symbols
            which will form the alphabet.

    Returns:
        Alphabet: The alphabet.
    """

    if isinstance(symbols, Alphabet):
        return symbols
    else:
        return Alphabet(
            [to_sym(sym) if isinstance(sym, str) else sym for sym in symbols]
        )
