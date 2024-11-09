import dataclasses
import math
import random
from typing import Literal

from recognizers.automata.automaton import Symbol
from recognizers.automata.reserved import ReservedSymbol

String = tuple[Symbol, ...]
ValidNextSymbolSet = set[Symbol | Literal[ReservedSymbol.EOS]]
ValidNextSymbolList = list[ValidNextSymbolSet]

@dataclasses.dataclass
class Parse:
    log_probability: float | None
    next_symbols: ValidNextSymbolList | None

class EmptyLanguageError(ValueError):
    pass

class LengthRestrictedWeightedLanguage:

    def sample(self,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, Parse]:
        """Sample a string from this language.

        :param generator: Random number generator.
        :return: The sampled string and any requested data.
        """
        raise NotImplementedError

    def is_negative(self,
        s: String,
        include_edit_distance: bool
    ) -> tuple[bool, int]:
        raise NotImplementedError

    def supports_log_probability(self) -> bool:
        raise NotImplementedError

    def supports_next_symbols(self) -> bool:
        raise NotImplementedError

    def supports_edit_distance(self) -> bool:
        raise NotImplementedError

class WeightedLanguage:

    def alphabet_size(self) -> int:
        """Return the size of this language's alphabet."""
        raise NotImplementedError

    def symbol_to_str(self, symbol: Symbol) -> str:
        """Given a symbol as an int, return a string representation of it."""
        return str(symbol)

    def with_length_range(self,
        length_range: tuple[int, int]
    ) -> LengthRestrictedWeightedLanguage:
        raise NotImplementedError
