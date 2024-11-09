import math

from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse
)

MISSING = 2

class MissingDuplicateString(WeightedLanguage):

    def alphabet_size(self):
        return 3

    def symbol_to_str(self, symbol):
        if symbol < 2:
            return str(symbol)
        elif symbol == MISSING:
            return '_'
        else:
            raise ValueError

    def with_length_range(self, length_range):
        return LengthRestrictedMissingDuplicateString(self, length_range)

    def is_positive(self, s):
        n, r = divmod(len(s), 2)
        if r != 0:
            return False
        i = None
        for j, a in enumerate(s):
            if a == MISSING:
                if i is None:
                    i = j
                else:
                    # More than one _
                    return False
        # No _
        if i is None:
            return False
        ss = list(s)
        ss[i] = 1
        return ss[:n] == ss[n:]

class LengthRestrictedMissingDuplicateString(LengthRestrictedWeightedLanguage):

    def __init__(self, parent, length_range):
        super().__init__()
        self.parent = parent
        self.min_length, self.max_length = length_range
        self.min_n = max(1, math.ceil(self.min_length / 2))
        self.max_n = math.floor(self.max_length / 2)
        if self.min_n > self.max_n:
            raise ValueError

    def supports_log_probability(self):
        return False

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False

    def sample(self, generator, include_log_probability, include_next_symbols):
        w = self._sample_w(generator)
        s = self._w_to_string(w, generator)
        if include_next_symbols:
            next_symbols = self._s_to_next_symbols(s)
        else:
            next_symbols = None
        return s, Parse(None, next_symbols)

    def is_negative(self, s, include_edit_distance):
        return (not self._is_positive(s), None)

    def _is_positive(self, s):
        return self.min_length <= len(s) <= self.max_length and self.parent.is_positive(s)

    def _sample_w(self, generator):
        n = generator.randint(self.min_n, self.max_n)
        w = [generator.randrange(2) for _ in range(n)]
        i = generator.randrange(n)
        w[i] = 1
        return w

    def _w_to_string(self, w, generator):
        s = [*w, *w]
        indexes_with_1 = [i for i, a in enumerate(s) if a == 1]
        i = generator.choice(indexes_with_1)
        s[i] = MISSING
        return tuple(s)

    def _s_to_next_symbols(self, s):
        result = []
        n = len(s)
        i = 0
        next_symbol_set = [0, 1, MISSING]
        result.append(next_symbol_set)
        while i < n and s[i] != MISSING:
            result.append(next_symbol_set)
            i += 1
        while i < n:
            next_symbol_set = [0, 1]
            if self.parent.is_positive(s[:i+1]):
                next_symbol_set.append(ReservedSymbol.EOS)
            result.append(next_symbol_set)
            i += 1
        return result
