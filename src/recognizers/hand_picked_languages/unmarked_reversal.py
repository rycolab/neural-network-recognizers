import math

from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse
)

class UnmarkedReversal(WeightedLanguage):

    def __init__(self):
        super().__init__()
        self.num_symbols = 2
        self.log_num_symbols = math.log(self.num_symbols)

    def alphabet_size(self):
        return self.num_symbols

    def symbol_to_str(self, symbol):
        if symbol < self.num_symbols:
            return str(symbol)
        else:
            raise ValueError

    def with_length_range(self, length_range):
        return LengthRestrictedUnmarkedReversal(self, length_range)

    def is_positive(self, s):
        n, r = divmod(len(s), 2)
        last_index = len(s) - 1
        return (
            r == 0 and
            all(
                s[i] == s[last_index - i]
                for i in range(n)
            )
        )

class LengthRestrictedUnmarkedReversal(LengthRestrictedWeightedLanguage):

    def __init__(self, parent, length_range):
        super().__init__()
        self.parent = parent
        self.min_length, self.max_length = length_range
        self.min_n = math.ceil(self.min_length / 2)
        self.max_n = math.floor(self.max_length / 2)
        if self.min_n > self.max_n:
            raise ValueError
        n_range_size = self.max_n - self.min_n + 1
        self.n_log_prob = -math.log(n_range_size)

    def supports_log_probability(self):
        return True

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False

    def sample(self, generator, include_log_probability, include_next_symbols):
        w = self._sample_w(generator)
        s = self._w_to_string(w)
        if include_log_probability:
            log_probability = self._n_to_log_probability(len(w))
        else:
            log_probability = None
        if include_next_symbols:
            next_symbols = self._s_to_next_symbols(s)
        else:
            next_symbols = None
        return s, Parse(log_probability, next_symbols)

    def is_negative(self, s, include_edit_distance):
        return (not self._is_positive(s), None)

    def _is_positive(self, s):
        return self.min_length <= len(s) <= self.max_length and self.parent.is_positive(s)

    def _sample_w(self, generator):
        n = generator.randint(self.min_n, self.max_n)
        return [generator.randrange(self.parent.num_symbols) for _ in range(n)]

    def _w_to_string(self, w):
        return (*w, *reversed(w))

    def _n_to_log_probability(self, n):
        return self.n_log_prob - n * self.parent.log_num_symbols

    def _s_to_next_symbols(self, s):
        result = []
        for i in range(len(s) + 1):
            next_symbol_set = [0, 1]
            if self.parent.is_positive(s[:i]):
                next_symbol_set.append(ReservedSymbol.EOS)
            result.append(next_symbol_set)
        return result
