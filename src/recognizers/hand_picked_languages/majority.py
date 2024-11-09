from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse,
    EmptyLanguageError
)

class Majority(WeightedLanguage):

    def alphabet_size(self):
        return 2

    def with_length_range(self, length_range):
        return LengthRestrictedMajority(length_range)

class LengthRestrictedMajority(LengthRestrictedWeightedLanguage):

    def __init__(self, length_range):
        super().__init__()
        self.min_length, self.max_length = length_range
        if self.min_length > self.max_length or self.max_length < 1:
            raise EmptyLanguageError

    def supports_log_probability(self):
        # It's possible to compute this, but it's a little complicated.
        return False

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False
    
    def sample(self, generator, include_log_probability, include_next_symbols):
        w = self._sample_string(generator)
        log_probability = None
        if include_next_symbols:
            next_symbols = self._string_to_next_symbols(w)
        else:
            next_symbols = None
        return w, Parse(log_probability, next_symbols)
    
    def is_negative(self, s, include_edit_distance):
        return (self._parse_string(s) is None, None)
    
    def _parse_string(self, s):
        n_1s = sum(s)
        n_0s = len(s) - n_1s
        if n_1s > n_0s:
            return n_1s, n_0s
        return None
    
    def _sample_string(self, generator):
        n = generator.randint(max(self.min_length, 1), self.max_length)
        c_1 = generator.randint(n // 2 + 1, n)
        c_0 = n - c_1
        result = [0] * c_0 + [1] * c_1
        generator.shuffle(result)
        return tuple(result)
    
    def _string_to_next_symbols(self, w):
        next_symbols = []
        for t in range(len(w) + 1):
            if self._parse_string(w[:t]) is None:
                next_symbols.append([0, 1])
            else:
                next_symbols.append([0, 1, ReservedSymbol.EOS])
        return next_symbols
