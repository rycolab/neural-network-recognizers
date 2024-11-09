import math

from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse
)
from .binary_util import (
    python_to_numpy_generator,
    proportions_to_ints,
    binary_encoding,
    binary_encoding_no_padding,
    decode_binary
)

EQUALS = 2

class ComputeSqrt(WeightedLanguage):

    def alphabet_size(self):
        return 3

    def symbol_to_str(self, symbol):
        if symbol in (0, 1):
            return str(symbol)
        elif symbol == EQUALS:
            return '='
        else:
            raise ValueError

    def with_length_range(self, length_range):
        return LengthRestrictedComputeSqrt(self, length_range)

class LengthRestrictedComputeSqrt(LengthRestrictedWeightedLanguage):

    def __init__(self, parent, length_range):
        super().__init__()
        self.parent = parent
        self.min_length, self.max_length = length_range

    def supports_log_probability(self):
        return False

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False

    def compute_z(self, x):
        return math.floor(math.sqrt(x))

    def sample(self, generator, include_log_probability, include_next_symbols):
        n, n_x, n_z = self.sample_lengths(generator)
        x = generator.randint(0, min(2 ** n_x - 1, 2 ** (2 * n_z) - 1))
        u_x = binary_encoding(x, n_x)
        z = self.compute_z(x)
        u_z = binary_encoding_no_padding(z)
        n_z_pad = n_z - len(u_z)
        u_z_padding = (0,) * n_z_pad
        s = (*u_x, EQUALS, *u_z, *u_z_padding)
        if include_next_symbols:
            next_symbols = []
            next_symbols.append([0, 1])
            for _ in range(len(u_x)):
                next_symbols.append([0, 1, EQUALS])
            for a in u_z:
                next_symbols.append([a])
            for _ in range(n_z_pad + 1):
                next_symbols.append([0, ReservedSymbol.EOS])
        else:
            next_symbols = None
        return s, Parse(None, next_symbols)

    def sample_lengths(self, generator):
        n = generator.randint(max(3, self.min_length), self.max_length)
        numpy_generator = python_to_numpy_generator(generator)
        proportions = numpy_generator.dirichlet((2, 1))
        n_x, n_z = map(int, proportions_to_ints(proportions, n - 3) + 1)
        return n, n_x, n_z

    def is_negative(self, s, include_edit_distance):
        return (not self._is_positive(s), None)

    def _is_positive(self, s):
        if not (self.min_length <= len(s) <= self.max_length):
            return False
        i = 0
        n = len(s)
        u_x = []
        while i < n and s[i] in (0, 1):
            u_x.append(s[i])
            i += 1
        if i < n and s[i] == EQUALS:
            i += 1
        else:
            return False
        u_z = s[i:]
        if not all(a in (0, 1) for a in u_z):
            return False
        if not (u_x and u_z):
            return False
        x = decode_binary(u_x)
        z = decode_binary(u_z)
        return self.compute_z(x) == z
