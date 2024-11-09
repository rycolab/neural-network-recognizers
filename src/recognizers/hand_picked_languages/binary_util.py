import numpy
import numpy.random

from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse
)

def binary_encoding(x, n):
    r = binary_encoding_no_padding(x)
    if len(r) > n:
        raise ValueError
    while len(r) < n:
        r.append(0)
    return r

def binary_encoding_no_padding(x):
    r = []
    if x == 0:
        r.append(0)
    else:
        while x != 0:
            bit = (x & 1)
            r.append(bit)
            x >>= 1
    return r

def decode_binary(s):
    if not s:
        raise ValueError
    x = 0
    mask = 1
    for bit in s:
        if bit:
            x |= mask
        mask <<= 1
    return x

def python_to_numpy_generator(generator):
    return numpy.random.default_rng(generator.getrandbits(32))

def proportions_to_ints(proportions, total):
    cumsum = numpy.cumsum(proportions[:-1])
    scaled_cumsum = cumsum * total
    int_cumsum = numpy.floor(scaled_cumsum)
    ints = numpy.diff(int_cumsum, prepend=0, append=total).astype(numpy.uint)
    return ints

OPERATOR = 2
EQUALS = 3

class BinaryArithmeticOperation(WeightedLanguage):

    def operator_str(self):
        raise NotImplementedError

    def alphabet_size(self):
        return 4

    def symbol_to_str(self, symbol):
        if symbol in (0, 1):
            return str(symbol)
        elif symbol == OPERATOR:
            return self.operator_str()
        elif symbol == EQUALS:
            return '='
        else:
            raise ValueError

    def with_length_range(self, length_range):
        return LengthRestrictedBinaryAddition(self, length_range)

class LengthRestrictedBinaryArithmeticOperation(LengthRestrictedWeightedLanguage):

    def __init__(self, parent, length_range):
        super().__init__()
        self.parent = parent
        self.min_length, self.max_length = length_range

    def get_max_x(self, max_z):
        # This should be the solution for x to: compute_z(x, 0) <= max_z.
        raise NotImplementedError

    def get_max_y(self, max_z, x):
        # This should be the solution for y to: compute_z(x, y) <= max_z.
        raise NotImplementedError

    def compute_z(self, x, y):
        # Compute the operation on x and y.
        raise NotImplementedError

    def get_length_weights(self):
        raise NotImplementedError

    def supports_log_probability(self):
        return False

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False
    
    def sample(self, generator, include_log_probability, include_next_symbols):
        n, n_x, n_y, n_z = self.sample_lengths(generator)
        max_z = 2 ** n_z - 1
        x = generator.randint(0, min(2 ** n_x - 1, self.get_max_x(max_z)))
        y = generator.randint(0, min(2 ** n_y - 1, self.get_max_y(max_z, x)))
        u_x = binary_encoding(x, n_x)
        u_y = binary_encoding(y, n_y)
        if generator.randrange(2):
            u_x, u_y = u_y, u_x
        z = self.compute_z(x, y)
        u_z = binary_encoding_no_padding(z)
        n_z_pad = n_z - len(u_z)
        u_z_padding = (0,) * n_z_pad
        s = (*u_x, OPERATOR, *u_y, EQUALS, *u_z, *u_z_padding)
        if include_next_symbols:
            next_symbols = []
            next_symbols.append([0, 1])
            for _ in range(len(u_x)):
                next_symbols.append([0, 1, OPERATOR])
            next_symbols.append([0, 1])
            for _ in range(len(u_y)):
                next_symbols.append([0, 1, EQUALS])
            for a in u_z:
                next_symbols.append([a])
            for _ in range(n_z_pad + 1):
                next_symbols.append([0, ReservedSymbol.EOS])
        else:
            next_symbols = None
        return s, Parse(None, next_symbols)

    def sample_lengths(self, generator):
        n = generator.randint(max(5, self.min_length), self.max_length)
        numpy_generator = python_to_numpy_generator(generator)
        proportions = numpy_generator.dirichlet(self.get_length_weights())
        n_x, n_y, n_z = map(int, proportions_to_ints(proportions, n - 5) + 1)
        # Make sure x doesn't have more bits than y.
        if n_x > n_y:
            n_x, n_y = n_y, n_x
        return n, n_x, n_y, n_z

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
        if i < n and s[i] == OPERATOR:
            i += 1
        else:
            return False
        u_y = []
        while i < n and s[i] in (0, 1):
            u_y.append(s[i])
            i += 1
        if i < n and s[i] == EQUALS:
            i += 1
        else:
            return False
        u_z = s[i:]
        if not all(a in (0, 1) for a in u_z):
            return False
        if not (u_x and u_y and u_z):
            return False
        x = decode_binary(u_x)
        y = decode_binary(u_y)
        z = decode_binary(u_z)
        return self.compute_z(x, y) == z
