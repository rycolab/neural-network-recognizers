from collections import defaultdict as dd
from fractions import Fraction
from math import exp, log

import numpy as np
from frozendict import frozendict


# base code from
# https://github.com/timvieira/hypergraphs/blob/master/hypergraphs/semirings/boolean.py
class Semiring:
    zero: "Semiring"
    one: "Semiring"
    idempotent = False

    def __init__(self, value):
        self.value = value

    @classmethod
    def zeros(cls, N, M):
        import numpy as np

        return np.full((N, M), cls.zero)

    @classmethod
    def chart(cls, default=None):
        if default is None:
            default = cls.zero
        return dd(lambda: default)

    @classmethod
    def diag(cls, N):
        W = cls.zeros(N, N)
        for n in range(N):
            W[n, n] = cls.one

        return W

    @classmethod
    @property
    def is_field(self):
        return False

    def __add__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Derivation(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return Derivation.one

    def __add__(self, other):
        return Derivation(frozenset(self.value.union(other.value)))

    def __mul__(self, other):
        # TODO: add special cases
        return Derivation(frozenset([x + y for x in self.value for y in other.value]))

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"Derivation({self.value})"

    def __hash__(self):
        return hash(self.value)


Derivation.zero = Derivation(frozenset([]))
Derivation.one = Derivation(frozenset([tuple()]))
Derivation.idempotent = False


class KBest(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def __add__(self, other):
        return KBest(self.value.union(other.value))

    def __mul__(self, other):
        # TODO: add special cases
        return KBest(set([x + y for x in self.value for y in other.value]))

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"KBest({self.value})"

    def __hash__(self):
        return hash(self.value)


KBest.zero = Derivation(set([]))
KBest.one = Derivation(set([tuple()]))
KBest.idempotent = False


class Free(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return "(" + self.value + ")^*"

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Free(self.value + " + " + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Free(self.value + other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"Free({self.value})"

    def __hash__(self):
        return hash(self.value)


Free.zero = Free("∞")
Free.one = Free("")
Free.idempotent = False


class Count(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return self.one

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Count(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Count(self.value * other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"{self.value}"

    def __hash__(self):
        return hash(self.value)

    def __float__(self):
        return float(self.value)


Count.zero = Count(0)
Count.one = Count(1)
Count.idempotent = False


class Entropy(Semiring):
    def __init__(self, x, y):
        super().__init__((x, y))

    def star(self):
        tmp = 1.0 / (1.0 - self.value[0])
        return Entropy(tmp, tmp * tmp * self.value[1])

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return Entropy(self.value[0] + other.value[0], self.value[1] + other.value[1])

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Entropy(
            self.value[0] * other.value[0],
            self.value[0] * other.value[1] + self.value[1] * other.value[0],
        )

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"Entropy({self.value})"

    def __hash__(self):
        return hash(self.value)


Entropy.zero = Entropy(0.0, 0.0)
Entropy.one = Entropy(1.0, 0.0)
Entropy.idempotent = False


def cky_semiring_builder(G, R):  # noqa: C901
    one = "1"

    class CKY(Semiring):
        def __init__(self, value):
            super().__init__(frozendict(value))

        def star(self):
            return CKY.one

        def __add__(self, other):
            if other is self.zero:
                return self
            if self is self.zero:
                return other

            result = dd(lambda: self.R.zero)
            for k, v in self.value.items():
                result[k] += v
            for k, v in other.value.items():
                result[k] += v

            return CKY(frozendict(result))

        def __mul__(self, other):  # noqa: C901
            if other is self.one:
                return self
            if self is self.one:
                return other
            if other is self.zero:
                return self.zero
            if self is self.zero:
                return self.zero

            result = dd(lambda: self.R.zero)

            # special handling of "1" symbol
            if one in self.value:
                for nt, v in other.value.items():
                    result[nt] += v
            if one in other.value:
                for nt, v in self.value.items():
                    result[nt] += v

            # Cartesian product subject to grammar constraint
            for p, w in self.G.binary:
                if p.body[0] in self.value and p.body[1] in other.value:
                    result[p.head] += self.value[p.body[0]] * other.value[p.body[1]] * w

            return CKY(frozendict(result))

        def __eq__(self, other):
            return self.value == other.value

        def __repr__(self):
            return f"{self.value}"

        def __hash__(self):
            return hash(self.value)

    CKY.G = G
    CKY.R = R
    CKY.zero = CKY(dict())
    CKY.one = CKY({one: CKY.R.one})
    CKY.idempotent = False
    CKY.cancellative = False

    return CKY


class String(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return String.one

    def __add__(self, other):
        from rayuela.base.misc import lcp

        if other is self.zero:
            return self
        if self is self.zero:
            return other
        return String(lcp(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return String(self.value + other.value)

    def __truediv__(self, other):
        from rayuela.base.misc import lcp

        prefix = lcp(self.value, other.value)
        return String(self.value[len(prefix) :])

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"{self.value}"

    def __hash__(self):
        return hash(self.value)


# unique "infinity" string
String.zero = String("∞")
# empty string
String.one = String("")
String.idempotent = False
String.cancellative = False


class Segment:
    def __init__(self, segment, inverse=False):
        self.segment = segment
        self.inverse = inverse

    def __invert__(self):
        return Segment(self.segment, inverse=not self.inverse)

    def __eq__(self, other):
        return self.segment == other.segment and self.inverse == other.inverse

    def __repr__(self):
        return f"{self.segment, self.inverse}"

    def __str__(self):
        return f"{self.segment}"

    def __hash__(self):
        return hash((self.segment, self.inverse))


class SegmentationGroup(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return SegmentationGroup.one

    def __add__(self, other):
        if other is self.zero:
            return self
        if self is self.zero:
            return other

        if len(self.value) < len(other.value):
            return self
        return other

    @staticmethod
    def _simplify(arg1, arg2):
        changed = True
        while changed:
            changed = False
            if len(arg1) == 0 or len(arg2) == 0:
                break

            if ~arg1[-1] == arg2[0]:
                arg1 = arg1[:-1]
                arg2 = arg2[1:]
                changed = True

        return arg1 + arg2

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero

        # make this better
        simplified = tuple(
            [
                x
                for x in SegmentationGroup._simplify(self.value, other.value)
                if x.segment != ""
            ]
        )
        return SegmentationGroup(simplified)

    def __invert__(self):
        return SegmentationGroup(tuple(reversed([~x for x in self.value])))

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"{'|'.join(map(str, self.value))}"

    def __hash__(self):
        return hash(self.value)


# unique "infinity" string
SegmentationGroup.zero = SegmentationGroup(None)
# empty string
# TODO: inverse
SegmentationGroup.one = SegmentationGroup(())
SegmentationGroup.idempotent = False
SegmentationGroup.cancellative = False


class Boolean(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return Boolean.one

    def __add__(self, other):
        return Boolean(self.value or other.value)

    def __mul__(self, other):
        if other.value is self.one:
            return self.value
        if self.value is self.one:
            return other.value
        if other.value is self.zero:
            return self.zero
        if self.value is self.zero:
            return self.zero
        return Boolean(other.value and self.value)

    # TODO: is this correct?
    def __invert__(self):
        return Boolean.one

    def __truediv__(self, other):
        return Boolean.one

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)


Boolean.zero = Boolean(False)
Boolean.one = Boolean(True)
Boolean.idempotent = True
# TODO: check
Boolean.cancellative = True


class MaxPlus(Semiring):
    def __init__(self, value):
        super().__init__(value)

    def star(self):
        return self.one

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return MaxPlus(max(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return MaxPlus(self.value + other.value)

    def __invert__(self):
        return MaxPlus(-self.value)

    def __truediv__(self, other):
        return MaxPlus(self.value - other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"MaxPlus({self.value})"


MaxPlus.zero = MaxPlus(float("-inf"))
MaxPlus.one = MaxPlus(0.0)
MaxPlus.idempotent = True
MaxPlus.superior = True
MaxPlus.cancellative = True


class Tropical(Semiring):
    def __init__(self, value):
        self.value = value

    def star(self):
        return self.one

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __add__(self, other):
        return Tropical(min(self.value, other.value))

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Tropical(self.value + other.value)

    def __invert__(self):
        return Tropical(-self.value)

    def __truediv__(self, other):
        return Tropical(self.value - other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"Tropical({self.value})"

    def __str__(self):
        return str(self.value)


Tropical.zero = Tropical(float("inf"))
Tropical.one = Tropical(0.0)
Tropical.idempotent = True
Tropical.superior = True
Tropical.cancellative = True


class Rational(Semiring):
    def __init__(self, value):
        self.value = Fraction(value)

    def star(self):
        return Rational(Fraction("1") / (Fraction("1") - self.value))

    @classmethod
    @property
    def is_field(self):
        return True

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Rational(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Rational(self.value * other.value)

    def __invert__(self):
        return Rational(1 / self.value)

    def __truediv__(self, other):
        return Rational(self.value / other.value)

    def __eq__(self, other):
        return np.allclose(float(self.value), float(other.value))

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        # return f'Real({self.value})'
        return f"{self.value}"

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.value)


Rational.zero = Rational(Fraction("0"))
Rational.one = Rational(Fraction("1"))
Rational.idempotent = False
Rational.cancellative = True


class Real(Semiring):
    def __init__(self, value):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.value = value

    def star(self):
        return Real(1.0 / (1.0 - self.value))

    @classmethod
    @property
    def is_field(self):
        return True

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Real(self.value + other.value)

    def __sub__(self, other):
        return Real(self.value - other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Real(self.value * other.value)

    def __invert__(self):
        return Real(1.0 / self.value)

    def __pow__(self, other):
        return Real(self.value**other)

    def __truediv__(self, other):
        return Real(self.value / other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        # return f'Real({self.value})'
        return f"{round(self.value, 5)}"

    def __eq__(self, other):
        # return float(self.value) == float(other.value)
        return np.allclose(float(self.value), float(other.value), atol=1e-100)

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.value)


Real.zero = Real(0.0)
Real.one = Real(1.0)
Real.idempotent = False
Real.cancellative = True


class Log(Semiring):
    def __init__(self, value):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.value = value

    def star(self):
        return Log(-log(1 / exp(self.value) - 1) - self.value)

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        # stolen from https://github.com/timvieira/crf/blob/master/crf/basecrf.py
        if self is Log.zero:
            return other
        if other is Log.zero:
            return self

        if self.value > other.value:
            return Log(self.value + log(1 + exp(other.value - self.value)))
            # return Log(self.value + log(sum(exp(other.value-self.value)).sum()))
        return Log(other.value + log(1 + exp(self.value - other.value)))

    def __mul__(self, other):
        if self is Log.zero:
            return Log.zero
        if other is Log.zero:
            return Log.zero
        return Log(self.value + other.value)

    def __repr__(self):
        # return f'Real({self.value})'
        return f"{round(self.value, 15)}"

    def __eq__(self, other):
        # return float(self.value) == float(other.value)
        return np.allclose(float(self.value), float(other.value), atol=1e-3)

    # TODO: find out why this wasn't inherited
    def __hash__(self):
        return hash(self.value)


Log.zero = Log(-float("inf"))
Log.one = Log(0.0)
Log.idempotent = False
Log.cancellative = True


class Integer(Semiring):
    def __init__(self, value):
        # TODO: this is hack to deal with the fact
        # that we have to hash weights
        self.value = value

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Integer(self.value + other.value)

    def __mul__(self, other):
        if other is self.one:
            return self
        if self is self.one:
            return other
        if other is self.zero:
            return self.zero
        if self is self.zero:
            return self.zero
        return Integer(self.value * other.value)

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"Integer({self.value})"

    def __eq__(self, other):
        return float(self.value) == float(other.value)

    def __hash__(self):
        return hash(self.value)


Integer.zero = Integer(0)
Integer.one = Integer(1)
Integer.idempotent = False
Integer.cancellative = True


def vector_semiring_builder(semiring, N):
    class VectorSemiring(Semiring):
        def __init__(self, x):
            super().__init__(x)

        def star(self):
            raise NotImplementedError

        def __add__(self, other):
            return VectorSemiring(self.value + other.value)

        def __mul__(self, other):
            return VectorSemiring(self.value * other.value)

        def __eq__(self, other):
            return self.value == other.value

        def __repr__(self):
            return f"Vector({self.value})"

        def __hash__(self):
            return hash(self.value)

    VectorSemiring.zero = VectorSemiring(np.full(N, semiring.zero))
    VectorSemiring.one = VectorSemiring(np.full(N, semiring.one))
    VectorSemiring.idempotent = semiring.idempotent

    return VectorSemiring


class ProductSemiring(Semiring):
    def __init__(self, x, y):
        super().__init__((x, y))

    def __add__(self, other):
        w1, w2 = self.value[0], other.value[0]
        v1, v2 = self.value[1], other.value[1]
        return ProductSemiring(w1 + w2, v1 + v2)

    def __mul__(self, other):
        w1, w2 = self.value[0], other.value[0]
        v1, v2 = self.value[1], other.value[1]
        return ProductSemiring(w1 * w2, v1 * v2)

    def __truediv__(self, other):
        w1, w2 = self.value[0], other.value[0]
        v1, v2 = self.value[1], other.value[1]
        return ProductSemiring(w1 / w2, v1 / v2)

    def __invert__(self):
        return ProductSemiring(~self.value[0], ~self.value[1])

    def star(self):
        return ProductSemiring(self.value[0].star(), self.value[1].star())

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        if isinstance(self.value[0], String):
            # the important special case of encoding transducers
            if len(self.value[0].value) > 0:
                return f"{self.value[0]} / {self.value[1]}"
            else:
                return f"{self.value[1]}"
        return f"〈{repr(self.value[0])}, {repr(self.value[1])}〉"

    def __str__(self):
        if isinstance(self.value[0], String):
            # the important special case of encoding transducers
            if len(self.value[0].value) > 0:
                return f"{str(self.value[0])} / {str(self.value[1])}"
            else:
                return f"{str(self.value[1])}"
        return f"〈{str(self.value[0])}, {str(self.value[1])}〉"

    def __hash__(self):
        return hash(self.value)


def product_semiring_builder(R1, R2):
    ProductSemiring.zero = ProductSemiring(R1.zero, R2.zero)
    ProductSemiring.one = ProductSemiring(R1.one, R2.one)
    ProductSemiring.idempotent = R1.idempotent and R2.idempotent

    return ProductSemiring


def expectation_semiring_builder(R1, R2):
    class ExpectationSemiring(Semiring):
        def __init__(self, x, y):
            super().__init__((x, y))

        def star(self):
            w, v = self.value[0], self.value[1]
            return ExpectationSemiring(w.star(), w.star() * w.star() * v)

        def __add__(self, other):
            w1, w2 = self.value[0], other.value[0]
            v1, v2 = self.value[1], other.value[1]
            return ExpectationSemiring(w1 + w2, v1 + v2)

        def __mul__(self, other):
            w1, w2 = self.value[0], other.value[0]
            v1, v2 = self.value[1], other.value[1]
            return ExpectationSemiring(w1 * w2, w1 * v2 + w2 * v1)

        def __eq__(self, other):
            return self.value == other.value

        def __repr__(self):
            return f"Expect({self.value})"

        def __hash__(self):
            return hash(self.value)

    ExpectationSemiring.zero = ExpectationSemiring(R1.zero, R2.zero)
    ExpectationSemiring.one = ExpectationSemiring(R1.one, R2.zero)
    ExpectationSemiring.idempotent = R1.idempotent and R2.idempotent

    return ExpectationSemiring


def conditionalpoisson_semiring_builder(K):
    class ConditionalPoisson(Semiring):
        def __init__(self, x):
            super().__init__(x)

        def star(self):
            raise NotImplementedError

        def __add__(self, other):
            return ConditionalPoisson(np.convolve(self.value, other.value)[:K])

        def __mul__(self, other):
            return ConditionalPoisson(self.value * other.value)

        def __eq__(self, other):
            return (
                isinstance(other, ConditionalPoisson)
                and (self.value == other.value).all()
            )

        def __repr__(self):
            return str(self.value)

        def __hash__(self):
            return hash(self.value)

    tmp = np.zeros((K))
    tmp[0] = 1
    ConditionalPoisson.zero = ConditionalPoisson(tmp)
    ConditionalPoisson.one = ConditionalPoisson(np.ones((K)))

    ConditionalPoisson.idempotent = False
    ConditionalPoisson.cancellative = False

    return ConditionalPoisson
