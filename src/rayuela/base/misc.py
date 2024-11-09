import string
from fractions import Fraction
from math import sqrt

import numpy as np

from rayuela.base.semiring import (
    Boolean,
    Count,
    Integer,
    MaxPlus,
    Rational,
    Real,
    String,
    Tropical,
)


def spans(min, max, depth, span=()):
    for n in range(min + 1, max):
        if depth == 0:
            yield span + (n,)
        else:
            for x in spans(n, max, depth - 1, span + (n,)):
                yield x


def lcp(str1, str2):
    """computes the longest common prefix"""
    prefix = ""
    for n in range(min(len(str1), len(str2))):
        if str1[n] == str2[n]:
            prefix += str1[n]
        else:
            break
    return prefix


def symify(string):
    from rayuela.base.symbol import Sym

    return [Sym(x) for x in list(string)]


def straight(string, R):
    from rayuela.base.symbol import Sym
    from rayuela.fsa.fsa import FSA
    from rayuela.base.state import State

    fsa = FSA(R=R)
    for i, x in enumerate(list(string)):
        fsa.add_arc(State(i), Sym(x), State(i + 1), R.one)
    fsa.set_I(State(0), R.one)
    fsa.add_F(State(len(string)), R.one)

    return fsa


def _random_weight(semiring, rng=None, **kwargs):  # noqa: C901
    if rng is None:
        rng = np.random.default_rng()
    if semiring is String:
        str_len = int(rng.random() * 8 + 1)
        return semiring(
            "".join(rng.choice(string.ascii_lowercase) for _ in range(str_len))
        )

    elif semiring is Boolean:
        return semiring(True)

    elif semiring is Real:
        tol = 1e-3
        s = kwargs.get("divide_by", 2)
        random_weight = round(rng.random() / s, 3)
        while random_weight < sqrt(tol):
            random_weight = round(rng.random() / s, 3)
        return semiring(random_weight)

    elif semiring is Rational:
        return semiring(Fraction(f"{rng.randint(1, 1)}/{rng.randint(10, 15)}"))

    elif semiring is Tropical:
        return semiring(rng.randint(0, 50))

    elif semiring is Integer:
        return semiring(rng.randint(1, 10))

    elif semiring is MaxPlus:
        return semiring(rng.randint(-10, -1))

    elif semiring is Count:
        return semiring(1.0)


def random_weight_negative(semiring, rng=None):
    from rayuela.base.semiring import Tropical
    if rng is None:
        rng = np.random.default_rng()

    if semiring is Tropical:
        return semiring(rng.randint(-50, 50))
    else:
        raise AssertionError("Unsupported Semiring")


def is_pathsum_positive(fsa):
    from rayuela.fsa.fsa import FSA
    from rayuela.fsa.fst import FST

    assert isinstance(fsa, FSA) or isinstance(fsa, FST)

    return fsa.pathsum() > fsa.R.zero


def filter_negative_pathsums(list_of_fsas):
    return [fsa for fsa in list_of_fsas if is_pathsum_positive(fsa)]


def compare_fsas(original_fsa, student_fsa) -> bool:
    from rayuela.fsa.fsa import FSA
    from rayuela.fsa.fst import FST

    assert isinstance(original_fsa, FSA) or isinstance(original_fsa, FST)
    assert isinstance(student_fsa, FSA) or isinstance(student_fsa, FST)

    if is_pathsum_positive(original_fsa):
        # TODO: Change check for: there is no an arbitrary number of initial states
        # same_number_initial_states = len(list(original_fsa.I)) == len(
        #     list(student_fsa.I)
        # )
        return np.allclose(
            float(original_fsa.pathsum()), float(student_fsa.pathsum()), atol=1e-3
        )  # and same_number_initial_states
        # --> This would break some correct implementations
    # Skip non-convergent pathsums
    return True


def is_topologically_sorted_scc(sccs, fsa, verbose=False):
    assert fsa.acyclic, "FSA must be acyclic to have a topological order"
    visited = []
    if verbose:
        print(sccs)
    for component in sccs:
        for p in component:
            visited.append(p)
            for _, q, w in fsa.arcs(p):
                if verbose:
                    print(p, "->", q)
                if q in visited:
                    return False

    return True


def same_number_of_arcs(fsa1, fsa2):
    n1 = sum([len(list(fsa1.arcs(q))) for q in fsa1.Q])
    n2 = sum([len(list(fsa2.arcs(q))) for q in fsa2.Q])

    return n1 == n2


def fsa_to_code(fsa, fsa_name: str, fst=False):  # noqa: C901
    """
    This function prints the code that produces the given fsa.
    Currently works for fsas with 'State's and 'MinimizeState's

    input:
    -----------
    - fsa: target fsa
    - fsa_name: variable name for the fsa
    """
    assert isinstance(fsa_name, str), "fsa_name must be a string"

    def _w(w):
        """
        Some semirings like Rational need to pass the weight between quotes
        """
        if fsa.R == Rational:
            return f"'{w}'"
        else:
            return w

    cls_name = "FST" if fst else "FSA"
    srng_cls = fsa.R.__name__
    print(f"{fsa_name} = {cls_name}({srng_cls})")

    for q in fsa.Q:
        state_cls = type(q).__name__
        if state_cls == "MinimizeState":
            org = f"{state_cls}({[f'State({i})' for i in q.idx]})".replace("'", "")
            for ll, p, w in fsa.arcs(q):
                tgt = f"{state_cls}({[f'State({i})' for i in p.idx]})".replace("'", "")
                print(f"{fsa_name}.add_arc({org}, '{ll}', {tgt}, {srng_cls}({_w(w)}))")
            for i, w in fsa.I:
                if q == i:
                    state = f"{[f'State({i})' for i in q.idx]}".replace("'", "")
                    print(
                        f"{fsa_name}.set_I({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )
            for i, w in fsa.F:
                if q == i:
                    state = f"{[f'State({i})' for i in q.idx]}".replace("'", "")
                    print(
                        f"{fsa_name}.add_F({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )
        if state_cls == "State":
            org = f"{state_cls}({q})".replace("'", "")
            for ll, p, w in fsa.arcs(q):
                tgt = f"{state_cls}({p})".replace("'", "")
                lbl = f"{ll}"
                if fst:
                    lbl = f"'{ll[0]}', '{ll[1]}'"
                print(f"{fsa_name}.add_arc({org}, {lbl}, {tgt}, {srng_cls}({_w(w)}))")
            for i, w in fsa.I:
                if q == i:
                    state = f"{q}".replace("'", "")
                    print(
                        f"{fsa_name}.set_I({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )
            for i, w in fsa.F:
                if q == i:
                    state = f"{q}".replace("'", "")
                    print(
                        f"{fsa_name}.add_F({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )
        if state_cls == "PairState":
            q1, q2 = q.state1, q.state2
            org = f"{state_cls}({q1}, {q2})".replace("'", "")
            for ll, p, w in fsa.arcs(q):
                p1, p2 = p.state1, p.state2
                tgt = f"{state_cls}({p1}, {p2})".replace("'", "")
                lbl = f"{ll}"
                if fst:
                    lbl = f"'{ll[0]}', '{ll[1]}'"
                print(f"{fsa_name}.add_arc({org}, {lbl}, {tgt}, {srng_cls}({_w(w)}))")
            for i, w in fsa.I:
                if q == i:
                    state = f"{q1},{q2}".replace("'", "")
                    print(
                        f"{fsa_name}.set_I({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )
            for i, w in fsa.F:
                if q == i:
                    state = f"{q1},{q2}".replace("'", "")
                    print(
                        f"{fsa_name}.add_F({state_cls}({state}), {srng_cls}({_w(w)}))"
                    )


def compare_charts(chart1, chart2) -> "tuple[bool,bool]":
    # Assert both have the same keys
    same_keys = set(chart1.keys()) == set(chart2.keys())

    # Assert all values are similar
    same_values = False
    if same_keys:
        same_values = all(
            [
                np.allclose(float(chart1[key]), float(chart2[key]), atol=1e-3)
                for key in chart1.keys()
            ]
        )

    return same_keys and same_values


def compare_chart(semiring, chart1, chart2):
    correct = True
    for item in set(chart1.keys()).intersection(set(chart2.keys())):
        if np.allclose(float(chart1[item]), float(chart2[item]), atol=1e-5):
            print("\t".join([colors.green % str(item), str(chart1[item])]))
        else:
            print(
                "\t".join(
                    [colors.light.red % str(item), str(chart1[item]), str(chart2[item])]
                )
            )
            correct = False
    for item in chart2:
        if item not in chart1:
            if chart2[item] != semiring.zero:
                print("\t".join([colors.yellow % str(item), str(chart2[item])]))
    return correct


def ansi(color=None, light=None, bg=3):
    return "\x1b[%s;%s%sm" % (light, bg, color)


_reset = "\x1b[0m"


def colorstring(s, c):
    return c + s + _reset


class colors:
    black, red, green, yellow, blue, magenta, cyan, white = [
        colorstring("%s", ansi(c, 0)) for c in range(8)
    ]

    class light:
        black, red, green, yellow, blue, magenta, cyan, white = [
            colorstring("%s", ansi(c, 1)) for c in range(8)
        ]

    class dark:
        black, red, green, yellow, blue, magenta, cyan, white = [
            colorstring("%s", ansi(c, 2)) for c in range(8)
        ]

    class bg:
        black, red, green, yellow, blue, magenta, cyan, white = [
            colorstring("%s", ansi(c, 0, bg=4)) for c in range(8)
        ]

    normal = "\x1b[0m%s\x1b[0m"
    bold = "\x1b[1m%s\x1b[0m"
    italic = "\x1b[3m%s\x1b[0m"
    underline = "\x1b[4m%s\x1b[0m"
    strike = "\x1b[9m%s\x1b[0m"
    # overline = lambda x: (u''.join(unicode(c) +
    # u'\u0305' for c in unicode(x))).encode('utf-8')

    leftarrow = "←"
    rightarrow = "→"
    reset = _reset
