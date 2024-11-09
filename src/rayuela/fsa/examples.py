from rayuela.base.semiring import Real, Boolean, Tropical, String
from rayuela.base.symbol import Sym
from rayuela.base.state import State
from rayuela.fsa.fsa import FSA


class NGram:
    def ngram(self, V, n, R=Real, BOS="#"):
        from rayuela.fsa.fsa import FSA
        from itertools import product

        self.V, self.n = V, n
        fsa = FSA(R)
        for ngram in product(self.V.union(set([BOS])), repeat=n):
            # filter out BOS issues
            bad = False
            for i, g in enumerate(ngram):
                if i > 0 and g == BOS and ngram[i - 1] != BOS:
                    bad = True
            if bad:
                continue

            prefix = ngram[:-1]
            suffix = ngram[1:]

            def shorten(input):
                new = []
                for i, g in enumerate(input):
                    if i > 0 and g == BOS and prefix[i - 1] == BOS:
                        continue
                    else:
                        new.append(g)
                return tuple(new)

            prefix = shorten(prefix)
            suffix = shorten(suffix)

            if len(suffix) == 1:
                continue

            print(prefix, "-->", suffix)
            # q, p = State()


class MohriExamples:
    ###########################################################################
    #### From link.springer.com/content/pdf/10.1007/978-3-642-01492-5_6.pdf
    ###########################################################################

    def example8a():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Real)

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(0.1))

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[2], State(0), w=Real(0.2))
        fsa.add_arc(State(1), Sigma[2], State(2), w=Real(0.3))
        fsa.add_arc(State(1), Sigma[2], State(3), w=Real(0.4))

        # arcs from state 2
        fsa.add_arc(State(2), Sigma[2], State(3), w=Real(0.5))

        # arcs from state 3
        fsa.add_arc(State(3), Sigma[1], State(3), w=Real(0.6))

        # add final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=fsa.R(0.7))

        return fsa

    def example8b():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Real)
        fsa.set_I(State(0), w=Real(1.0))

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(0.1))

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[2], State(1), w=Real(0.2))
        fsa.add_arc(State(1), Sigma[1], State(2), w=Real(0.3))
        fsa.add_arc(State(1), Sigma[1], State(3), w=Real(0.4))

        # arcs from state 2
        fsa.add_arc(State(2), Sigma[2], State(3), w=Real(0.5))

        # add final states
        fsa.add_F(State(3), w=Real(0.6))

        return fsa

    def example8c():
        """
        Intersection of 8a and 8b
        """

        Sigma = {0: "eps", 1: "a", 2: "b"}

        states = [
            State(0, label=(0, 0)),
            State(1, label=(1, 1)),
            State(2, label=(0, 1)),
            State(3, label=(2, 1)),
            State(4, label=(3, 1)),
            State(5, label=(3, 2)),
            State(6, label=(3, 3)),
        ]

        fsa = FSA(R=Real)
        fsa.set_I(states[0], w=Real(1.0))

        # arcs from state 0, 0
        fsa.add_arc(states[0], Sigma[2], states[1], w=Real(0.01))

        # arcs from state 1, 1
        fsa.add_arc(states[1], Sigma[2], states[2], w=Real(0.04))
        fsa.add_arc(states[1], Sigma[2], states[3], w=Real(0.06))
        fsa.add_arc(states[1], Sigma[2], states[4], w=Real(0.08))

        # arcs from state 0, 1
        fsa.add_arc(states[2], Sigma[2], states[1], w=Real(0.02))

        # arcs from state 2, 1
        fsa.add_arc(states[3], Sigma[2], states[4], w=Real(0.1))

        # arcs from state 3, 1
        fsa.add_arc(states[4], Sigma[1], states[5], w=Real(0.18))
        fsa.add_arc(states[4], Sigma[1], states[6], w=Real(0.24))

        # add final states
        fsa.add_F(states[6], w=Real(0.42))

        return fsa

    def example9a():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Boolean)
        fsa.set_I(State(0), w=fsa.R.one)

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[1], State(1), w=fsa.R.one)

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[2], State(2), w=fsa.R.one)

        # arcs from state 2
        fsa.add_arc(State(2), Sigma[1], State(3), w=fsa.R.one)
        fsa.add_arc(State(2), Sigma[2], State(3), w=fsa.R.one)

        # arcs from state 3
        fsa.add_arc(State(3), Sigma[2], State(1), w=fsa.R.one)

        # add final states
        fsa.add_F(State(3), w=fsa.R.one)

        return fsa

    def example9b():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Real)

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[1], State(1), w=fsa.R.one)
        fsa.add_arc(State(0), Sigma[2], State(4), w=fsa.R.one)

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[2], State(2), w=fsa.R.one)
        fsa.add_arc(State(1), Sigma[1], State(4), w=fsa.R.one)

        # arcs from state 2
        fsa.add_arc(State(2), Sigma[1], State(3), w=fsa.R.one)
        fsa.add_arc(State(2), Sigma[2], State(3), w=fsa.R.one)

        # arcs from state 3
        fsa.add_arc(State(3), Sigma[2], State(1), w=fsa.R.one)
        fsa.add_arc(State(3), Sigma[1], State(4), w=fsa.R.one)

        # arcs from state 4
        fsa.add_arc(State(4), Sigma[1], State(4), w=fsa.R.one)
        fsa.add_arc(State(4), Sigma[2], State(4), w=fsa.R.one)

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=fsa.R.one)

        return fsa

    def example9c():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Real)

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[1], State(1), w=fsa.R.one)
        fsa.add_arc(State(0), Sigma[2], State(4), w=fsa.R.one)

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[2], State(2), w=fsa.R.one)
        fsa.add_arc(State(1), Sigma[1], State(4), w=fsa.R.one)

        # arcs from state 2
        fsa.add_arc(State(2), Sigma[1], State(3), w=fsa.R.one)
        fsa.add_arc(State(2), Sigma[2], State(3), w=fsa.R.one)

        # arcs from state 3
        fsa.add_arc(State(3), Sigma[2], State(1), w=fsa.R.one)
        fsa.add_arc(State(3), Sigma[1], State(4), w=fsa.R.one)

        # arcs from state 4
        fsa.add_arc(State(4), Sigma[1], State(4), w=fsa.R.one)
        fsa.add_arc(State(4), Sigma[2], State(4), w=fsa.R.one)

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(0), w=fsa.R.one)
        fsa.add_F(State(1), w=fsa.R.one)
        fsa.add_F(State(2), w=fsa.R.one)
        fsa.add_F(State(4), w=fsa.R.one)

        return fsa

    def example9d():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(R=Real)

        # arcs from state 0
        fsa.add_arc(State(0), Sigma[1], State(0), w=fsa.R(0.1))
        fsa.add_arc(State(0), Sigma[2], State(1), w=fsa.R(0.2))

        # arcs from state 1
        fsa.add_arc(State(1), Sigma[1], State(0), w=fsa.R(0.3))

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(1), w=fsa.R(0.5))

        return fsa

    def example9e():
        Sigma = {0: "eps", 1: "a", 2: "b"}

        states = [
            State(0, label=(0, 0)),
            State(1, label=(0, 1)),
            State(2, label=(1, 2)),
            State(3, label=(0, 3)),
            State(4, label=(1, 1)),
            State(5, label=(0, 4)),
            State(6, label=(1, 4)),
        ]

        fsa = FSA(R=Real)

        fsa.add_arc(states[0], Sigma[1], states[1], w=Real(0.1))
        fsa.add_arc(states[0], Sigma[2], states[6], w=Real(0.2))

        fsa.add_arc(states[1], Sigma[2], states[2], w=Real(0.2))
        fsa.add_arc(states[1], Sigma[1], states[5], w=Real(0.1))

        fsa.add_arc(states[2], Sigma[1], states[3], w=Real(0.3))

        fsa.add_arc(states[3], Sigma[2], states[4], w=Real(0.2))
        fsa.add_arc(states[3], Sigma[1], states[5], w=Real(0.1))

        fsa.add_arc(states[4], Sigma[1], states[5], w=Real(0.3))

        fsa.add_arc(states[5], Sigma[1], states[5], w=Real(0.1))
        fsa.add_arc(states[5], Sigma[2], states[6], w=Real(0.2))

        fsa.add_arc(states[6], Sigma[1], states[5], w=Real(0.3))

        # add initial & final states
        fsa.set_I(states[0], w=fsa.R.one)
        fsa.add_F(states[2], w=fsa.R(0.5))
        fsa.add_F(states[4], w=fsa.R(0.5))
        fsa.add_F(states[6], w=fsa.R(0.5))

        return fsa

    def example11a():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(1))
        fsa.add_arc(State(0), Sigma[1], State(2), w=Real(2))

        fsa.add_arc(State(1), Sigma[2], State(1), w=Real(3))
        fsa.add_arc(State(1), Sigma[3], State(3), w=Real(5))

        fsa.add_arc(State(2), Sigma[2], State(2), w=Real(3))
        fsa.add_arc(State(2), Sigma[4], State(3), w=Real(6))

        # add initial and final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=Real(0))

        return fsa

    def example11b():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d"}

        states = [
            State(0, label=(0, 0)),
            State(1, label=((1, 0), (2, 1))),
            State(2, label=(3, 0)),
        ]

        fsa = FSA(R=Real)

        fsa.add_arc(states[0], Sigma[1], states[1], w=Real(1))

        fsa.add_arc(states[1], Sigma[2], states[1], w=Real(3))
        fsa.add_arc(states[1], Sigma[3], states[2], w=Real(5))
        fsa.add_arc(states[1], Sigma[4], states[2], w=Real(7))

        # add initial & final states
        fsa.set_I(states[0], w=fsa.R.one)
        fsa.add_F(states[2], w=fsa.R.zero)

        return fsa

    def example11c():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(1))
        fsa.add_arc(State(0), Sigma[1], State(2), w=Real(2))

        fsa.add_arc(State(1), Sigma[2], State(1), w=Real(3))
        fsa.add_arc(State(1), Sigma[3], State(3), w=Real(5))

        fsa.add_arc(State(2), Sigma[2], State(2), w=Real(4))
        fsa.add_arc(State(2), Sigma[4], State(3), w=Real(6))

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=fsa.R.zero)

        return fsa

    def example12a():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(0))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(1))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(5))
        fsa.add_arc(State(0), Sigma[4], State(2), w=Real(0))
        fsa.add_arc(State(0), Sigma[5], State(2), w=Real(1))

        fsa.add_arc(State(1), Sigma[5], State(3), w=Real(0))
        fsa.add_arc(State(1), Sigma[6], State(3), w=Real(1))

        fsa.add_arc(State(2), Sigma[5], State(3), w=Real(4))
        fsa.add_arc(State(2), Sigma[6], State(3), w=Real(5))

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=fsa.R.one)

        return fsa

    def example12b():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(0))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(1))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(5))
        fsa.add_arc(State(0), Sigma[4], State(2), w=Real(4))
        fsa.add_arc(State(0), Sigma[5], State(2), w=Real(5))

        fsa.add_arc(State(1), Sigma[5], State(3), w=Real(0))
        fsa.add_arc(State(1), Sigma[6], State(3), w=Real(1))

        fsa.add_arc(State(2), Sigma[5], State(3), w=Real(0))
        fsa.add_arc(State(2), Sigma[6], State(3), w=Real(1))

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.zero)
        fsa.add_F(State(3), w=fsa.R.zero)

        return fsa

    def example12c():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(0 / 15))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(1 / 15))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(5 / 15))
        fsa.add_arc(State(0), Sigma[4], State(2), w=Real(0 / 15))
        fsa.add_arc(State(0), Sigma[5], State(2), w=Real(9 / 15))

        fsa.add_arc(State(1), Sigma[5], State(3), w=Real(0))
        fsa.add_arc(State(1), Sigma[6], State(3), w=Real(1))

        fsa.add_arc(State(2), Sigma[5], State(3), w=Real(4 / 9))
        fsa.add_arc(State(2), Sigma[6], State(3), w=Real(5 / 9))

        # add initial & final states
        fsa.set_I(State(0), w=Real(15))
        fsa.add_F(State(3), w=Real(1))

        return fsa

    def example12d():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(0))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(1))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(5))

        fsa.add_arc(State(1), Sigma[5], State(3), w=Real(0))
        fsa.add_arc(State(1), Sigma[6], State(3), w=Real(1))

        # add initial & final states
        fsa.set_I(State(0), w=fsa.R.zero)
        fsa.add_F(State(3), w=fsa.R.zero)

        return fsa

    def example13a(self, semiring=Real):
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=semiring)

        fsa.add_arc(State(0), Sigma[1], State(1), w=semiring(1))
        fsa.add_arc(State(0), Sigma[2], State(1), w=semiring(2))
        fsa.add_arc(State(0), Sigma[3], State(1), w=semiring(3))
        fsa.add_arc(State(0), Sigma[4], State(2), w=semiring(4))
        fsa.add_arc(State(0), Sigma[5], State(2), w=semiring(5))

        fsa.add_arc(State(1), Sigma[5], State(3), w=semiring(0.8))
        fsa.add_arc(State(1), Sigma[6], State(3), w=semiring(1))

        fsa.add_arc(State(2), Sigma[5], State(3), w=semiring(4))
        fsa.add_arc(State(2), Sigma[6], State(3), w=semiring(5))

        # add final states
        fsa.set_I(State(0), w=fsa.R.one)
        fsa.add_F(State(3), w=fsa.R.one)

        return fsa

    def example13b(self, semiring=Real):
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(0.04))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(0.08))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(0.12))
        fsa.add_arc(State(0), Sigma[4], State(1), w=Real(0.9))
        fsa.add_arc(State(0), Sigma[5], State(1), w=Real(1))

        fsa.add_arc(State(1), Sigma[5], State(2), w=Real(0.8))
        fsa.add_arc(State(1), Sigma[6], State(2), w=Real(1))

        # add initial & final states
        fsa.set_I(State(0), w=Real(25))
        fsa.add_F(State(2), w=Real(1))

        return fsa

    def example13c():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(R=Real)

        fsa.add_arc(State(0), Sigma[1], State(1), w=Real(1 / 51))
        fsa.add_arc(State(0), Sigma[2], State(1), w=Real(2 / 51))
        fsa.add_arc(State(0), Sigma[3], State(1), w=Real(3 / 51))
        fsa.add_arc(State(0), Sigma[4], State(1), w=Real(20 / 51))
        fsa.add_arc(State(0), Sigma[5], State(1), w=Real(25 / 51))

        fsa.add_arc(State(1), Sigma[5], State(2), w=Real(4 / 9))
        fsa.add_arc(State(1), Sigma[6], State(2), w=Real(5 / 9))

        # add initial & final states
        fsa.set_I(State(0), w=Real(459 / 5))
        fsa.add_F(State(2), w=Real(1))

        return fsa


class MinimizationExamples:
    def example1():
        """
        Example from upper right corner of https://en.wikipedia.org/wiki/DFA_minimization
        """

        F = FSA(Boolean)

        F.set_I(State("a"), F.R.one)

        # arcs from state a
        F.add_arc(State("a"), "0", State("b"), F.R(True))
        F.add_arc(State("a"), "1", State("c"), F.R(True))

        # arcs from state b
        F.add_arc(State("b"), "1", State("d"), F.R(True))
        F.add_arc(State("b"), "0", State("a"), F.R(True))

        # arcs from state c
        F.add_arc(State("c"), "1", State("f"), F.R(True))
        F.add_arc(State("c"), "0", State("e"), F.R(True))

        # arcs from state d
        F.add_arc(State("d"), "1", State("f"), F.R(True))
        F.add_arc(State("d"), "0", State("e"), F.R(True))

        # arcs from state f
        F.add_arc(State("f"), "1", State("f"), F.R(True))
        F.add_arc(State("f"), "0", State("f"), F.R(True))

        # arcs from state e
        F.add_arc(State("e"), "0", State("e"), F.R(True))
        F.add_arc(State("e"), "1", State("f"), F.R(True))

        # add final states
        F.add_F(State("c"), F.R.one)
        F.add_F(State("d"), F.R.one)
        F.add_F(State("e"), F.R.one)

        return F

    def example1_solution():
        """
        Example from upper right corner of https://en.wikipedia.org/wiki/DFA_minimization
        """

        Sigma = {0: "eps", 1: "0", 2: "1"}

        F = FSA(Boolean)

        F.set_I(State("a,b"), F.R.one)

        F.add_arc(State("a,b"), "0", State("a,b"), F.R(True))
        F.add_arc(State("a,b"), "1", State("c,d,e"), F.R(True))

        F.add_arc(State("c,d,e"), "0", State("c,d,e"), F.R(True))
        F.add_arc(State("c,d,e"), "1", State("f"), F.R(True))

        F.add_arc(State("f"), "0", State("f"), F.R(True))
        F.add_arc(State("f"), "1", State("f"), F.R(True))

        # add final states
        F.add_F(State("c,d,e"), F.R.one)

        return F

    def example2():
        """
        Test case from Hopcroft and Ullman (1979) Figure 3.5
        """

        Sigma = {0: "eps", 1: "0", 2: "1"}

        fsa = FSA(Sigma, semiring=Boolean)
        fsa.set_initial(State("a"), weight=Boolea(True))

        fsa.add_arc(State("a"), 1, State("b"), weight=Boolean(True))
        fsa.add_arc(State("a"), 1, State("b"), weight=Boolean(True))

        fsa.add_final(State("b'"), weight=Boolean(True))

        return fsa

    def example3():
        """
        Test case from Revuz (1991)
        """

        fsa = FSA(Boolean)
        fsa.set_I(State(1), Boolean(True))

        fsa.add_arc(State(1), "a", State(2), Boolean(True))
        fsa.add_arc(State(1), "b", State(3), Boolean(True))
        fsa.add_arc(State(1), "c", State(4), Boolean(True))

        fsa.add_arc(State(2), "a", State(5), Boolean(True))
        fsa.add_arc(State(2), "b", State(6), Boolean(True))

        fsa.add_arc(State(3), "a", State(10), Boolean(True))
        fsa.add_arc(State(3), "b", State(7), Boolean(True))

        fsa.add_arc(State(4), "a", State(8), Boolean(True))
        fsa.add_arc(State(4), "b", State(9), Boolean(True))

        fsa.add_arc(State(5), "a", State(15), Boolean(True))
        fsa.add_arc(State(5), "b", State(10), Boolean(True))

        fsa.add_arc(State(6), "a", State(10), Boolean(True))
        fsa.add_arc(State(6), "b", State(11), Boolean(True))

        fsa.add_arc(State(7), "a", State(10), Boolean(True))
        fsa.add_arc(State(7), "b", State(11), Boolean(True))

        fsa.add_arc(State(8), "a", State(12), Boolean(True))

        fsa.add_arc(State(9), "a", State(12), Boolean(True))
        fsa.add_arc(State(9), "b", State(15), Boolean(True))

        fsa.add_arc(State(10), "a", State(13), Boolean(True))
        fsa.add_arc(State(10), "b", State(15), Boolean(True))

        fsa.add_arc(State(11), "a", State(13), Boolean(True))

        fsa.add_arc(State(12), "a", State(14), Boolean(True))
        fsa.add_arc(State(12), "c", State(15), Boolean(True))

        fsa.add_arc(State(13), "b", State(15), Boolean(True))

        fsa.add_arc(State(14), "d", State(15), Boolean(True))

        fsa.add_F(State(15), Boolean(True))

        return fsa

    def example4():
        """
        Modified test case from Revuz (1991)
        """

        fsa = FSA(Boolean)
        fsa.set_I(State(1), Boolean(True))

        fsa.add_arc(State(1), "a", State(2), Boolean(True))
        fsa.add_arc(State(1), "b", State(3), Boolean(True))
        fsa.add_arc(State(1), "c", State(4), Boolean(True))

        fsa.add_arc(State(2), "a", State(5), Boolean(True))
        fsa.add_arc(State(2), "b", State(6), Boolean(True))

        fsa.add_arc(State(3), "a", State(10), Boolean(True))
        fsa.add_arc(State(3), "b", State(7), Boolean(True))

        fsa.add_arc(State(4), "a", State(8), Boolean(True))
        fsa.add_arc(State(4), "b", State(9), Boolean(True))

        # fsa.add_arc(State(5), "a", State(15), Boolean(True))
        fsa.add_arc(State(5), "b", State(10), Boolean(True))

        # fsa.add_arc(State(6), "a", State(10), Boolean(True))
        fsa.add_arc(State(6), "b", State(11), Boolean(True))

        fsa.add_arc(State(7), "a", State(10), Boolean(True))
        fsa.add_arc(State(7), "b", State(11), Boolean(True))

        fsa.add_arc(State(8), "a", State(12), Boolean(True))

        fsa.add_arc(State(9), "a", State(12), Boolean(True))

        fsa.add_arc(State(10), "a", State(13), Boolean(True))
        fsa.add_arc(State(10), "b", State(15), Boolean(True))

        fsa.add_arc(State(11), "a", State(13), Boolean(True))
        fsa.add_arc(State(11), "b", State(15), Boolean(True))

        fsa.add_arc(State(12), "a", State(14), Boolean(True))
        fsa.add_arc(State(12), "c", State(15), Boolean(True))

        fsa.add_arc(State(13), "b", State(15), Boolean(True))

        fsa.add_arc(State(14), "d", State(15), Boolean(True))

        fsa.add_F(State(15), Boolean(True))

        return fsa


class SCCExamples:
    def example1():
        F = FSA(R=Real)
        F.set_I(State(0), F.R(1 / 2))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(1 / 5.0))

        F.add_arc(State(1), Sym("b"), State(1), F.R(1 / 8.0))
        F.add_arc(State(1), Sym("c"), State(3), F.R(1 / 10.0))

        F.add_arc(State(2), Sym("d"), State(3), F.R(1 / 5.0))

        F.add_arc(State(3), Sym("b"), State(2), F.R(1 / 10.0))

        F.add_F(State(3), F.R(1 / 4))

        return F

    def example2():
        F = FSA(R=Real)
        F.set_I(State(0), F.R(1 / 2))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(1 / 5.0))

        F.add_arc(State(1), Sym("c"), State(3), F.R(1 / 10.0))
        F.add_arc(State(1), Sym("c"), State(4), F.R(1 / 10.0))

        F.add_arc(State(2), Sym("d"), State(3), F.R(1 / 5.0))
        F.add_arc(State(2), Sym("c"), State(5), F.R(1 / 10.0))

        F.add_arc(State(3), Sym("b"), State(2), F.R(1 / 10.0))

        F.add_arc(State(4), Sym("c"), State(1), F.R(1 / 10.0))

        F.add_arc(State(5), Sym("c"), State(3), F.R(1 / 10.0))

        F.add_F(State(3), F.R(1 / 4))

        return F

    def example3():
        F = FSA(R=Real)
        F.set_I(State(0), F.R(1 / 2))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(1 / 5.0))

        F.add_arc(State(1), Sym("b"), State(1), F.R(1 / 8.0))
        F.add_arc(State(1), Sym("c"), State(2), F.R(1 / 10.0))

        F.add_arc(State(2), Sym("d"), State(2), F.R(1 / 5.0))

        F.add_F(State(2), F.R(1 / 4))

        return F

    def example4():
        F = FSA(R=Real)
        F.set_I(State(0), F.R(1 / 2))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("b"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("c"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(1 / 5.0))

        F.add_arc(State(1), Sym("c"), State(1), F.R(1 / 10.0))
        F.add_arc(State(1), Sym("d"), State(1), F.R(1 / 10.0))
        F.add_arc(State(1), Sym("c"), State(3), F.R(1 / 10.0))
        F.add_arc(State(1), Sym("c"), State(4), F.R(1 / 10.0))

        F.add_arc(State(2), Sym("d"), State(3), F.R(1 / 5.0))
        F.add_arc(State(2), Sym("c"), State(5), F.R(1 / 10.0))

        F.add_arc(State(3), Sym("b"), State(2), F.R(1 / 10.0))

        F.add_arc(State(4), Sym("c"), State(1), F.R(1 / 10.0))

        F.add_arc(State(5), Sym("c"), State(3), F.R(1 / 10.0))

        F.add_F(State(3), F.R(1 / 4))

        return F

    def example5():
        F = FSA(R=Boolean)
        F.set_I(State(0), F.R(True))

        F.add_arc(State(0), Sym("a"), State(1), F.R(True))
        F.add_arc(State(0), Sym("a"), State(2), F.R(True))

        F.add_arc(State(1), Sym("b"), State(1), F.R(True))
        F.add_arc(State(1), Sym("c"), State(3), F.R(True))

        F.add_arc(State(2), Sym("d"), State(3), F.R(True))

        F.add_arc(State(3), Sym("b"), State(2), F.R(True))

        F.add_F(State(3), F.R(True))

        return F

    def example6():
        F = FSA(R=Real)
        F.set_I(State(0), F.R(1))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1 / 2.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(1 / 5.0))

        F.add_arc(State(1), Sym("b"), State(1), F.R(1 / 8.0))
        F.add_arc(State(1), Sym("c"), State(3), F.R(1 / 10.0))

        F.add_arc(State(2), Sym("d"), State(3), F.R(1 / 5.0))

        F.add_arc(State(3), Sym("b"), State(2), F.R(1 / 10.0))

        F.add_F(State(3), F.R(1))

        return F


class DeterminizationExamples:
    def example1():
        """
        Test case from Allauzen and Mohri (2003) Figure 1
        https://cs.nyu.edu/~mohri/pub/twins.pdf
        """

        F = FSA(R=Tropical)
        F.set_I(State(0), F.R(1.0))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(2.0))

        F.add_arc(State(1), Sym("b"), State(1), F.R(3.0))
        F.add_arc(State(1), Sym("c"), State(3), F.R(5.0))

        F.add_arc(State(2), Sym("b"), State(2), F.R(3.0))
        F.add_arc(State(2), Sym("d"), State(3), F.R(6.0))

        F.add_F(State(3), F.R.one)

        return F

    def example1_solution():
        """
        Test case from Allauzen and Mohri (2003) Figure 1
        https://cs.nyu.edu/~mohri/pub/twins.pdf
        """

        fsa = FSA(R=Tropical)

        states = [
            PowerState(
                [State(0), State(0)], weights={State(0): fsa.R.one, State(0): fsa.R.one}
            ),
            PowerState(
                [State(1), State(0)], weights={State(1): fsa.R.one, State(0): fsa.R.one}
            ),
            PowerState(
                [State(3), State(0)], weights={State(3): fsa.R.one, State(0): fsa.R.one}
            ),
        ]

        fsa.set_I(states[0], fsa.R(1.0))

        fsa.add_arc(states[0], "a", states[1], fsa.R(1.0))
        fsa.add_arc(states[1], "b", states[1], fsa.R(3.0))

        fsa.add_arc(states[1], "c", states[2], fsa.R(5.0))
        fsa.add_arc(states[1], "d", states[2], fsa.R(7.0))

        fsa.add_F(states[2], fsa.R.one)

        return fsa

    def example2():
        """
        Test case from Allauzen and Mohri (2003) Figure 2
        https://cs.nyu.edu/~mohri/pub/twins.pdf
        """

        Sigma = {0: "eps", 1: "x", 2: "y", 3: "z"}

        fsa = FSA(R=String)
        fsa.add_I(State(0), w=fsa.R.one)

        fsa.add_arc(State(0), Sym("1"), State(1), w=fsa.R("a"))
        fsa.add_arc(State(0), Sym("1"), State(2), w=fsa.R(""))

        fsa.add_arc(State(1), Sym("2"), State(3), w=fsa.R("b"))
        fsa.add_arc(State(1), Sym("1"), State(5), w=fsa.R("b"))

        fsa.add_arc(State(2), Sym("3"), State(5), w=fsa.R("a"))
        fsa.add_arc(State(2), Sym("2"), State(4), w=fsa.R("a"))

        fsa.add_arc(State(3), Sym("2"), State(1), w=fsa.R("a"))

        fsa.add_arc(State(4), Sym("2"), State(2), w=fsa.R("b"))

        fsa.add_F(State(5), w=fsa.R.one)

        return fsa

    def example3():
        """
        Test case from Allauzen and Mohri (2003) Figure 1
        https://cs.nyu.edu/~mohri/pub/twins.pdf
        """

        F = FSA(R=Tropical)
        F.set_I(State(0), F.R(1.0))

        F.add_arc(State(0), Sym("a"), State(1), F.R(1.0))
        F.add_arc(State(0), Sym("a"), State(2), F.R(2.0))

        F.add_arc(State(1), Sym("b"), State(1), F.R(3.0))
        F.add_arc(State(1), Sym("c"), State(3), F.R(5.0))

        F.add_arc(State(2), Sym("b"), State(2), F.R(4.0))
        F.add_arc(State(2), Sym("d"), State(3), F.R(6.0))

        F.add_F(State(3), F.R.one)

        return F


class PushingExamples:
    def example1():
        """
        Figure 1 from Riley and Mohri (2001)
        https://www.isca-speech.org/archive/pdfs/eurospeech_2001/mohri01_eurospeech.pdf
        """

        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"}

        fsa = FSA(Sigma, semiring=Tropical, initial=State(0))

        # arcs from state 0
        fsa.add_arc(State(0), 1, State(1), weight=Tropical(0.0))
        fsa.add_arc(State(0), 2, State(1), weight=Tropical(1.0))
        fsa.add_arc(State(0), 3, State(1), weight=Tropical(4.0))
        fsa.add_arc(State(0), 4, State(2), weight=Tropical(0.0))
        fsa.add_arc(State(0), 5, State(2), weight=Tropical(1.0))

        # arcs from state 1
        fsa.add_arc(State(1), 5, State(3), weight=Tropical(0.0))
        fsa.add_arc(State(1), 6, State(3), weight=Tropical(1.0))

        # arcs from state 2
        fsa.add_arc(State(2), 5, State(3), weight=Tropical(10.0))
        fsa.add_arc(State(2), 6, State(3), weight=Tropical(11.0))

        # add final
        fsa.add_final(State(3, weight=Tropical(0.0)))

        return fsa


class IntersectionExamples:
    """
    From http://www.cs.um.edu.mt/gordon.pace/Research/Software/Relic/Transformations/FSA/intersection.html
    """

    def example1(self):
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(Sigma, semiring=Boolean)

        fsa.add_arc(State(0), 1, State(1))
        fsa.add_arc(State(1), 1, State(0))

        fsa.set_initial(State(0), weight=None)
        fsa.add_final(State(1), weight=None)

        return fsa

    def example2(self):
        Sigma = {0: "eps", 1: "a", 2: "b"}

        fsa = FSA(Sigma, semiring=Boolean)

        fsa.add_arc(State(0), 1, State(1))
        fsa.add_arc(State(1), 1, State(1))
        fsa.add_arc(State(1), 2, State(2))
        fsa.add_arc(State(2), 2, State(2))

        fsa.set_initial(State(0), weight=None)
        fsa.add_final(State(2), weight=None)

        return fsa


class TwinsExamples:
    """
    From https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.9.3166&rep=rep1&type=pdf
    Figure 3 a
    """

    def example1():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d"}

        fsa = FSA(Sigma, semiring=Real)

        fsa.add_arc(State(0), 1, State(1), weight=fsa.semiring(1))
        fsa.add_arc(State(0), 1, State(2), weight=fsa.semiring(2))
        fsa.add_arc(State(1), 2, State(1), weight=fsa.semiring(3))
        fsa.add_arc(State(2), 2, State(2), weight=fsa.semiring(3))
        fsa.add_arc(State(1), 3, State(3), weight=fsa.semiring(5))
        fsa.add_arc(State(2), 4, State(3), weight=fsa.semiring(6))

        fsa.set_initial(State(0), weight=fsa.semiring.one)
        fsa.add_final(State(3), weight=fsa.semiring.zero)

        return fsa

    def example2():
        Sigma = {0: "eps", 1: "a", 2: "b", 3: "c", 4: "d"}

        fsa = FSA(Sigma, semiring=Real)

        fsa.add_arc(State(0), 1, State(1), weight=fsa.semiring(1))
        fsa.add_arc(State(0), 1, State(2), weight=fsa.semiring(2))
        fsa.add_arc(State(1), 2, State(1), weight=fsa.semiring(3))
        fsa.add_arc(State(2), 2, State(2), weight=fsa.semiring(4))
        fsa.add_arc(State(1), 3, State(3), weight=fsa.semiring(5))
        fsa.add_arc(State(2), 4, State(3), weight=fsa.semiring(6))

        fsa.set_initial(State(0), weight=fsa.semiring.one)
        fsa.add_final(State(3), weight=fsa.semiring.zero)

        return fsa


class NotesExamples:
    def example_1():
        F = FSA()

        F.add_arc(State(1), Sym("a"), State(2), F.R.one)
        F.add_arc(State(1), Sym("b"), State(3), F.R.one)
        F.add_arc(State(2), Sym("c"), State(4), F.R.one)
        F.add_arc(State(2), Sym("b"), State(2), F.R.one)
        F.add_arc(State(3), Sym("c"), State(4), F.R.one)
        F.add_arc(State(3), Sym("b"), State(5), F.R.one)
        F.add_arc(State(4), Sym("a"), State(6), F.R.one)
        F.add_arc(State(5), Sym("a"), State(6), F.R.one)

        F.add_I(State(1), F.R.one)
        F.add_F(State(6), F.R.one)

        return F


class EditDistanceExamples:

    def string_fsa(s, Sigma):
        """
        Given an input string, constructs a chain-like edit-distance
        WFSA in the Tropical semiring
        """
        from rayuela.base.symbol import ε

        F = FSA(R=Tropical)
        str_to_sym = [Sym(x) for x in s]

        for i in range(len(s)):
            for a in Sigma:
                F.add_arc(State(i), a, State(i), Tropical(1.0))
                if a != str_to_sym[i]:
                    F.add_arc(State(i), a, State(i+1), Tropical(1.0))
            F.add_arc(State(i), str_to_sym[i], State(i + 1), Tropical(0.0))
            F.add_arc(State(i), ε, State(i + 1), Tropical(1.0))

        for a in Sigma:
            F.add_arc(State(len(s)), a, State(len(s)), Tropical(1.0))

        F.set_I(State(0), F.R.one)
        F.set_F(State(len(s)), F.R.one)

        return F

    def to_tropical(fsa):
        """
        Lifts an input fsa to the Tropical semiring, i.e.,
        constructs an FSA that assigns weight 0 to all strings
        in the language of the original automaton and ∞ to all
        other strings
        """
        F = FSA(R=Tropical)

        for q, w in fsa.I:
            F.set_I(q, F.R.one)

        for q, w in fsa.F:
            F.set_F(q, F.R.one)

        for q in fsa.Q:
            for a, p, w in fsa.arcs(q):
                F.add_arc(q, a, p, F.R.one)

        return F

    def edit_distance(fsa, s):
        """
        Given an input string and an FSA, computes the minimum
        edit distance between the input string and any other
        string in the language of the input FSA
        """
        from rayuela.fsa.pathsum import Pathsum

        str_fsa = EditDistanceExamples.string_fsa(s, fsa.Sigma)
        tropical_fsa = EditDistanceExamples.to_tropical(fsa)

        i = str_fsa.intersect(tropical_fsa)
        pathsum = Pathsum(i)
        pathsum_res = pathsum.lehmann()

        return pathsum.allpairs_pathsum(pathsum_res)




