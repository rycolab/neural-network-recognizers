import copy
from collections import defaultdict as dd

import numpy as np
import pandas as pd

from rayuela.base.semiring import Real
from rayuela.base.symbol import Sym
from rayuela.fsa.fsa import FSA, State
from rayuela.fsa.transformer import Transformer


class Hankel:
    def __init__(self, Σ=None) -> None:
        self.Σ = Σ
        self.matrix = dd(dict)
        self.S = set()
        self.P = set()
        self.PΣ = set()

    @property
    def Sigma(self):
        if self.Σ is not None:
            return self.Σ
        Σ = set()
        for x in self.S.union(self.P):
            for a in list(x):
                Σ.add(a)
        return Σ

    @property
    def complete(self):
        for x in self.P:
            for a in self.Sigma:
                a = a.value if isinstance(a, Sym) else a
                if x + a not in self.PΣ:
                    return False
        return True

    def set_val(self, x, y, val):
        self.P.add(x)
        self.S.add(y)
        self.matrix[x][y] = val

    def add_row(self, x, membership_query):
        assert x not in self.P, f'row "{x}" already present'
        for y in self.S:
            self.set_val(x, y, membership_query(x + y))

    def add_col(self, y, membership_query):
        assert y not in self.S, f'column "{x}" already present'
        for x in self.P:
            self.set_val(x, y, membership_query(x + y))

    def extend(self, w, membership_query):
        for x in list(self.P):
            if x + w not in self.P:
                self.add_row(x + w, membership_query)

    def complete(self, membership_query):
        for x in self.P:
            for a in self.Sigma:
                a = a.value if isinstance(a, Sym) else a
                self.PΣ.add(x + a)
                for y in self.S:
                    self.matrix[x + a][y] = membership_query(x + a + y)

    def get_submatrix(self, X, Y):
        return self.to_df(X, Y).to_numpy()

    def to_df(self, X=None, Y=None):
        if X is None:
            X = self.P.union(self.PΣ)
        if Y is None:
            Y = self.S
        df = pd.DataFrame.from_dict(self.matrix, orient="index", dtype=float)
        return pd.DataFrame(
            df,
            index=sorted(X, key=lambda x: len(x)),
            columns=sorted(Y, key=lambda x: len(x)),
        )

    def to_fsa(self, X, Y):
        assert self.complete, "Hankel block must be complete in order to create FSA."
        assert (
            "" in self.S
        ), "Hankel block needs to have ε column in order to create FSA."

        assert (
            set(X).difference(self.P) == set()
        ), f"Unknown rows in arg X: {set(X).difference(self.P)}"
        assert (
            set(Y).difference(self.S) == set()
        ), f"Unknown cols in arg Y : {set(Y).difference(self.S)}"

        Σ = self.Sigma
        n = len(X)
        λ = np.zeros(n)
        λ[0] = 1.0

        M = {}
        for a in Σ:
            a = a.value if isinstance(a, Sym) else a
            Xa = [x + a for x in X]
            M[Sym(a)] = np.dot(
                self.get_submatrix(Xa, Y), np.linalg.inv(self.get_submatrix(X, Y))
            )

        ρ = self.get_submatrix(X, [""]).flatten()

        fsa = Transformer.from_lin_alg(n, Σ, M, λ, ρ)
        return fsa

    def __str__(self):
        return str(self.to_df().rename(columns={"": "ε"}, index={"": "ε"}))

    def __repr__(self):
        return self.__str__()


class Oracle:
    def __init__(self, fsa):
        self.fsa = fsa
        self.F, self.V = Transformer.forward_basis(self.fsa, vocab=True)
        self.W = self.V.copy()

        U = self.V.copy()
        U2 = U.copy()
        while len(U2) > 0:
            U2 = []
            while len(U) > 0:
                u = U.pop()
                for v in self.V:
                    if v == "":
                        continue
                    w = u + v
                    if len(u) < fsa.num_states:
                        U2.append(w)
                        self.W.append(w)
            U = U2
        self.W = sorted(set(self.W), key=lambda x: len(x))

    def membership_query(self, string):
        return self.fsa.accept(string)

    def equivalence_query(self, other_fsa):
        for v in self.W[::-1]:
            if self.fsa.accept(v) != other_fsa.accept(v):
                return False, v
        return True, None


class FSA_Util:
    @staticmethod
    def example_fsa():
        fsa = FSA(Real)
        fsa.set_I(State("a"), Real(0.5))

        fsa.add_arc(State("a"), "a", State("d"), Real(0.125))
        fsa.add_arc(State("d"), "b", State("e"), Real(0.125))
        fsa.add_arc(State("e"), "a", State("e"), Real(0.25))

        fsa.add_F(State("d"), Real(0.25))
        fsa.add_F(State("e"), Real(0.5))

        return fsa

    @staticmethod
    def example_fsa2():
        fsa = FSA(Real)
        fsa.set_I(State("a"), Real(0.5))

        fsa.add_arc(State("a"), "a", State("b"), Real(0.125))
        fsa.add_arc(State("a"), "b", State("c"), Real(0.25))
        fsa.add_arc(State("b"), "a", State("a"), Real(0.25))
        fsa.add_arc(State("b"), "b", State("d"), Real(0.25))
        fsa.add_arc(State("c"), "a", State("e"), Real(0.125))
        fsa.add_arc(State("d"), "a", State("e"), Real(0.125))
        fsa.add_arc(State("e"), "a", State("e"), Real(0.25))

        fsa.add_F(State("c"), Real(0.02))
        fsa.add_F(State("d"), Real(0.25))
        fsa.add_F(State("e"), Real(0.5))
        return fsa

    @staticmethod
    def empty_fsa(Σ, λ):
        return Transformer.from_lin_alg(
            1, Σ, {a: np.zeros((1, 1)) for a in Σ}, np.ones(1), np.array([λ])
        )


class Beimel:
    def __init__(self, oracle, Σ):
        self.oracle = oracle
        self.X = set()
        self.Y = set()
        self.Σ = Σ
        self.h = Hankel(Σ)

    def correct(self, fsa, string):
        # fsa
        n, _, M, _, ρ = Transformer.to_lin_alg(fsa)
        W = np.eye(n)
        for a in list(string):
            a = Sym(a) if isinstance(a, str) else a
            W = W.dot(M[a])
        W = W.dot(ρ)
        # hankel
        for i, x in enumerate(sorted(self.X)):
            if self.oracle.membership_query(x + string).value != W[i]:
                return False, x
        return True, None

    def learn(self):
        # Start with an empty FSA
        fsa = FSA_Util.empty_fsa(self.Σ, self.oracle.membership_query("").value)

        # Check if equal, else get counterexample y
        equal, y = self.oracle.equivalence_query(fsa)
        if equal:
            return fsa

        # Initialize column and row indices
        self.X = {""}
        self.Y = {y}

        # Add the initial prefix and suffix to hankel matrix, as well as a default empty column
        self.h.set_val("", y, self.oracle.membership_query(y))
        self.h.add_col("", self.oracle.membership_query)

        # Fill in the rows extended by a single letter
        self.h.complete(self.oracle.membership_query)

        while True:
            fsa = self.h.to_fsa(self.X, self.Y)
            equal, z = self.oracle.equivalence_query(fsa)
            if equal:
                return fsa

            for i in range(0, len(z)):
                w = z[len(z) - i :]
                a = z[len(z) - i - 1 : len(z) - i]
                correct, x = self.correct(fsa, a + w)
                if not correct:
                    self.X.add(x + a)
                    self.Y.add(w)
                    if x + a not in self.h.P:
                        self.h.add_row(x + a, self.oracle.membership_query)
                    if w not in self.h.S:
                        self.h.add_col(w, self.oracle.membership_query)
                    break
            self.h.complete(self.oracle.membership_query)


if __name__ == "__main__":
    # FSA to learn
    orig_fsa = FSA_Util.example_fsa2()
    Σ = orig_fsa.Sigma

    # Initialize Oracle
    oracle = Oracle(orig_fsa)

    # Initialize Beimel
    beimel = Beimel(oracle, Σ)

    fsa = beimel.learn()
    print(fsa)
