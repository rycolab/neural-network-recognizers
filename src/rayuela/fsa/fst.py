from itertools import product
from typing import Generator, Optional, Tuple, Union

from frozendict import frozendict

from rayuela.base.semiring import Boolean, Semiring
from rayuela.base.symbol import Sym, ε, ε_1, ε_2
from rayuela.fsa.fsa import FSA
from rayuela.base.state import PairState, State


class FST(FSA):
    def __init__(self, R=Boolean):
        # DEFINITION
        # A weighted finite-state transducer is a 6-tuple <Σ, Δ, Q, δ, λ, ρ> where
        # • Σ is an alphabet of symbols;
        # • Δ is an alphabet of symbols;
        # • Q is a finite set of states;
        # • δ is a finite relation Q × Σ × Δ × Q × R;
        # • λ is an initial weight function;
        # • ρ is a final weight function.

        # NOTATION CONVENTIONS
        # • single states (elements of Q) are denoted q
        # • multiple states not in sequence are denoted, p, q, r, ...
        # • multiple states in sequence are denoted i, j, k, ...
        # • symbols (elements of Σ and Δ) are denoted lowercase a, b, c, ...
        # • single weights (elements of R) are denoted w
        # • multiple weights (elements of R) are denoted u, v, w, ...

        super().__init__(R=R)

        # alphabet of output symbols
        self.Delta = set()

    def add_arc(self, i: State, a: Sym, b: Sym, j: State, w=None) -> None:
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(b, Sym):
            b = Sym(b)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        self.Delta.add(b)
        self.δ[i][(a, b)][j] += w
        self.δ_inv[j][(a, b)][i] += w

    def set_arc(self, i: State, a: Sym, b: Sym, j: State, w=None) -> None:
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(b, Sym):
            b = Sym(b)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        self.Delta.add(b)
        self.δ[i][(a, b)][j] = w
        self.δ_inv[j][(a, b)][i] = w

    def spawn(self, keep_init: bool = False, keep_final: bool = False) -> "FST":
        """returns a new FST in the same semiring"""
        F = FST(R=self.R)

        if keep_init:
            for q, w in self.I:
                F.set_I(q, w)
        if keep_final:
            for q, w in self.F:
                F.set_F(q, w)

        return F

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.Delta = frozenset(self.Delta)
        self.Q = frozenset(self.Q)
        self.δ = frozendict(self.δ)
        self.λ = frozendict(self.λ)
        self.ρ = frozendict(self.ρ)

    def arcs(
        self, i: State, no_eps: bool = False
    ) -> Generator[Tuple[Tuple[Sym, Sym], State, Semiring], None, None]:
        """
        Returns the arcs of the FST starting from the state i.

        Args:
            i (State): The starting state.
            no_eps (bool, optional): Whether to filter out the epsilon transitions.
                Defaults to False.

        Yields:
            Union[Tuple[Tuple[Sym, Sym], State, Semiring],
                Tuple[Sym, Sym, State, Semiring]]:
                The arcs of the FST starting from the state i.
        """
        for ab, T in self.δ[i].items():
            if no_eps and ab == (ε, ε):
                continue
            for j, w in T.items():
                if w == self.R.zero:
                    continue
                yield ab, j, w

    def reverse(self):
        """creates a reversed machine"""

        # create the new machine
        Tr = self.spawn()

        # add the arcs in the reversed machine
        for i in self.Q:
            for (a, b), j, w in self.arcs(i):
                Tr.add_arc(j, a, b, i, w)

        # reverse the initial and final states
        for q, w in self.I:
            Tr.set_F(q, w)
        for q, w in self.F:
            Tr.set_I(q, w)

        return Tr

    def decouple(self) -> "FST":
        """Transforms the FST into one where the consumed and output symbols are
        decoupled into individual transitions.
        If the original FST contains the transition q -a:b-> q',
        the transformed transducer contains the transitions
        q -a:ε-> (q, a) and (q, a) -ε:b-> q'.

        Returns:
            FST: _description_
        """
        Tʼ = self.spawn()

        for q in self.Q:
            for (a, b), j, w in self.arcs(q):
                # This will add 1 for every arc mapping a to something, resulting in
                # a sort of a conditional "probability"
                Tʼ.add_arc(q, a, ε, State((q.idx, a.value), label="sep"), w=self.R.one)
                Tʼ.add_arc(State((q.idx, a.value), label="sep"), ε, b, j, w=w)

        for q, w in self.I:
            Tʼ.set_I(q, w)
        for q, w in self.F:
            Tʼ.set_F(q, w)

        return Tʼ

    def accept(self, x: str, y: Optional[str] = None) -> Union[Semiring, FSA]:
        """Returns the acceptance weight of the string pair x:y by the FST or the
        FSA representing the weighted language of all strings y such that x:y is
        accepted by the FST.

        Args:
            x (str): The input string.
            y (str): The output string.

        Returns:
            Union[Semiring, FSA]: The acceptance weight of the string pair x:y by the
                FST or the FSA representing the weighted language of all strings y such
                that x:y is accepted by the FST.
        """
        from rayuela.fsa.fsa_classes import string_fsa
        from rayuela.fsa.pathsum import Pathsum, Strategy

        if y is not None:
            T_x = string_fsa(x, self.R).to_identity_fst()
            T_y = string_fsa(y, self.R).to_identity_fst()

            T_xy = T_x.compose(self.compose(T_y))

            return Pathsum(T_xy).pathsum(Strategy.VITERBI)
        else:
            T_x = string_fsa(x, self.R).to_identity_fst()
            T_x = T_x.compose(self)

            return T_x.project(1)

    def project(self, component: int) -> "FSA":
        """Projects the FST into a FSA accepting the input or output strings.

        Args:
            component (int): The component to project onto (0 for the input string,
                1 for the output string).

        Returns:
            FSA: The projected FSA.
        """
        assert component in [0, 1]

        A = FSA(R=self.R)
        for q in self.Q:
            for (a, b), j, w in self.arcs(q):
                if component == 0:
                    A.add_arc(q, a, j, w=w)
                else:
                    A.add_arc(q, b, j, w=w)

        for q, w in self.I:
            A.add_I(q, w=w)

        for q, w in self.F:
            A.add_F(q, w=w)

        return A

    def _transform_arc(
        self, q: State, a: Sym, b: Sym, j: State, w: Semiring, idx: int
    ) -> Tuple[Sym, Sym]:
        if idx == 1:
            if b != ε:
                return a, b
            else:
                return a, ε_2
        else:
            if a != ε:
                return a, b
            else:
                return ε_1, b

    def augment_epsilon_transitions(self, idx: int) -> "FST":
        """Augments the FST by changing the appropriate epsilon transitions to
        epsilon_1 or epsilon_2 transitions to be able to perform the composition
        with epsilon transitions correctly.
        See also Fig. 7 in Mohri, Weighted Automata Algorithms, p. 17.

        Args:
            idx (int): 1 if the FST is the first one in the composition, 2 otherwise.

        Returns:
            FST: The augmented FST.
        """
        assert idx in [1, 2]

        T = self.spawn()

        for q in self.Q:
            if idx == 1:
                T.add_arc(q, ε, ε_1, q, w=self.R.one)
            else:
                T.add_arc(q, ε_2, ε, q, w=self.R.one)

            for (a, b), j, w in self.arcs(q):
                _a, _b = self._transform_arc(q, a, b, j, w, idx)
                T.add_arc(q, _a, _b, j, w=w)

        for q, w in self.I:
            T.set_I(q, w=w)

        for q, w in self.F:
            T.set_F(q, w=w)

        return T

    def _compose(self, T: "FST") -> "FST":
        """Implements the on-the-fly composition of the FST self with the FST T.

        Args:
            T (FST): The FST to compose with.

        Returns:
            FST: The result of the composition.
        """
        # the two machines need to be in the same semiring
        assert self.R == T.R

        # add initial states
        T_c = FST(R=self.R)
        for (q, w1), (p, w2) in product(self.I, T.I):
            T_c.add_I(PairState(q, p), w=w1 * w2)

        I1, I2 = {q: w for q, w in self.I}, {q: w for q, w in T.I}
        F1, F2 = {q: w for q, w in self.F}, {q: w for q, w in T.F}

        visited = set([(i1, i2) for i1, i2 in product(I1, I2)])
        stack = [(i1, i2) for i1, i2 in product(I1, I2)]

        while stack:
            q, p = stack.pop()

            for ((a, b), qʼ, w1), ((c, d), pʼ, w2) in product(self.arcs(q), T.arcs(p)):
                if b != c:
                    continue

                T_c.add_arc(PairState(q, p), a, d, PairState(qʼ, pʼ), w=w1 * w2)

                if (qʼ, pʼ) not in visited:
                    stack.append((qʼ, pʼ))
                    visited.add((qʼ, pʼ))

        # final state handling
        for q, p in product(F1, F2):
            T_c.set_F(PairState(q, p), w=F1[q] * F2[p])

        return T_c

    def compose(self, fst: "FST", augment: bool = True) -> "FST":
        from rayuela.fsa.fsa_classes import get_epsilon_filter

        F = get_epsilon_filter(self.R, self.Delta)

        if augment:
            T1 = self.augment_epsilon_transitions(1)
            T2 = fst.augment_epsilon_transitions(2)
        else:
            T1, T2 = self, fst

        T1_F_T2 = T1._compose(F).trim()._compose(T2).trim()

        return T1_F_T2

    def top_compose(self, fst: "FST", augment: bool = True):
        return self.compose(fst, augment)

    def bottom_compose(self, fst: "FST", augment: bool = True):
        return self.top_compose(fst, augment)

    def _compose_brute(self, T: "FST"):
        # the two machines need to be in the same semiring
        assert self.R == T.R

        # add initial states
        product_fst = FST(R=self.R)
        for (q, w1), (p, w2) in product(self.I, T.I):
            product_fst.add_I(PairState(q, p), w=w1 * w2)

        # add "body" states
        for q, p in product(self.Q, T.Q):
            for ((a, b), qʼ, w1), ((c, d), pʼ, w2) in product(self.arcs(q), T.arcs(p)):
                if b != c:
                    continue

                product_fst.add_arc(PairState(q, p), a, d, PairState(qʼ, pʼ), w=w1 * w2)

        # final state handling
        for (q, w1), (p, w2) in product(self.F, T.F):
            if PairState(q, p) in product_fst.Q:
                product_fst.add_F(PairState(q, p), w=w1 * w2)

        return product_fst

    def compose_brute(self, fst: "FST", augment: bool = True) -> "FST":
        from rayuela.fsa.fsa_classes import get_epsilon_filter

        F = get_epsilon_filter(self.R, self.Delta)

        if augment:
            T1 = self.augment_epsilon_transitions(1)
            T2 = fst.augment_epsilon_transitions(2)
        else:
            T1, T2 = self, fst

        T1_F_T2 = T1._compose_brute(F).trim()._compose_brute(T2).trim()

        return T1_F_T2

    def bottom_compose_brute(self, fst: "FST", augment: bool = True):
        return self.top_compose_brute(fst, augment)

    def top_compose_brute(self, fst: "FST", augment: bool = True):
        return self.compose_brute(fst, augment)
