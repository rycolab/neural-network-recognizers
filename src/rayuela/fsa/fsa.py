from __future__ import annotations

import copy
from collections import Counter
from collections import defaultdict as dd
from collections import deque
from itertools import product
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
from frozendict import frozendict

import rayuela
from rayuela.base.semiring import Boolean, ProductSemiring, Real, Semiring
from rayuela.base.state import PairState, State
from rayuela.base.symbol import Expr, Sym, ε, ε_1, ε_2, φ
from rayuela.cfg.nonterminal import NT, S
from rayuela.fsa.pathsum import Pathsum, Strategy


class FSA:
    def __init__(self, R: Type[Semiring] = Boolean):
        # DEFINITION
        # A weighted finite-state automaton is a 5-tuple <R, Σ, Q, δ, λ, ρ> where
        # • R is a semiring;
        # • Σ is an alphabet of symbols;
        # • Q is a finite set of states;
        # • δ is a finite relation Q × Σ × Q × R;
        # • λ is an initial weight function;
        # • ρ is a final weight function.

        # NOTATION CONVENTIONS
        # • single states (elements of Q) are denoted q
        # • multiple states not in sequence are denoted, p, q, r, ...
        # • multiple states in sequence are denoted i, j, k, ...
        # • symbols (elements of Σ) are denoted lowercase a, b, c, ...
        # • single weights (elements of R) are denoted w
        # • multiple weights (elements of R) are denoted u, v, w, ...alphabet

        # semiring
        self.R = R

        # alphabet of symbols
        self.Sigma = set([])
        self.symbol2idx, self.idx2symbol = {}, {}

        # a finite set of states
        self.Q = set([])
        self.state2idx, self.idx2state = {}, {}

        # transition function : Q × Σ × Q → R
        self.δ = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))
        # We also define the inverse transition function δ_inv
        self.δ_inv = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        # initial weight function
        self.λ = R.chart()

        # final weight function
        self.ρ = R.chart()

        self._is_probabilistic = False

        # For displaying the FSA in a juptyer notebook
        self.theme = "dark"  # Set to "light" for a light theme

    def add_state(self, q: State) -> None:
        """Adds a state to the automaton.
        This method should mainly be accessed through the add_arc method.

        Args:
            q (State): The state to be added.
        """
        assert isinstance(self.Q, set), "Cannot add to frozen FSA"
        self.Q.add(q)

    def add_states(self, Q: Union[List[State], Set[State], Tuple[State, ...]]) -> None:
        """Adds a list of states to the automaton."""
        for q in Q:
            if q not in self.state2idx:
                self.state2idx[q] = len(self.state2idx)
                self.idx2state[self.state2idx[q]] = q
            self.add_state(q)

    def new_state(self):
        """ "Generates a fresh state"""
        counter = len(self.Q)

        q = State(f"@{counter}")
        while q in self.Q:
            counter += 1
            q = State(f"@{counter}")

        return q

    def add_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] += w
        self.δ_inv[j][a][i] += w

    def set_arc(self, i: State, a: Sym, j: State, w: Optional[Semiring] = None):
        assert isinstance(self.Sigma, set), "Cannot add to frozen FSA"
        if w is None:
            w = self.R.one

        if not isinstance(i, State):
            i = State(i)
        if not isinstance(j, State):
            j = State(j)
        if not isinstance(a, Sym):
            a = Sym(a)
        if not isinstance(w, self.R):
            w = self.R(w)

        self.add_states([i, j])
        self.Sigma.add(a)
        if a not in self.symbol2idx:
            self.symbol2idx[a] = len(self.symbol2idx)
            self.idx2symbol[self.symbol2idx[a]] = a
        self.δ[i][a][j] = w
        self.δ_inv[j][a][i] = w

    def set_I(self, q, w=None):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.λ[q] = w

    def set_F(self, q, w=None):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        if not isinstance(q, State):
            q = State(q)

        if w is None:
            w = self.R.one
        self.add_state(q)
        self.ρ[q] = w

    def add_I(self, q, w):
        assert isinstance(self.λ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.λ[q] += w

    def add_F(self, q, w):
        assert isinstance(self.ρ, dict), "Cannot add to frozen FSA"
        self.add_state(q)
        self.ρ[q] += w

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.Q = frozenset(self.Q)
        self.δ = frozendict(self.δ)
        self.δ_inv = frozendict(self.δ_inv)
        self.λ = frozendict(self.λ)
        self.ρ = frozendict(self.ρ)

    @property
    def I(self) -> Generator[Tuple[State, Semiring], None, None]:  # noqa: E741, E743
        """Returns the initial states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the initial states of the FSA.
        """
        for q, w in self.λ.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self) -> Generator[Tuple[State, Semiring], None, None]:
        """Returns the final states of the FSA.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the final states of the FSA.
        """
        for q, w in self.ρ.items():
            if w != self.R.zero:
                yield q, w

    def arcs(
        self, i: State, no_eps: bool = False, nozero: bool = True, reverse: bool = False
    ) -> Generator[Tuple[Sym, State, Semiring], None, None]:
        """Returns the arcs stemming from state i or going into the state i in the FSA.
        in the form of tuples (a, j, w) where a is the symbol, j is the target state of
        the transition and w is the weight.

        Args:
            i (State): The state out of which the arcs stem or into which the arcs go.
            no_eps (bool, optional): If True, epsilon arcs are not returned.
                Defaults to False.
            nozero (bool, optional): If True, zero-weight arcs are not returned.
                Defaults to True.
            reverse (bool, optional): If False, the arcs stemming from state i are
                returned. If True, the arcs going into the state i are returned.
                Defaults to False.

        Yields:
            Generator[Tuple[Sym, State, Semiring], None, None]:
            Generator of the arcs stemming from state i in the FSA.
        """
        δ = self.δ if not reverse else self.δ_inv
        for a, transitions in δ[i].items():
            if no_eps and a == ε:
                continue
            for j, w in transitions.items():
                if w == self.R.zero and nozero:
                    continue
                yield a, j, w

    def out_symbols(
        self, q: State, ignore_eps: bool = False, ignore_phi: bool = False
    ) -> Generator[Sym, None, None]:
        """Returns the set of symbols that have outgoing arcs from state q.

        Args:
            q (State): The state for which the outgoing symbols are returned.
            ignore_eps (bool, optional): If True, epsilon arcs are not returned.
                Defaults to False.
            ignore_phi (bool, optional): If True, phi arcs are not returned.
                Defaults to False.

        Yields:
            Generator[Sym, None, None]: Generator of the symbols that have outgoing
            arcs from state q.
        """
        for a in self.δ[q]:
            if ignore_eps and a in [ε, ε_1, ε_2]:
                continue
            if ignore_phi and a == φ:
                continue
            yield a

    def a_out_arcs(
        self, q: State, a: Sym
    ) -> Generator[Tuple[State, Semiring], None, None]:
        """Returns the arcs stemming from state q with label a.

        Args:
            q (State): The state out of which the arcs stem.
            a (Sym): The label of the arcs.

        Yields:
            Generator[Tuple[State, Semiring], None, None]:
            Generator of the arcs stemming from state q with label a.
        """
        for j, w in self.δ[q][a].items():
            yield j, w

    def transition_matrix(self, a: Sym) -> List[List[Semiring]]:
        """Returns the transition matrix of the FSA for a given symbol.

        Args:
            a (Sym): The symbol for which the transition matrix is returned.

        Returns:
            List[List[Semiring]]: The transition matrix of the FSA for a given symbol.
        """
        n = self.num_states
        T_a = [[self.R.zero] * n for _ in range(n)]
        for q in self.Q:
            for r in self.Q:
                T_a[self.state2idx[q]][self.state2idx[r]] = self.δ[q][a][r]
        return T_a

    def predecessors(
        self, q: Union[State, Set[State]], a: Union[Sym, Set[Sym]]
    ) -> Set[State]:
        """Returns the set of predecessors of a set of states q
            for a given set of symbols a.

        Args:
            q (Union[State, Set[State]]): The set of states or an individual state
            for which the predecessors are returned.
            a (Union[Sym, Set[Sym]]): The symbols or an individual symbol for which
            the predecessors are returned.

        Returns:
            Set[State]: The set of predecessors of a state q for a given symbol a.
        """
        if not isinstance(q, set):
            q = {q}
        if not isinstance(a, set):
            a = {a}
        P = set()
        for q_, a_ in product(q, a):
            P |= set(self.δ_inv[q_][a_].keys())
        return P

    def accept(
        self, string: Union[str, Sequence[Union[Sym, NT]]], eps_free: bool = False
    ) -> Semiring:
        """Determines the stringsum/acceptance weight of the string `string`
        in the rational series defined by the WFSA.

        Args:
            string (Union[str, Sequence[Union[Sym, NT]]]):
                The string whose stringsum is to be determined.
            eps_free (bool, optional): Whether to ignore epsilon transitions.
                This enables the computation of the acceptance with the faster
                Viterbi algorithm. Defaults to False.

        Returns:
            Semiring: The stringsum value.
        """
        from rayuela.fsa.fsa_classes import string_fsa

        fsa_s = self.intersect(string_fsa(string, self.R), eps_free=eps_free)

        return (
            Pathsum(fsa_s).pathsum(Strategy.LEHMANN)
            if not eps_free
            else Pathsum(fsa_s).pathsum(Strategy.VITERBI)
        )

    @property
    def num_states(self) -> int:
        """Returns the number of states of the FSA."""
        return len(self.Q)

    @property
    def num_initial_states(self) -> int:
        """Returns the number of initial states of the FSA."""
        return len(list(self.I))

    @property
    def num_final_states(self) -> int:
        """Returns the number of final states of the FSA."""
        return len(list(self.F))

    @property
    def acyclic(self):
        cyclic, _ = self.dfs()
        return not cyclic

    @property
    def deterministic(self) -> bool:
        if len(list(self.I)) != 1:
            return False
        for q in self.Q:
            counter = Counter()
            for a, _, _ in self.arcs(q):
                if a == ε:  # a deterministic fsa cannot have ε transitions
                    return False
                counter[a] += 1
            most_common = counter.most_common(1)
            if len(most_common) > 0 and most_common[0][1] > 1:
                return False
        return True

    @property
    def pushed(self) -> bool:
        for i in self.Q:
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                total += w
            if total != self.R.one:
                return False
        return True

    @property
    def probabilistic(self) -> bool:  # noqa: C901
        assert self.R == Real

        if self._is_probabilistic:
            return True

        total = self.R.zero
        for i, w in self.I:
            if not w.value >= 0:
                return False, "Initial weights must be non-negative."
            total += w
        if total != self.R.one:
            return False, "Total weight of initial states must be 1."

        for i in self.Q:
            if not self.ρ[i].value >= 0:
                return False, "Final weights must be non-negative."
            total = self.ρ[i]
            for _, _, w in self.arcs(i):
                if not w.value >= 0:
                    return False, "Transition weights must be non-negative."
                total += w
            if total != self.R.one:
                return False, "Total weight of outgoing arcs must be 1."

        self._is_probabilistic = True
        return True

    @property
    def epsilon(self):
        for q in self.Q:
            for a, _, _ in self.arcs(q):
                if a == ε:
                    return True
        return False

    @property
    def ordered_states(self):
        """Returns a list of states ordered by their lexicographical index"""
        Q = list(self.Q)
        Q.sort(key=lambda a: str(a.idx))
        return Q

    @property
    def T(self) -> Dict[Sym, np.ndarray]:
        """Returns a dictionary of symbols to transition matrices.

        The matrices are indexed by state idx in lexicographical order.
        Matrix entry [i, j] corresponds to the transition weight
        from state i to state j.

        Returns:
            Dictionary of transition matrices M (one for each symbol).
        """

        assert self.R.is_field

        M = {}
        n = self.num_states
        Q = self.ordered_states

        for a in self.Sigma:
            M[a] = np.zeros((n, n))
            for i, p in enumerate(Q):
                if a in self.δ[p]:
                    for j, q in enumerate(Q):
                        M[a][i, j] = self.δ[p][a][q]

        return M

    @property
    def init_vector(self) -> np.ndarray:
        """Returns a vector of initial weights of the states, sorted by state idx
        in lexicographical order."""

        assert self.R.is_field

        n = self.num_states
        Q = self.ordered_states
        λ = np.zeros(n)

        for i, q in enumerate(Q):
            λ[i] = self.λ[q]

        return λ

    @property
    def final_vector(self) -> np.ndarray:
        """Returns a vector of final weights of the states, sorted by state idx
        in lexicographical order."""

        assert self.R.is_field

        n = self.num_states
        Q = self.ordered_states
        ρ = np.zeros(n)

        for i, q in enumerate(Q):
            ρ[i] = self.ρ[q]

        return ρ

    def copy(self):
        """deep copies the machine"""
        return copy.deepcopy(self)

    def spawn(self, keep_init=False, keep_final=False):
        """returns a new FSA in the same semiring"""
        F = FSA(R=self.R)

        if keep_init:
            for q, w in self.I:
                F.set_I(q, w)
        if keep_final:
            for q, w in self.F:
                F.set_F(q, w)

        return F

    def _enumerate_paths(  # noqa: C901
        self,
        q: State,
        label: Optional[Sequence[Sym]] = None,
        length: Optional[int] = None,
        max_length: Optional[int] = None,
        reverse: bool = False,
    ) -> Generator[List[Tuple[State, Sym, Semiring, State]], None, None]:
        """Returns a list of all paths of length exactly `length` of of length at most
        `max_length` from state `q` scanning the sequence `label` (if provided).

        ! NOTE: This is *not* thoroughly tested.

        Args:
            q (State): The start state of the paths.
            label (Sequence[Sym]): Optional required sequence of transition labels.
                Defaults to None.
            length (int, optional): The optional exact length of the path.
            max_length (int, optional): The optional maximal length of the path.
                Defaults to None.
            reverse (bool, optional): Whether to look for paths in the reverse FSA.

        Returns:
            Generator[List[Tuple[State, Sym, Semiring, State]], None, None]:
            The list of paths where each path is represented as a list of
            (start state, label, weight, end state) tuples.
        """
        if length is not None and max_length is not None:
            assert length <= max_length

        if label is not None and label != []:
            _a, label = label[0], label[1:]
        else:
            _a = None

        if label is None or _a is None and label == []:
            yield []
        elif length is not None and length == 0:
            yield []
        elif length is not None and length > 0:
            for a, j, w in self.arcs(q, reverse=reverse):
                if _a is None or a == _a or a == ε:
                    for path in self._enumerate_paths(
                        j, label, length=length - 1, reverse=reverse
                    ):
                        yield [(q, a, w, j)] + path
        elif max_length is not None and max_length == 0:
            yield []
        elif max_length is not None and max_length > 0:
            for a, j, w in self.arcs(q, reverse=reverse):
                if _a is None or a == _a or a == ε:
                    for path in self._enumerate_paths(
                        j, label, max_length=max_length - 1, reverse=reverse
                    ):
                        yield [(q, a, w, j)] + path
        else:
            for a, j, w in self.arcs(q, reverse=reverse):
                if _a is None or a == _a or a == ε:
                    for path in self._enumerate_paths(j, label, reverse=reverse):
                        yield [(q, a, w, j)] + path

    def enumerate_paths(
        self,
        q: State,
        label: Optional[Sequence[Sym]] = None,
        length: Optional[int] = None,
        max_length: Optional[int] = None,
        reverse: bool = False,
    ) -> Generator[List[Tuple[State, Sym, Semiring, State]], None, None]:
        """Returns a list of all paths of length exactly `length` of of length at most
        `max_length` from state `q` scanning the sequence `label` (if provided).

        Args:
            q (State): The start state of the paths.
            label (Sequence[Sym]): Optional required sequence of transition labels.
                Defaults to None.
            length (int, optional): The optional exact length of the path.
            max_length (int, optional): The optional maximal length of the path.
                Defaults to None.
            reverse (bool, optional): Whether to look for paths in the reverse FSA.

        Returns:
            Generator[List[Tuple[State, Sym, Semiring, State]], None, None]:
            The list of paths where each path is represented as a list of
            (start state, label, weight, end state) tuples.
        """
        for path in self._enumerate_paths(
            q, label=label, length=length, max_length=max_length, reverse=reverse
        ):
            yield path

    def unit(self):
        """returns a copy of the current FSA with all the weights set to unity"""
        nfsa = self.spawn()
        one = self.R.one

        for q, _ in self.I:
            nfsa.set_I(q, one)

        for q, _ in self.F:
            nfsa.set_F(q, one)

        for i in self.Q:
            for a, j, _ in self.arcs(i):
                nfsa.add_arc(i, a, j, one)

        return nfsa

    def lift(self, R: Semiring, lifter: Callable[[Semiring], Semiring]) -> "FSA":
        """Lifts the weights of the FSA into a different different semiring where
        the weights are defined by the lifter function.

        Args:
            R (Semiring): The semiring into which the weights are lifted.
            lifter (Callable[[Semiring], Semiring]): The function that maps the original
                weights into the lifted ones.

        Returns:
            FSA: The lifted FSA.
        """
        A = FSA(R)
        for q, w in self.I:
            A.set_I(q, lifter(w))
        for q, w in self.F:
            A.set_F(q, lifter(w))
        for q in self.Q:
            for a, j, w in self.arcs(q):
                A.add_arc(q, a, j, lifter(w))
        return A

    def entropy(self) -> Real:
        """Computes the entropy of the FSA.

        Returns:
            Real: The entropy of the FSA.
        """
        from math import log

        from rayuela.base.semiring import Entropy

        assert self.R == Real

        return Real(
            self.lift(Entropy, lambda w: Entropy(float(w), -log(float(w))))
            .pathsum()
            .value[1]
        )

    def push(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.push(self)

    def normalize(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.normalize(self)

    def determinize(self, timeout=1000):
        from rayuela.fsa.transformer import Transformer

        if self.epsilon:
            return self.epsremove().determinize()
        return Transformer.determinize(self, timeout=timeout)

    def minimize(self, strategy=None):
        from rayuela.fsa.transformer import Transformer

        assert self.deterministic

        if self.R != Boolean:
            trim_fsa = self.trim()
            pushed_fsa = trim_fsa.push()
            lifted_fsa, if_weights = Transformer.lift_weights_to_labels(pushed_fsa)
            min_fsa = Transformer._minimize(lifted_fsa, strategy=strategy)
            return Transformer.get_weights_from_labels(min_fsa, self.R, if_weights)

        else:
            return Transformer._minimize(self, strategy=strategy)

    def epsremove(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.epsremoval(self)

    def _q2NT(self, q: State) -> NT:
        if isinstance(q.idx, int):
            return NT(chr(64 + q.idx))
        elif isinstance(q.idx, tuple):
            return NT("".join([chr(64 + elem) for elem in q.idx]))
        elif isinstance(q.idx, str):
            return NT(q.idx)

    def to_cfg(self):
        """converts the WFSA to an equivalent WCFG"""

        from rayuela.cfg.cfg import CFG

        cfg = CFG(R=self.R)
        s = State(0)
        NTs = {s: S}

        for i in self.Q:
            NTs[i] = self._q2NT(i)

        for i in self.Q:
            # add production rule for initial states
            if i in self.λ.keys():
                cfg.add(self.λ[i], NTs[s], NTs[i])

            # add production rule for final states
            if i in self.ρ.keys():
                cfg.add(self.ρ[i], NTs[i], ε)

            # add other production rules
            for a, j, w in self.arcs(i):
                cfg.add(w, NTs[i], a, NTs[j])

        return cfg

    def reverse(self):
        """creates a reversed machine"""

        # create the new machine
        R = self.spawn()

        # add the arcs in the reversed machine
        for i in self.Q:
            for a, j, w in self.arcs(i):
                R.add_arc(j, a, i, w)

        # reverse the initial and final states
        for q, w in self.I:
            R.set_F(q, w)
        for q, w in self.F:
            R.set_I(q, w)

        return R

    def undirected(self) -> "FSA":
        """Produces an undirected version of the FSA (where all the transitions
           run in both directions).

        Returns:
            FSA: The undirected FSA.
        """

        undirected_fsa = self.copy()
        for q in self.Q:
            for a, t, w in self.arcs(q):
                undirected_fsa.add_arc(t, a, q, w)

        return undirected_fsa

    def accessible(self):
        """computes the set of accessible states"""
        A = set()
        stack = [q for q, w in self.I if w != self.R.zero]
        while stack:
            i = stack.pop()
            for _, j, _ in self.arcs(i):
                if j not in A:
                    stack.append(j)
            A.add(i)

        return A

    def coaccessible(self):
        """computes the set of co-accessible states"""
        return self.reverse().accessible()

    def is_parent(self, p: State, q: State) -> bool:
        """Checks whether `p` is a parent of `q` in the FSA.

        Args:
            p (State): The candidate parent
            q (State): The candidate child

        Returns:
            bool: Whether `p` is a parent of `q`
        """
        return q in [t for _, t, _ in self.arcs(p)]

    def connected_by_symbol(self, p: State, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an edge from `p` to `q` with the label `symbol`.

        Args:
            p (State): The candidate parent
            q (State): The candidate child
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an edge from `p` to `q` with the label `symbol`
        """
        return symbol in self.δ[p] and q in self.δ[p][symbol]

    def has_incoming_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an incoming edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbom (Sym): The arc label to check

        Returns:
            bool: Whereher there is an incoming edge into `q` with the label `symbol`.
        """
        for p in self.Q:
            for a, t, _ in self.arcs(p):
                if a == symbol and t == q:
                    return True
        return False

    def has_outgoing_arc(self, q: State, symbol: Sym) -> bool:
        """Checks whereher there is an outgoing edge into `q` with the label `symbol`.

        Args:
            q (State): The state
            symbol (Sym): The arc label to check

        Returns:
            bool: Whether there is an outgoing edge into `q` with the label `symbol`.
        """
        return symbol in self.δ[q]

    def has_fallback_state(self, q: State) -> bool:
        """Checks whether `q` has a fallback state (i.e., an outgoing failure arc).

        Args:
            q (State): The state.

        Returns:
            bool: Whether `q` has a fallback state.
        """
        return φ in self.δ[q]

    def qφ(self, q: State) -> State:
        """Returns the fallback state of `q`, if it exists."""
        assert self.has_fallback_state(q), "The state has no fallback state"
        return list(self.δ[q][φ].keys())[0]

    def transition(
        self, q: State, a: Sym, weight: bool = False
    ) -> Optional[Union[State, Tuple[State, Semiring]]]:
        """If the FSA is deterministic and there exists an a-transition out of q,
            then the function returns the target state of the transition.

        Args:
            q (State): The state.
            a (Sym): The symbol.
            weight (bool, optional): Whether to return the weight of the transition.

        Returns:
            State: The target state of the transition.
        """
        assert self.deterministic

        if self.has_outgoing_arc(q, a):
            if weight:
                return list(self.δ[q][a].items())[0]
            else:
                return list(self.δ[q][a].keys())[0]
        else:
            return None

    def dfs(
        self, Is: Optional[Set[State]] = None, intervals: bool = False
    ) -> Union[
        Tuple[bool, Dict[State, int]], Tuple[bool, Dict[State, Tuple[int, int]]]
    ]:
        """Depth-first search (Cormen et al. 2019; Section 22.3)

        Args:
            Is (Optional[Set[State]], optional): The set of initial states to start
            the DFS from.
            intervals (bool, optional): Whether to return the intervals of the DFS.
            Defaults to False.

            Returns:
                Union[Tuple[bool, Dict[State, int]],
                    Tuple[bool, Dict[State, Tuple[int, int]]]]:
                If `intervals` is False, the function returns a tuple (cyclic, finished)
                where `cyclic` is a boolean indicating whether the FSA is cyclic and
                `finished` is a dictionary mapping each state to its finishing time.
                If `intervals` is True, the function returns a tuple (cyclic, finished)
                where `cyclic` is a boolean indicating whether the FSA is cyclic and
                `finished` is a dictionary mapping each state to its
                interval on the stack.
        """

        in_progress, finished = set([]), dict()
        cyclic, counter = False, 0

        def _dfs(p):
            nonlocal in_progress
            nonlocal finished
            nonlocal cyclic
            nonlocal counter

            in_progress.add(p)
            if intervals:
                finished[p] = (counter, None)

            for _, q, _ in self.arcs(p):
                if q in in_progress:
                    cyclic = True
                elif q not in finished:
                    _dfs(q)

            in_progress.remove(p)
            finished[p] = counter if not intervals else (finished[p][0], counter)
            counter += 1

        Is = Is if Is is not None else set([q for q, _ in self.I])
        for q in Is:
            _dfs(q)

        return cyclic, finished

    def finish(self, rev=False, acyclic_check=False):
        """
        Returns the nodes in order of their finishing time.
        """

        _, finished = self.dfs()

        if acyclic_check:
            assert self.acyclic

        sort = {}
        for s, n in finished.items():
            sort[n] = s
        if rev:
            for n in sorted(list(sort.keys())):
                yield sort[n]
        else:
            for n in reversed(sorted(list(sort.keys()))):
                yield sort[n]

    def toposort(self, rev=False):
        return self.finish(rev=rev, acyclic_check=True)

    def trim(self):
        from rayuela.fsa.transformer import Transformer

        return Transformer.trim(self)

    def pathsum(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.pathsum(strategy)

    def forward(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.forward(strategy)

    def backward(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.backward(strategy)

    def allpairs(self, strategy=Strategy.LEHMANN):
        pathsum = Pathsum(self)
        return pathsum.allpairs(strategy)

    def booleanize(self):
        fsa = FSA(Boolean)

        for q, w in self.I:
            fsa.add_I(q, fsa.R(w != self.R.zero))

        for q, w in self.F:
            fsa.add_F(q, fsa.R(w != self.R.zero))

        for q in self.Q:
            for a, j, w in self.arcs(q):
                fsa.add_arc(q, a, j, fsa.R(w != self.R.zero))

        return fsa

    # TODO
    def topologically_equivalent(self, fsa):
        """Tests topological equivalence."""

        # We need to enforce both self and fsa are determinized, pushed and minimized
        assert self.deterministic and fsa.deterministic, "The FSA are not deterministic"
        assert self.pushed and fsa.pushed, "The FSA are not pushed"
        assert self.minimized and fsa.minimized, "The FSA are not minimized"

        # Theorem. If G and H are graphs with out-degree at most 1, then
        # the greedy works to determine whether G and H are isomorphic

        # A deterministic machine has exactly one start state

        # Two minimized DFAs are input
        # If number of states is different, return False

        # Our goal it to trying to find a matching the vertices

        stack = [(q1, q2) for (q1, _), (q2, _) in product(self.I, fsa.I)]
        iso = {stack[0][0]: stack[0][1]}

        while stack:
            p, q = stack.pop()
            for a in self.Sigma:
                r, s = self.δ[p][a], fsa.δ[q][a]
                if not iso[r] == s:
                    return False
                iso[r] = s
        return True

    def equivalent(self, fsa):
        """Tests equivalence."""
        from rayuela.fsa.transformer import Transformer

        if self.R is not fsa.R:
            print("Not equivalent due to different semiring")
            return False

        if self.Sigma != fsa.Sigma:
            print("Not equivalent due to different alphabet")
            return False

        fsa0 = Transformer.determinize(
            Transformer.epsremoval(self.single_I().booleanize())
        ).trim()
        fsa1 = Transformer.determinize(
            Transformer.epsremoval(fsa.single_I().booleanize())
        ).trim()

        fsa2 = fsa0.intersect(fsa1.complement())
        fsa3 = fsa1.intersect(fsa0.complement())

        U = fsa2.union(fsa3)

        return U.trim().num_states == 0

    def edge_marginals(self) -> "dd[State, dd[Sym, dd[State, Semiring]]]":
        """computes the edge marginals μ(q→q')"""
        marginals = dd(lambda: dd(lambda: dd(lambda: self.R.zero)))

        α = Pathsum(self).forward(strategy=Strategy.VITERBI)
        β = Pathsum(self).backward(strategy=Strategy.VITERBI)

        for q in self.Q:
            for a, q_prime, w in self.arcs(q):
                marginals[q][a][q_prime] += α[q] * w * β[q_prime]

        return marginals

    def difference(self, fsa):
        """coputes the difference with a uniweighted DFA"""

        fsa = fsa.complement()

        # fsa will be a boolean FSA, need to make the semirings compatible
        F = FSA(self.R)
        for q, w in fsa.I:
            F.add_I(q, F.R(w.value))
        for q, w in fsa.F:
            F.add_F(q, F.R(w.value))
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                F.add_arc(q, a, j, F.R(w.value))

        return self.intersect(F)

    def _union_add(self, A: "FSA", U: "FSA", idx: int):
        for i in A.Q:
            for a, j, w in A.arcs(i):
                U.add_arc(PairState(State(idx), i), a, PairState(State(idx), j), w)

        for q, w in A.I:
            U.set_I(PairState(State(idx), q), w)

        for q, w in A.F:
            U.set_F(PairState(State(idx), q), w)

    def union(self, A: "FSA") -> "FSA":
        """construct the union of the two FSAs"""

        assert self.R == A.R

        U = self.spawn()

        # add arcs, initial and final states from self
        self._union_add(self, U, 1)

        # add arcs, initial and final states from argument
        self._union_add(A, U, 2)

        return U

    def single_I(self):
        """Returns an equivalent FSA with only 1 initial state"""
        if len([q for q, _ in self.I]) == 1:
            return self

        # Find suitable names for the new state
        ixs = [q.idx for q in self.Q]
        start_id = 0
        while f"single_I_{start_id}" in ixs:
            start_id += 1

        F = self.spawn(keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                F.add_arc(i, a, j, w)

        for i, w in self.I:
            F.add_arc(State(f"single_I_{start_id}"), ε, i, w)

        F.set_I(State(f"single_I_{start_id}"), F.R.one)

        return F

    def concatenate(self, fsa):
        """construct the concatenation of the two FSAs"""

        assert self.R == fsa.R

        C = self.spawn()

        # add arcs, initial and final states from self
        for i in self.Q:
            for a, j, w in self.arcs(i):
                C.add_arc(PairState(State(1), i), a, PairState(State(1), j), w)

        for q, w in self.I:
            C.set_I(PairState(State(1), q), w)

        # add arcs, initial and final states from argument
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                C.add_arc(PairState(State(2), i), a, PairState(State(2), j), w)

        for q, w in fsa.F:
            C.set_F(PairState(State(2), q), w)

        # connect the final states from self to initial states from argument
        for (i1, w1), (i2, w2) in product(self.F, fsa.I):
            C.add_arc(PairState(State(1), i1), ε, PairState(State(2), i2), w1 * w2)

        return C

    def kleene_closure(self):
        # Find suitable names for new states
        ixs = [q.idx for q in self.Q]
        start_id, final_id = 0, 0
        while f"kleene_closure_start_{start_id}" in ixs:
            start_id += 1
        while f"kleene_closure_final_{final_id}" in ixs:
            final_id += 1

        K = self.spawn()

        for q in self.Q:
            for a, j, w in self.arcs(q):
                K.set_arc(q, a, j, w)

        i = State(f"kleene_closure_start_{start_id}")
        f = State(f"kleene_closure_final_{final_id}")

        K.add_I(i, K.R.one)
        K.add_F(f, K.R.one)

        for q, w in self.I:
            K.add_arc(i, ε, q, w)

        for q, w in self.F:
            K.add_arc(q, ε, f, w)

        K.set_arc(i, ε, f, K.R.one)

        for (f, wi), (i, wf) in product(self.F, self.I):
            K.add_arc(f, ε, i, wi * wf)

        return K

    def regex(self, combine: bool = False) -> Union[ProductSemiring, Expr]:
        """Constructs the weighted regular expression corresponding to the FSA.

        Args:
            combine (bool, optional): Whether to combine the regular expressions
                of the labels and weights into one.

        Returns:
            Union[ProductSemiring, Expr]: The regular expression.
        """
        from rayuela.fsa.transformer import Transformer

        Al = Transformer.lift_labels_to_weight(self)

        R = Al.pathsum("lehmann")

        if combine:
            return Transformer.combine_regex_and_weight(R)
        else:
            return R

    def coaccessible_intersection(self, fsa: FSA):
        """
        on-the-fly weighted intersection
        Implementation by @giacomocamposampiero
        """
        # the two machines need to be in the same semiring
        assert self.R == fsa.R

        # add initial states
        product_fsa = FSA(R=self.R)

        # create final and initial sets
        self_finals = {q: w for q, w in self.F}
        fsa_finals = {q: w for q, w in fsa.F}

        self_initials = {q: w for q, w in self.I}
        fsa_initials = {q: w for q, w in fsa.I}

        visited = set([(i1, i2) for i1, i2 in product(self_finals, fsa_finals)])
        stack = [(i1, i2) for i1, i2 in product(self_finals, fsa_finals)]

        r_self = self.reverse()
        r_fsa = fsa.reverse()

        # "backward" dfs from final states
        while stack:
            q1, q2 = stack.pop()
            # analyse incoming edges in q1, q2
            for (a1, j1, w1), (a2, j2, w2) in product(r_self.arcs(q1), r_fsa.arcs(q2)):
                if a1 == a2:
                    # add new arc in the intersection
                    product_fsa.set_arc(
                        PairState(j1, j2), a1, PairState(q1, q2), w=w1 * w2
                    )
                    # add new node to the visiting queue
                    if (j1, j2) not in visited:
                        stack.append((j1, j2))
                        visited.add((j1, j2))

            if q1 in self_initials and q2 in fsa_initials:
                product_fsa.add_I(
                    PairState(q1, q2), w=self_initials[q1] * fsa_initials[q2]
                )
        # final state handling
        for (q1, w1), (q2, w2) in product(self.F, fsa.F):
            product_fsa.add_F(PairState(q1, q2), w=w1 * w2)

        return product_fsa

    def intersect_brute(self, other: "FSA") -> "FSA":
        """
        this methods performs a weighted brute-force intersection
        """

        return self.to_identity_fst().compose_brute(other.to_identity_fst()).project(1)

    def intersect(self, other: "FSA", eps_free: bool = False) -> "FSA":
        """This method performs an on-the-fly weighted intersection of two FSA.
        It works by keeping a stack of accessible states in the intersection WFSA.
        It uses the epsilon filter to handle epsilon transitions.

        Args:
            fsa (FSA): The other FSA to intersect with.
            eps_free (bool, optional): Whether to ignore epsilon transitions.

        Returns:
            FSA: The intersection FSA.
        """

        # the two machines need to be in the same semiring
        assert self.R == other.R

        return (
            self.to_identity_fst()
            .compose(other.to_identity_fst(), augment=not eps_free)
            .project(1)
        )

    def binary_intersection(self, other: "FSA") -> "FSA":
        """Performs a binary (set-based) intersection between two FSAs.

        Args:
            other (FSA): The FSA to intersect with.

        Returns:
            FSA: The intersection FSA.
        """
        return (self.complement()).union(other.complement()).trim().complement()

    def invert(self):
        """computes inverse"""

        inv = self.spawn(keep_init=True, keep_final=True)

        for i in self.Q:
            for a, j, w in self.arcs(i):
                inv.add_arc(i, a, j, ~w)

        return inv

    def complete(self):
        """constructs a complete FSA"""

        nfsa = self.copy()

        sink = State(self.num_states)
        for a in self.Sigma:
            nfsa.add_arc(sink, a, sink, self.R.one)

        for q in self.Q:
            if q == sink:
                continue
            for a in self.Sigma:
                if a == ε:  # ignore epsilon
                    continue
                if q not in nfsa.δ or not any(
                    value == a for value, _, _ in nfsa.arcs(q)
                ):
                    nfsa.add_arc(q, a, sink, self.R.one)

        return nfsa

    def complement(self):
        """create the complement of the machine"""

        assert self.deterministic
        was_bool = self.R == Boolean

        one = self.R.one
        tfsa = self.complete()
        C = FSA(R=tfsa.R)

        for q in tfsa.Q:
            if was_bool:
                for a, j, w in tfsa.arcs(q, no_eps=True):
                    C.add_arc(q, a, j, w)
            else:
                arcs = [(a, j) for a, j, _ in tfsa.arcs(q, no_eps=True)]
                Z = len(arcs)
                if self.ρ[q] == self.R.zero:
                    # If the state wasn't final, it will be now
                    Z += 1
                    C.set_F(q, one / self.R(Z))
                for a, j in arcs:
                    C.add_arc(q, a, j, one / self.R(Z))

        for q, w in self.I:
            C.set_I(q, w)

        if was_bool:
            finals = set([q for q, _ in tfsa.F])
            for q in tfsa.Q:
                if q not in finals:
                    C.set_F(q, one)

        return C

    def to_identity_fst(self) -> "rayuela.fsa.fst.FST":
        """Method converts FSA to FST.

        Returns:
            FST: FST object that accepts X:X iff `self` accepts X.
        """
        from rayuela.fsa.fst import FST

        fst = FST(self.R)
        for q in self.Q:
            for a, j, w in self.arcs(q):
                fst.add_arc(i=q, a=a, b=a, j=j, w=w)
        for q, w in self.I:
            fst.set_I(q=q, w=w)
        for q, w in self.F:
            fst.set_F(q=q, w=w)
        return fst

    def sample(
        self, K: int = 1, method: str = "normal", to_str: bool = True, sep: str = " "
    ) -> Union[
        List[str],
        List[List[Sym]],
        List[Tuple[str, List[Sym]], Tuple[List[Sym], List[Sym]]],
    ]:
        from rayuela.fsa.sampler import Sampler

        sampler = Sampler(self)

        if method == "normal":
            return sampler.sample(K, to_str=to_str, sep=sep)
        elif method == "lm":
            return sampler.sample(K, to_str=to_str, sep=sep, lm=True)
        elif method == "accessible_symbols":
            return sampler.sample_all_next(K, to_str=to_str, sep=sep)

    def to_uniform(self) -> "FSA":
        """Converts the FSA into a FSA whose topology is the same but the weights are
        uniform over the next transitions.

        Returns:
            FSA: The uniform FSA.
        """
        from rayuela.fsa.transformer import Transformer

        return Transformer.uniform(self)

    def tikz(self, max_per_row: int = 4) -> str:  # noqa: C901
        """Generates a bacis TikZ string for printing the FSA in LaTeX.
        It arranges the states of the FSA in a grid with `max_per_row` states per row.
        The code usually needs to be adjusted to fit the needs of the user.

        Args:
            max_per_row (int, optional): The maximum number of states to print in a row.
            Defaults to 4.

        Returns:
            str: The TikZ string.
        """

        tikz_string = []
        previous_ids, positioning = [], ""
        rows = {}

        initial = {q: w for q, w in self.I}
        final = {q: w for q, w in self.F}

        for jj, q in enumerate(self.Q):
            options = "state"
            additional = ""

            if q in initial:
                options += ", initial"
                additional = f" / {initial[q]}"
            if q in final:
                options += ", accepting"
                additional = f" / {final[q]}"

            if jj >= max_per_row:
                positioning = f"below = of {previous_ids[jj - max_per_row]}"
            elif len(previous_ids) > 0:
                positioning = f"right = of {previous_ids[-1]}"
            previous_ids.append(f"q{q.idx}")
            rows[q] = jj // max_per_row

            tikz_string.append(
                f"\\node[{options}] (q{q.idx}) [{positioning}]"
                + f"{{ ${q.idx}{additional}$ }}; \n"
            )

        tikz_string.append("\\draw")

        seen_pairs, drawn_pairs = set(), set()

        for jj, q in enumerate(self.Q):
            target_edge_labels = dict()
            for a, j, w in self.arcs(q):
                if j not in target_edge_labels:
                    target_edge_labels[j] = f"{a} / {w}"
                else:
                    target_edge_labels[j] += f"$\\\\${a} / {w}"
                seen_pairs.add(frozenset([q, j]))

            for ii, (target, label) in enumerate(target_edge_labels.items()):
                edge_options = "align=left"
                if q == target:
                    edge_options += ", loop above"
                elif frozenset([q, target]) not in seen_pairs:
                    edge_options += "a, bove"
                elif frozenset([q, target]) not in drawn_pairs:
                    if rows[q] == rows[target]:
                        edge_options += ", bend left, above"
                    else:
                        edge_options += ", bend left, right"
                else:
                    if rows[q] == rows[target]:
                        edge_options += ", bend left, below"
                    else:
                        edge_options += ", bend left, right"
                end = "\n"
                if jj == self.num_states - 1 and ii == len(target_edge_labels) - 1:
                    end = "; \n"
                tikz_string.append(
                    f"(q{q.idx}) edge[{edge_options}]"
                    + f" node{{ ${label}$ }} (q{target.idx}) {end}"
                )
                drawn_pairs.add(frozenset([q, target]))

        if not len(list(self.arcs(list(self.Q)[-1]))) > 0:
            tikz_string.append(";")

        return "".join(tikz_string)

    def __call__(self, str, eps_free=False):
        return self.accept(str, eps_free)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.difference(other)

    def __and__(self, other):
        return self.intersect(other)

    def __or__(self, other):
        return self.union(other)

    def __repr__(self):
        return f"WFSA({self.num_states} states, {self.R})"

    def __str__(self):
        output = []
        for q, w in self.I:
            output.append(f"initial state:\t{q.idx}\t{w}")
        for q, w in self.F:
            output.append(f"final state:\t{q.idx}\t{w}")
        for p in self.Q:
            for a, q, w in self.arcs(p):
                output.append(f"{p}\t----{a}/{w}---->\t{q}")
        return "\n".join(output)

    def __getitem__(self, n):
        return list(self.Q)[n]

    def __len__(self):
        return len(self.Q)

    def rename_states(self) -> "FSA":
        """Renames the states with names from 0 to N-1,
        where N is the number of states.
        This is useful after performing transformations which augment the state space,
        such as determinization of intersection.

        Returns:
            FSA: Strongly equivalent FSA with renamed states.
        """

        A = self.spawn()

        q2ix = {q: ix for ix, q in enumerate(self.finish())}

        for q in self.Q:
            for a, j, w in self.arcs(q):
                A.add_arc(State(q2ix[q]), a, State(q2ix[j]), w)

        for q, w in self.I:
            A.add_I(State(q2ix[q]), w)

        for q, w in self.F:
            A.add_F(State(q2ix[q]), w)

        return A

    def degrees(self, collapse_symbols: bool = True) -> Dict[State, int]:
        """Computes the out-degree of each state.

        Args:
            collapse_symbols (bool, optional): Whether to collapse the symbols.

        Returns:
            Dict[State, int]: The out-degree of each state.
        """
        if not collapse_symbols:
            return {q: len([a for a, _, _ in self.arcs(q)]) for q in self.Q}
        else:
            return {q: len(set([a for a, _, _ in self.arcs(q)])) for q in self.Q}

    def bfs_with_max_level(self, max_level: int):
        """Performs a breadth-first search of the FSA up to a maximum level.

        Args:
            max_level (int): The maximum level to search up to.
        """
        results = {q: [] for (q, _) in self.I}

        def _bfs(q: State):
            queue = deque([(q, 0)])  # Include level information

            while queue:
                qʼ, level = queue.popleft()

                if level > max_level:
                    break

                results[q].append((qʼ, level, len(set(a for a, _, _ in self.arcs(qʼ)))))

                for _, qʼʼ, _ in self.arcs(qʼ):
                    queue.append((qʼʼ, level + 1))

        for q, _ in self.I:
            _bfs(q)

        return results

    def _repr_html_(self):  # noqa: C901
        """
        When returned from a Jupyter cell, this will generate the FST visualization
        Based on: https://github.com/matthewfl/openfst-wrapper
        """
        import json
        from collections import defaultdict
        from uuid import uuid4

        from rayuela.base.semiring import ProductSemiring, Real, String
        from rayuela.fsa.fst import FST

        def weight2str(w):
            if isinstance(w, Real):
                return f"{w.value:.3f}"
            return str(w)

        ret = []
        if self.num_states == 0:
            return "<code>Empty FST</code>"

        if self.num_states > 64:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        finals = {q for q, _ in self.F}
        initials = {q for q, _ in self.I}

        # print initial
        for q, w in self.I:
            if q in finals:
                label = f"{str(q)} / [{weight2str(w)} / {str(self.ρ[q])}]"
                color = "af8dc3"
            else:
                label = f"{str(q)} / {weight2str(w)}"
                color = "66c2a5"

            ret.append(
                f'g.setNode("{repr(q)}", '
                + f'{{ label: {json.dumps(label)} , shape: "circle" }});\n'
            )

            ret.append(f'g.node("{repr(q)}").style = "fill: #{color}"; \n')

        # print normal
        for q in (self.Q - finals) - initials:
            lbl = str(q)

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #8da0cb"; \n')

        # print final
        for q, w in self.F:
            # already added
            if q in initials:
                continue

            if w == self.R.zero:
                continue
            lbl = f"{str(q)} / {weight2str(w)}"

            ret.append(
                f'g.setNode("{repr(q)}",{{label:{json.dumps(lbl)},shape:"circle"}});\n'
            )
            ret.append(f'g.node("{repr(q)}").style = "fill: #fc8d62"; \n')

        for q in self.Q:
            to = defaultdict(list)
            for a, j, w in self.arcs(q):
                if self.R is ProductSemiring and isinstance(w.value[0], String):
                    # the imporant special case of encoding transducers
                    label = f"{repr(a)}:{weight2str(w)}"
                elif isinstance(self, FST):
                    label = f"{repr(a[0])}:{repr(a[1])} / {weight2str(w)}"
                else:
                    a = str(repr(a))[1:-1]
                    label = f"{a} / {weight2str(w)}"
                to[j].append(label)

            for d, values in to.items():
                if len(values) > 6:
                    values = values[0:3] + [". . ."]
                label, qrep, drep = json.dumps("\n".join(values)), repr(q), repr(d)
                color = "rgb(192, 192, 192)" if self.theme == "dark" else "#333"
                edge_string = (
                    f'g.setEdge("{qrep}","{drep}",{{arrowhead:"vee",'
                    + f'label:{label},"style": "stroke: {color}; fill: none;", '
                    + f'"labelStyle": "fill: {color}; stroke: {color}; ", '
                    + f'"arrowheadStyle": "fill: {color}; stroke: {color};"}});\n'
                )
                ret.append(edge_string)

        # if the machine is too big, do not attempt to make the web browser display it
        # otherwise it ends up crashing and stuff...
        if len(ret) > 256:
            return (
                "FST too large to draw graphic, use fst.ascii_visualize()<br />"
                + f"<code>FST(num_states={self.num_states})</code>"
            )

        ret2 = [
            """
       <script>
       try {
       require.config({
       paths: {
       "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3",
       "dagreD3": "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min"
       }
       });
       } catch {
       ["https://cdnjs.cloudflare.com/ajax/libs/d3/4.13.0/d3.js",
       "https://cdnjs.cloudflare.com/ajax/libs/dagre-d3/0.6.1/dagre-d3.min.js"].forEach(
            function (src) {
            var tag = document.createElement('script');
            tag.src = src;
            document.body.appendChild(tag);
            }
        )
        }
        try {
        requirejs(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        try {
        require(['d3', 'dagreD3'], function() {});
        } catch (e) {}
        </script>
        <style>
        .node rect,
        .node circle,
        .node ellipse {
        stroke: #333;
        fill: #fff;
        stroke-width: 1px;
        }

        .edgePath path {
        stroke: #333;
        fill: #333;
        stroke-width: 1.5px;
        }
        </style>
        """
        ]

        obj = "fst_" + uuid4().hex
        ret2.append(
            f'<center><svg width="850" height="600" id="{obj}"><g/></svg></center>'
        )
        ret2.append(
            """
        <script>
        (function render_d3() {
        var d3, dagreD3;
        try { // requirejs is broken on external domains
          d3 = require('d3');
          dagreD3 = require('dagreD3');
        } catch (e) {
          // for google colab
          if(typeof window.d3 !== "undefined" && typeof window.dagreD3 !== "undefined"){
            d3 = window.d3;
            dagreD3 = window.dagreD3;
          } else { // not loaded yet, so wait and try again
            setTimeout(render_d3, 50);
            return;
          }
        }
        //alert("loaded");
        var g = new dagreD3.graphlib.Graph().setGraph({ 'rankdir': 'LR' });
        """
        )
        ret2.append("".join(ret))

        ret2.append(f'var svg = d3.select("#{obj}"); \n')
        ret2.append(
            """
        var inner = svg.select("g");

        // Set up zoom support
        var zoom = d3.zoom().scaleExtent([0.3, 5]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Create the renderer
        var render = new dagreD3.render();

        // Run the renderer. This is what draws the final graph.
        render(inner, g);

        // Center the graph
        var initialScale = 0.75;
        svg.call(zoom.transform, d3.zoomIdentity.translate(
            (svg.attr("width")-g.graph().width*initialScale)/2,20).scale(initialScale));

        svg.attr('height', g.graph().height * initialScale + 50);
        })();

        </script>
        """
        )

        return "".join(ret2)
