from collections import defaultdict
from functools import reduce
from typing import DefaultDict, Dict, Set, Tuple, Union

import numpy as np
from frozendict import frozendict
from numpy import linalg as LA

import rayuela
from rayuela.base.datastructures import PriorityQueue
from rayuela.base.semiring import Semiring
from rayuela.base.symbol import Sym, φ
from rayuela.fsa.aggregator import Aggregator
from rayuela.base.state import State


class Strategy:
    VITERBI = 1
    BELLMANFORD = 2
    DIJKSTRA = 3
    LEHMANN = 4
    JOHNSON = 5
    FIXPOINT = 6
    DECOMPOSED_LEHMANN = 7
    FAILURE_MEMORIZATION = 8
    FAILURE_RING = 9
    FAILURE_GENERAL = 10

    @staticmethod
    def str2strategy(name: str) -> int:  # noqa: C901
        """Returns the strategy corresponding to the given name.

        Args:
            name (str): The name of the strategy.

        Returns:
            int: The corresponding strategy.
        """
        if name.lower() == "viterbi":
            return Strategy.VITERBI
        elif name.lower() == "bellmanford":
            return Strategy.BELLMANFORD
        elif name.lower() == "dijkstra":
            return Strategy.DIJKSTRA
        elif name.lower() == "lehmann":
            return Strategy.LEHMANN
        elif name.lower() == "johnson":
            return Strategy.JOHNSON
        elif name.lower() == "fixpoint":
            return Strategy.FIXPOINT
        elif name.lower() == "decomposed_lehmann":
            return Strategy.DECOMPOSED_LEHMANN
        elif name.lower() == "failure_memorization":
            return Strategy.FAILURE_MEMORIZATION
        elif name.lower() == "failure_ring":
            return Strategy.FAILURE_RING
        elif name.lower() == "failure_general":
            return Strategy.FAILURE_GENERAL
        else:
            raise ValueError(f"Unknown strategy {name}")


class FailureBackward:
    def __init__(self, A: "rayuela.fsa.fsa.FSA"):
        self.A = A
        self.R = A.R
        self._β = dict()
        self.Σ = tuple(self.A.Sigma)
        self._β = defaultdict(lambda: self.R.zero)
        self._β_S = defaultdict(lambda: defaultdict(lambda: self.R.zero))

        Aφ, Ts = self._failure_trees()
        self.intervals = Aφ.reverse().dfs(intervals=True)[1]

    def β(self, q: State):
        if self._β[q] == self.R.zero:
            self._β[q] = self.β_Σ(q) + self.A.ρ[q]
        return self._β[q]

    def β_S(self, q: State, S: Set[Sym]):
        if self._β_S[q][S] == self.R.zero:
            self._β_S[q][S] = sum(list(self.β_a(q, a) for a in S), self.R.zero)
        return self._β_S[q][S]

    def β_Σ(self, q: State):  # Line 5
        Σ_q = tuple(self.A.out_symbols(q, ignore_phi=True))
        if φ not in list(self.A.out_symbols(q)):
            return self.β_S(q, Σ_q)
        return (
            self.β_S(self.A.qφ(q), self.Σ)
            + self.β_S(q, Σ_q)
            - self.β_S(self.A.qφ(q), Σ_q)
        )  # Line 9

    def β_a(self, q: State, a: Sym):  # Line 10
        if a in set(self.A.out_symbols(q, ignore_phi=True)):
            r = self.A.R.zero
            for qʼ, w in self.A.a_out_arcs(q, a):
                r += w * self._β[qʼ]
            return r
        elif φ in set(self.A.out_symbols(q)):
            return self.β_a(self.A.qφ(q), a)
        else:
            return self.A.R.zero

    def ring_compute(self) -> Dict[State, Semiring]:
        for q in self.A.toposort(rev=True):
            self._β[q] = self.β(q)
        return self._β

    def _visit(self, γ, q):  # Algorithm 5 Line 1
        for a in self.A.out_symbols(q, ignore_phi=True):
            γ.set(a, self.β_a(q, a))

    def _leave(self, γ, q):  # Algorithm 5 Line 4
        γ.undo(len(list(self.A.out_symbols(q, ignore_phi=True))))

    def _visit_plus(self, γ, q, qT):  # Algorithm 6 Line 14
        if self.A.qφ(q) != qT:
            self._visit_plus(γ, self.A.qφ(q), qT)
        self._visit(γ, q)

    def _descendant(self, q, qʼ) -> bool:  # Footnote 9
        """Tests whether q is a descendant of qʼ"""
        return (
            self.intervals[q][0] <= self.intervals[qʼ][0]
            and self.intervals[q][1] >= self.intervals[qʼ][1]
        )


class Pathsum:
    def __init__(self, fsa):
        # basic FSA stuff
        self.fsa = fsa
        self.R = fsa.R
        self.N = self.fsa.num_states

        # state dictionary
        self.I = {}  # noqa: E741
        for n, q in enumerate(self.fsa.Q):
            self.I[q] = n

        # lift into the semiring
        self.W = self.lift()

    def _convert(self):
        mat = np.zeros((self.N, self.N))
        for n in range(self.N):
            for m in range(self.N):
                mat[n, m] = self.W[n, m].value
        return mat

    def max_eval(self):
        # computes the largest eigenvalue
        mat = self._convert()
        if len(mat) == 0:
            return 0.0
        vals = []
        for val in LA.eigvals(mat):
            vals.append(np.abs(val))
        return np.max(vals)

    def lift(self):
        """creates the weight matrix from the automaton"""
        W = self.R.zeros(self.N, self.N)
        for p in self.fsa.Q:
            for a, q, w in self.fsa.arcs(p):
                W[self.I[p], self.I[q]] += w
        return W

    def pathsum(self, strategy: Union[int, str]) -> Semiring:  # noqa: C901
        if isinstance(strategy, str):
            strategy = Strategy.str2strategy(strategy)

        if strategy == Strategy.DIJKSTRA:
            assert self.R.superior, "Dijkstra's requires a superior semiring"
            return self.dijkstra_early()

        elif strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_pathsum()

        elif strategy == Strategy.BELLMANFORD:
            assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
            return self.bellmanford_pathsum()

        elif strategy == Strategy.JOHNSON:
            assert self.R.idempotent, "Johnson's requires an idempotent semiring"
            return self.johnson_pathsum()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_pathsum()

        elif strategy == Strategy.FIXPOINT:
            return self.fixpoint_pathsum()

        elif strategy == Strategy.DECOMPOSED_LEHMANN:
            return self.decomposed_lehmann_pathsum()

        elif strategy == Strategy.FAILURE_MEMORIZATION:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.memorization_pathsum()

        elif strategy == Strategy.FAILURE_RING:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.failure_ring_pathsum()

        elif strategy == Strategy.FAILURE_GENERAL:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.general_failure_pathsum()

        else:
            raise NotImplementedError

    def forward(self, strategy):
        if strategy == Strategy.DIJKSTRA:
            assert self.R.superior, "Dijkstra's requires a superior semiring"
            return self.dijkstra_fwd()

        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_fwd()

        elif strategy == Strategy.BELLMANFORD:
            assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
            return self.bellmanford_fwd()

        elif strategy == Strategy.JOHNSON:
            assert self.R.idempotent, "Johnson's requires an idempotent semiring"
            return self.johnson_fwd()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_fwd()

        elif strategy == Strategy.FIXPOINT:
            return self.fixpoint_fwd()
        else:
            raise NotImplementedError

    def backward(self, strategy: int) -> DefaultDict[State, Semiring]:  # noqa: C901
        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.viterbi_bwd()

        elif strategy == Strategy.BELLMANFORD:
            assert self.R.idempotent, "Bellman-Ford requires an idempotent semiring"
            return self.bellmanford_bwd()

        elif strategy == Strategy.JOHNSON:
            assert self.R.idempotent, "Johnson's requires an idempotent semiring"
            return self.johnson_bwd()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann_bwd()

        elif strategy == Strategy.FIXPOINT:
            return self.fixpoint_bwd()

        elif strategy == Strategy.FAILURE_MEMORIZATION:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.memorization_backward()

        elif strategy == Strategy.FAILURE_RING:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.failure_ring_backward()

        elif strategy == Strategy.FAILURE_GENERAL:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
            return self.general_failure_backward()

        else:
            raise NotImplementedError

    def allpairs(self, strategy=Strategy.LEHMANN, zero=True):
        if strategy == Strategy.VITERBI:
            assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"

        elif strategy == Strategy.JOHNSON:
            assert self.R.idempotent, "Johnson's requires an idempotent semiring"
            return self.johnson()

        elif strategy == Strategy.LEHMANN:
            return self.lehmann(zero=zero)

        elif strategy == Strategy.FIXPOINT:
            return self.fixpoint()

        else:
            raise NotImplementedError

    def allpairs_pathsum(self, W):
        pathsum = self.R.zero
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                pathsum += self.fsa.λ[p] * W[p, q] * self.fsa.ρ[q]
        return pathsum

    def allpairs_fwd(self, W):
        α = self.R.chart()
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                α[q] += self.fsa.λ[p] * W[p, q]
        return frozendict(α)

    def allpairs_bwd(self, W):
        β = self.R.chart()
        W = self.lehmann()
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                β[p] += W[p, q] * self.fsa.ρ[q]
        return frozendict(β)

    def viterbi_pathsum(self):
        pathsum = self.R.zero
        β = self.viterbi_bwd()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * β[q]
        return pathsum

    def memorization_pathsum(self):
        assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
        pathsum = self.R.zero
        β = self.memorization_backward()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * β[q]
        return pathsum

    def failure_ring_pathsum(self):
        assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
        pathsum = self.R.zero
        β = self.failure_ring_backward()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * β[q]
        return pathsum

    def general_failure_pathsum(self):
        assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
        pathsum = self.R.zero
        β = self.general_failure_backward()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * β[q]
        return pathsum

    def viterbi_fwd(self):
        """The Viterbi algorithm run forwards."""

        assert self.fsa.acyclic

        # chart
        α = self.R.chart()

        # base case (paths of length 0)
        for q, w in self.fsa.I:
            α[q] = w

        # recursion
        for p in self.fsa.toposort(rev=False):
            for _, q, w in self.fsa.arcs(p):
                α[q] += α[p] * w

        return α

    def viterbi_bwd(self) -> DefaultDict[State, Semiring]:
        """The Viterbi algorithm run backwards"""

        assert self.fsa.acyclic

        # chart
        β = self.R.chart()

        # base case (paths of length 0)
        for q, w in self.fsa.F:
            β[q] = w

        # recursion
        for p in self.fsa.toposort(rev=True):
            for _, q, w in self.fsa.arcs(p):
                β[p] += w * β[q]

        return β

    def dijkstra_early(self, stopearly=True):
        """Dijkstra's algorithm with early stopping."""

        assert self.fsa.R.superior

        # initialization
        α = self.R.chart()
        agenda = PriorityQueue(R=self.fsa.R)
        popped = set([])

        # base case
        for q, w in self.fsa.I:
            agenda.push((q, False), w)

        # main loop
        while agenda:
            (i, stop), v = agenda.pop()

            if stop and stopearly:
                return v

            popped.add(i)
            α[i] += v
            for _, j, w in self.fsa.arcs(i):
                if j not in popped:
                    agenda.push((j, False), α[i] * w)

            agenda.push((i, True), v * self.fsa.ρ[i])

    def dijkstra_fwd(self, Is=None) -> DefaultDict[State, Semiring]:
        """Dijkstra's algorithm without early stopping."""

        assert self.fsa.R.superior

        # initialization
        α = self.R.chart()
        agenda = PriorityQueue(R=self.fsa.R)
        popped = set([])

        # base case
        if Is is None:
            for q, w in self.fsa.I:
                agenda.push(q, w)
        else:
            for q in Is:
                agenda.push(q, self.R.one)

        # main loop
        while agenda:
            i, v = agenda.pop()
            popped.add(i)
            α[i] += v

            for _, j, w in self.fsa.arcs(i):
                if j not in popped:
                    agenda.push(j, v * w)

        return α

    def _gauss_jordan(self):
        """
        Algorithm 4 from
        https://link.springer.com/content/pdf/10.1007%2F978-0-387-75450-5.pdf
        """

        # initialization
        last = self.W.copy()
        A = self.R.zeros(self.N, self.N)

        # basic iterations
        for k in range(self.N):
            A[k, k] = last[k, k].star()
            for i in range(self.N):
                for j in range(self.N):
                    if i != k or j != k:
                        A[i, j] = last[i, j] + last[i, k] * A[k, k] * last[k, j]
            last = A.copy()
            A = self.R.zeros(self.N, self.N)

        for k in range(self.N):
            A[k, k] += self.R.one

        return A

    def _lehmann(self, zero=True):
        """
        Lehmann's (1977) algorithm.
        """

        # initialization
        V = self.W.copy()
        U = self.W.copy()

        # basic iteration
        for j in range(self.N):
            V, U = U, V
            V = self.R.zeros(self.N, self.N)
            for i in range(self.N):
                for k in range(self.N):
                    # i ➙ j ⇝ j ➙ k
                    V[i, k] = U[i, k] + U[i, j] * U[j, j].star() * U[j, k]

        # post-processing (paths of length zero)
        if zero:
            for i in range(self.N):
                V[i, i] += self.R.one

        return V

    def lehmann(self, zero=True):
        # TODO: check we if we can't do away with this method.

        V = self._lehmann(zero=zero)

        W = {}
        for p in self.fsa.Q:
            for q in self.fsa.Q:
                if p in self.I and q in self.I:
                    W[p, q] = V[self.I[p], self.I[q]]
                elif p == q and zero:
                    W[p, q] = self.R.one
                else:
                    W[p, q] = self.R.zero

        return frozendict(W)

    def lehmann_pathsum(self):
        return self.allpairs_pathsum(self.lehmann())

    def lehmann_fwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_fwd(self.lehmann())

    def lehmann_bwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_bwd(self.lehmann())

    def decomposed_lehmann_pathsum(self):
        from rayuela.fsa.scc import SCC

        sccs = SCC(self.fsa)

        # compute the forward values in a decomposed way
        # they are identical to the forward values in the full FSA
        αs = {}
        for scc in sccs.scc():
            α = Pathsum(sccs.to_fsa(scc, αs)).forward(Strategy.LEHMANN)
            for i in scc:
                αs[i] = α[i]

        # compute the actual pathsum
        return reduce(lambda x, y: x + y, [αs[i] * w for i, w in self.fsa.F])

    def bellmanford_pathsum(self):
        pathsum = self.R.zero
        𝜷 = self.bellmanford_bwd()
        for q in self.fsa.Q:
            pathsum += self.fsa.λ[q] * 𝜷[q]
        return pathsum

    def bellmanford_fwd(self) -> DefaultDict[State, Semiring]:
        # initialization
        α = self.R.chart()
        for q in self.fsa.Q:
            α[q] = self.fsa.λ[q]

        # main loop
        for _ in range(self.fsa.num_states):
            for i in self.fsa.Q:
                for _, j, w in self.fsa.arcs(i):
                    α[j] += α[i] * w

        return frozendict(α)

    def bellmanford_bwd(self) -> DefaultDict[State, Semiring]:
        # initialization
        𝜷 = self.R.chart()
        for q in self.fsa.Q:
            𝜷[q] = self.fsa.ρ[q]

        # main loop
        for _ in range(self.fsa.num_states):
            for i in self.fsa.Q:
                for _, j, w in self.fsa.arcs(i):
                    𝜷[i] += 𝜷[j] * w

        return frozendict(𝜷)

    def johnson(self):
        𝜷 = self.fsa.backward(Strategy.BELLMANFORD)
        pfsa = self.fsa.push()
        pathsum = Pathsum(pfsa)

        W = self.fsa.R.chart()
        for p in pfsa.Q:
            α = pathsum.dijkstra_fwd([p])
            for q, w in α.items():
                W[p, q] = 𝜷[p] * w * ~𝜷[q]

        return W

    def johnson_pathsum(self):
        return self.allpairs_pathsum(self.johnson())

    def johnson_fwd(self):
        return self.allpairs_fwd(self.johnson())

    def johnson_bwd(self):
        return self.allpairs_bwd(self.johnson())

    def _iterate(self, K):
        P = self.R.diag(self.N)
        for n in range(K):
            P += self.W @ P
        return P

    def _fixpoint(self, K=200):
        if self.fsa.R.idempotent:
            return self._iterate(self.fsa.num_states)

        diag = self.R.diag(self.fsa.num_states)
        P_old = diag

        # TODO: add an approximate stopping criterion

        # fixed point iteration
        # while True:
        for _ in range(K):
            P_new = diag + self.W @ P_old

            # if P_new == P_old:
            # 	return P_old
            P_old = P_new

        return P_old

    def fixpoint(self):
        P = self._fixpoint()
        W = {}

        for p in self.fsa.Q:
            for q in self.fsa.Q:
                if p in self.I and q in self.I:
                    W[p, q] = P[self.I[p], self.I[q]]
                elif p == q:
                    W[p, q] = self.R.one
                else:
                    W[p, q] = self.R.zero

        return frozendict(W)

    def fixpoint_pathsum(self):
        return self.allpairs_pathsum(self.fixpoint())

    def fixpoint_fwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_fwd(self.fixpoint())

    def fixpoint_bwd(self) -> DefaultDict[State, Semiring]:
        return self.allpairs_bwd(self.fixpoint())

    def memorization_backward(self) -> DefaultDict[State, Semiring]:
        """Svete et al. (2022) Algorithm 3"""
        assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"
        A = self.fsa

        β_q = defaultdict(lambda: self.R.zero)
        β_qa = defaultdict(lambda: self.R.zero)
        β_φ = defaultdict(lambda: self.R.zero)

        for q in A.toposort(rev=True):  # Line 2
            for a in A.out_symbols(q, ignore_phi=True):  # Line 3
                for qʼ, w in A.a_out_arcs(q, a):
                    β_qa[(q, a)] += w * β_q[qʼ]

            if A.has_fallback_state(q):  # Line 6
                # Line 7
                for b in set(A.Sigma) - set(A.out_symbols(q)) - {φ}:
                    β_qa[(q, b)] = β_qa[(A.qφ(q), b)]
                    β_φ[(q, "Σ - Σ(q)")] += β_qa[(A.qφ(q), b)]

            β_φ[(q, "Σ(q)")] = sum(
                list(β_qa[(q, a)] for a in A.out_symbols(q)), self.R.zero
            )
            β_φ[(q, "Σ")] += β_φ[(q, "Σ(q)")] + β_φ[(q, "Σ - Σ(q)")]  # Line 10

            β_q[q] = β_φ[(q, "Σ")] + A.ρ[q]  # Line 11

        return β_q

    def failure_ring_backward(self) -> Dict[State, Semiring]:
        """Svete et al. (2022) Algorithm 4"""
        assert self.fsa.acyclic, "Viterbi requires an acyclic FSA"

        return FailureBackward(self.fsa).ring_compute()

    def _failure_trees(self) -> Tuple["rayuela.fsa.fsa.FSA", Dict[State, State]]:
        from rayuela.fsa.transformer import Transformer

        _, Aφ = Transformer.partition(self.fsa, partition_symbol=φ)

        Iφ, Fφ = set(Aφ.Q), set(Aφ.Q)
        for q in Aφ:
            for _, t, _ in Aφ.arcs(q):
                Iφ.discard(t)
                Fφ.discard(q)

        for q in Iφ:
            Aφ.add_I(q, Aφ.R.one)
        for q in Fφ:
            Aφ.add_F(q, Aφ.R.one)

        Ts = dict()
        Aφ_r = Aφ.reverse()
        for q in Fφ:
            U = set(Aφ_r.dfs(Is={q})[1].keys())
            for u in U:
                Ts[u] = q

        return Aφ, Ts

    def general_failure_backward(self) -> Dict[State, Semiring]:
        A = self.fsa
        γs, qs = dict(), dict()
        fb = FailureBackward(A)

        _, Ts = self._failure_trees()

        for q in A.toposort(rev=True):  # Line 2
            T = Ts[q]

            if not A.has_fallback_state(q):  # Line 4
                γs[T] = Aggregator(A.R, A.Sigma)
                fb._visit(γs[T], q)
                qs[T] = q
            else:
                while not fb._descendant(qs[T], q):  # Line 8
                    fb._leave(γs[T], qs[T])
                    qs[T] = A.qφ(qs[T])

                fb._visit_plus(γs[T], q, qs[T])  # Line 10
                qs[T] = q

            fb._β[q] = γs[T].value() + A.ρ[q]  # Line 12

        return fb._β

    def rank1(self):
        """
        Novel algorithm
        """
        B = self.sr.zeros(1, 1)

        # base case
        B[0, 0] = self.W[0, 0].star()

        # main loop
        for n in range(1, self.N):
            q = self.W[n, :n]
            p = self.W[:n, n]

            d = self.sr.star(self.W[n, n] + np.dot(np.dot(p, B), q))

            block1 = B + np.outer(np.dot(B, q) * d, np.dot(p, B))
            block2 = np.dot(B, p) * d
            # print(d, type(d))
            # print(B, type(B))
            # print(np.dot(p, B), type(np.dot(p, B)))
            block3 = d * np.dot(p, B)
            block4 = d

            # dimension hacking
            block2 = np.expand_dims(block2, axis=1)
            block3 = np.expand_dims(block3, axis=1)
            block4 = np.expand_dims(block4, axis=0)
            block4 = np.expand_dims(block4, axis=1)

            # combine
            tmp1 = np.concatenate([block1, block2], axis=1).squeeze()
            tmp2 = np.concatenate([block3, block4], axis=0).squeeze()
            B = np.vstack([tmp1, tmp2]).squeeze().item()

        return B
