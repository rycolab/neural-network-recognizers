import queue
from collections import defaultdict as dd
from itertools import chain, product
from typing import Set, Tuple

import numpy as np

from rayuela.base.partitions import PartitionRefinement
from rayuela.base.semiring import (
    Boolean,
    ProductSemiring,
    Real,
    Semiring,
    product_semiring_builder,
)
from rayuela.base.state import MinimizeState, PowerState, State
from rayuela.base.symbol import Concatenation, Expr, Star, Sym, Union, ε, φ
from rayuela.fsa import utils
from rayuela.fsa.fsa import FSA
from rayuela.fsa.fst import FST
from rayuela.fsa.pathsum import Pathsum


class Transformer:
    @staticmethod
    def _add_trim_arcs(F: FSA, T: FSA, AC: Set[State]):
        for i in AC:
            if isinstance(F, FST):
                for (a, b), j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, b, j, w)

            else:
                for a, j, w in F.arcs(i):
                    if j in AC:
                        T.add_arc(i, a, j, w)

    @staticmethod
    def trim(F: FSA) -> FSA:
        """trims the machine"""

        # compute accessible and co-accessible arcs
        A, C = F.accessible(), F.coaccessible()
        AC = A.intersection(C)

        # create a new F with only the pruned arcs
        T = F.spawn()
        Transformer._add_trim_arcs(F, T, AC)

        # add initial state
        for q, w in F.I:
            if q in AC:
                T.set_I(q, w)

        # add final state
        for q, w in F.F:
            if q in AC:
                T.set_F(q, w)

        return T

    @staticmethod
    def _powerarcs(fsa, Q):
        """This helper method group outgoing arcs for determinization."""

        symbol2arcs, unnormalized_residuals = dd(set), fsa.R.chart()

        for q, old_residual in Q.residuals.items():
            for a, p, w in fsa.arcs(q):
                symbol2arcs[a].add(p)
                unnormalized_residuals[(a, p)] += old_residual * w

        for a, ps in symbol2arcs.items():
            normalizer = sum(
                [unnormalized_residuals[(a, p)] for p in ps], start=fsa.R.zero
            )
            # this does not assume commutivity
            residuals = {p: ~normalizer * unnormalized_residuals[(a, p)] for p in ps}
            # this is an alternative formulation
            # residuals = {p: unnormalized_residuals[(a, p)] / normalizer for p in ps}

            yield a, PowerState(residuals), normalizer

    @staticmethod
    def determinize(fsa, timeout=1000):
        """
        The on-the-fly determinization method (Mohri 2009).
        Link: https://link.springer.com/chapter/10.1007/978-3-642-01492-5_6
        """
        D = fsa.spawn()

        # if fsa.R.cancellative:
        #     assert Transformer.twins(fsa), 'The FSA cannot be determinized.'
        # else:
        #     # TODO: better logging
        #     print('The semiring is not cancellative. Can not check twins property.')

        stack, visited = [], set([])
        Q = PowerState({p: w for p, w in fsa.I})
        D.set_I(Q, fsa.R.one)
        stack.append(Q)
        visited.add(Q)

        counter = 0
        while stack:
            if counter > timeout:
                raise TimeoutError

            P = stack.pop()
            for a, Q, w in Transformer._powerarcs(fsa, P):
                if Q not in visited:
                    stack.append(Q)
                    visited.add(Q)

                # TODO: can we propogate a change where we make this add_arc in FSA add?
                D.add_arc(P, a, Q, w)

            counter += 1

        for powerstate in D.Q:
            for q in powerstate.idx:
                D.add_F(powerstate, fsa.ρ[q] * powerstate.residuals[q])

        return D

    @staticmethod
    def cycle_identity(fsa):
        """From https://dl.acm.org/doi/abs/10.5555/873977.873979"""

        def _dfs(q):
            nonlocal in_progress
            nonlocal finished
            nonlocal weights
            nonlocal twins_property

            in_progress.add(q)

            for a, j, w in fsa.arcs(q, nozero=False):
                p = weights[q] * w
                if j not in weights:
                    weights[j] = p
                elif weights[j] != p:
                    twins_property = False

                if j not in finished and j not in in_progress:
                    _dfs(j)

            in_progress.remove(q)
            finished.add(q)

        twins_property = True

        for i, w in fsa.I:
            in_progress, finished = set([]), set([])
            weights = {i: fsa.R.one}
            _dfs(i)

        return twins_property

    @staticmethod
    def twins(fsa):
        """
        Alluzen and Mohri's (2003) algorithm for testing whether a WFSA
        has the twins property.
        Time complexity: O(Q² + E²)
        """
        from rayuela.fsa.scc import SCC

        F = fsa.intersect(fsa.invert())

        scc = SCC(F)
        for c in scc.scc():
            Fscc = scc.to_fsa(c)
            # print(list(Fscc.Q))
            # test the cycle identity on every SCC
            if not Transformer.cycle_identity(Fscc):
                return False

        return True

    @staticmethod
    def normalize(fsa):
        """locally normalize the fsa"""

        pfsa = fsa.push()
        nfsa = fsa.spawn()

        Z = fsa.R.zero
        for q, w in pfsa.I:
            Z += w
        for q, w in pfsa.I:
            nfsa.set_I(q, w / Z)

        for q, w in pfsa.F:
            nfsa.set_F(q, w)
        for i in pfsa.Q:
            for a, j, w in pfsa.arcs(i):
                nfsa.add_arc(i, a, j, w)

        return nfsa

    @staticmethod
    def push(fsa):
        from rayuela.fsa.pathsum import Strategy

        W = Pathsum(fsa).backward(Strategy.LEHMANN)
        pfsa = Transformer._push(fsa, W)
        # assert pfsa.pushed  # sanity check
        return pfsa

    @staticmethod
    def _push(fsa, V):
        """
        Mohri (2001)'s weight pushing algorithm. See Eqs 1, 2, 3.
        Link: www.isca-speech.org/archive_v0/archive_papers/eurospeech_2001/e01_1603.pdf
        """
        from rayuela.fsa.fst import FST

        pfsa = fsa.spawn()
        for i in fsa.Q:
            pfsa.set_I(i, fsa.λ[i] * V[i])
            pfsa.set_F(i, ~V[i] * fsa.ρ[i])
            for a, j, w in fsa.arcs(i):
                if isinstance(fsa, FST):
                    pfsa.add_arc(i, a[0], a[1], j, ~V[i] * w * V[j])
                else:
                    pfsa.add_arc(i, a, j, ~V[i] * w * V[j])

        return pfsa

    @staticmethod
    def lift_labels_to_weight(A: FSA) -> FSA:
        """Lifts the labels of an FSA to weights of the arcs such that the new labels
        are tuples of symbols and the original weights.

        Args:
            fsa (FSA): The FSA to lift.

        Returns:
            FSA: The lifted FSA.
        """
        R = product_semiring_builder(Expr, Expr)

        Aʼ = FSA(R)

        for q in A.Q:
            for a, j, w in A.arcs(q):
                Aʼ.add_arc(q, w, j, R(a, Sym(w.value)))

        for q, w in A.I:
            Aʼ.add_I(q, R(ε, Sym(w.value)))

        for q, w in A.F:
            Aʼ.add_F(q, R(ε, Sym(w.value)))

        return Aʼ

    @staticmethod
    def combine_regex_and_weight(R: ProductSemiring) -> Expr:
        """Takes the result of running the pathsum algorithm on a label-lifted FSA
        (see `lift_labels_to_weight`) and combines the regexes and weights into a
        single regex.
        ! NOTE: This only works for regexes with a single initial/final state!

        Args:
            R (ProductSemiring): The result of running the pathsum algorithm on a
                label-lifted FSA. It contains the regexes of labels and weights.

        Returns:
            Expr: The combined regex.
        """

        def _compute(_L: Expr, _W: Expr) -> Expr:
            if isinstance(_L, Sym) and isinstance(_W, Sym):
                combined = Expr(f"{str(_W.value)}{str(_L.value)}")
            elif isinstance(_L, Concatenation) and isinstance(_W, Concatenation):
                combined = _compute(_L.x, _W.x) * _compute(_L.y, _W.y)
            elif isinstance(_L, Union) and isinstance(_W, Union):
                combined = _compute(_L.x, _W.x) + _compute(_L.y, _W.y)
            elif isinstance(_L, Star) and isinstance(_W, Star):
                combined = _compute(_L.x, _W.x).star()
            else:
                raise ValueError

            return combined

        λ = Expr(R.value[1].y)
        ρ = Expr(R.value[1].x.x)
        E = _compute(R.value[0], R.value[1].x.y)
        return λ * E * ρ

    @staticmethod
    def lift_weights_to_labels(fsa):
        assert fsa.pushed
        assert fsa.deterministic
        # assert len([i for i in fsa.I]) == 1, "More than one initial state"
        if_weights = {"init": {}, "final": {}}

        lifted_fsa = FSA(R=Boolean)
        for i, w in fsa.I:
            if_weights["init"][i] = w
            lifted_fsa.add_I(i, Boolean.one)
        for f, w in fsa.F:
            if_weights["final"][f] = w
            lifted_fsa.add_F(f, Boolean.one)
        for p in fsa.Q:
            for a, q, w in fsa.arcs(p):
                lifted_fsa.add_arc(p, (a, w), q, Boolean.one)
        return lifted_fsa, if_weights

    @staticmethod
    def get_weights_from_labels(fsa, R, if_weights):
        # assert len([i for i in fsa.I]) == 1, "More than one initial state"

        wfsa = FSA(R=R)
        for i, _ in fsa.I:
            wfsa.add_I(i, if_weights["init"][list(i.idx).pop()])
        for f, _ in fsa.F:
            wfsa.add_F(f, if_weights["final"][list(f.idx).pop()])
        for p in fsa.Q:
            for label, q, _ in fsa.arcs(p):
                a, w = label.value
                wfsa.add_arc(p, a, q, w)
        return wfsa

    @staticmethod
    def _minimize(fsa, strategy=None):
        assert fsa.R == Boolean
        if strategy is None:
            return Transformer.minimize_fast(fsa)
        elif strategy == "partition":
            return Transformer.minimize_partition(fsa)

    @staticmethod
    def _construct_minimized(fsa, clusters):
        """Takes in the produced minimized states (subsets) and constructs a
        new FSA with those states and correct arcs between them."""

        # create new power states
        minstates = {}
        for qs in clusters:
            minstate = MinimizeState(frozenset(qs))
            for q in qs:
                minstates[q] = minstate

        # create minimized FSA
        mfsa = fsa.spawn()

        # add arcs
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                mfsa.add_arc(minstates[q], a, minstates[j], w=w)

        # add initial states
        for q, w in fsa.I:
            mfsa.add_I(minstates[q], w)

        # add final states
        for q, w in fsa.F:
            mfsa.add_F(minstates[q], w)

        # update the state _L
        # for _, minstate in minstates.items():
        #     labels = [str(q.label) for q in minstate.idx]
        #     minstate._label = "{ " + ",".join(sorted(labels)) + " }"

        return mfsa

    @staticmethod
    def minimize_partition(fsa):
        assert fsa.deterministic
        assert fsa.R == Boolean

        F = set([q for q, _ in fsa.F])
        P = frozenset([frozenset(F), frozenset(fsa.Q - F)])

        for a in fsa.Sigma:
            f_a = {}
            for p in fsa.Q:
                for b, q, _ in fsa.arcs(p):
                    if a == b:
                        f_a[p] = q

            Q = frozenset(f_a.keys())

            for q in fsa.Q - f_a.keys():
                f_a[q] = q

            P = PartitionRefinement(f_a, Q).hopcroft(P)
        return Transformer._construct_minimized(fsa, P)

    # @staticmethod
    # def minimize_partition(fsa):
    #     assert fsa.deterministic

    #     δ = dd(lambda: dd(int))
    #     for i in fsa.Q:
    #         for a, j, _ in fsa.arcs(i):
    #             δ[a][i] = j

    #     nonfinals, finals = set(), set()
    #     for q in fsa.Q:
    #         if fsa.ρ[q] != fsa.R.zero:
    #             nonfinals.add(q)
    #         else:
    #             finals.add(q)
    #     P = frozenset([frozenset(nonfinals), frozenset(finals)])

    #     for a in fsa.Sigma:
    #         P = PartitionRefinement(frozendict(δ[a]), fsa.Q).hopcroft(P)

    #     return Transformer._construct_minimized(fsa, P)

    @staticmethod
    def minimize_fast(fsa):
        """Hopcroft's O(V log E) algorithm"""

        assert fsa.deterministic
        assert fsa.R == Boolean

        # calculate inverse of E
        δinv = {}
        for q in fsa.Q:
            for a, j, _ in fsa.arcs(q):
                if j not in δinv:
                    δinv[j] = dict()
                if a not in δinv[j]:
                    δinv[j][a] = set()
                δinv[j][a].add(q)

        final = frozenset([q for q, _ in fsa.F])
        nonfinal = frozenset(fsa.Q - final)

        P = set([final, nonfinal])
        W = [nonfinal, final]
        while W:
            A = W.pop()
            for a in fsa.Sigma:
                X = frozenset(
                    set().union(*[δinv.get(q, {}).get(a, frozenset([])) for q in A])
                )

                for Y in [
                    Y for Y in P if Y.intersection(X) and Y.intersection(fsa.Q - X)
                ]:
                    P.remove(Y)
                    YX = Y.intersection(X)
                    Y_X = Y.intersection(fsa.Q - X)
                    P.add(YX)
                    P.add(Y_X)

                    W.append(YX if len(YX) < len(Y_X) else Y_X)
        return Transformer._construct_minimized(fsa, P)

    @staticmethod
    def minimize_brzozowski(fsa):
        """
        Implements Brzozowski's_algorithm.
        See https://ralphs16.github.io/src/CatLectures/HW_Brzozowski.pdf and
        https://link.springer.com/chapter/10.1007/978-3-642-39274-0_17.
        """
        return Transformer.trim(
            Transformer.determinize(
                Transformer.trim(Transformer.determinize(fsa.reverse().single_I()))
                .reverse()
                .single_I()
            )
        )

    @staticmethod
    def _heights(fsa):
        F = fsa.reverse()

        heights = dd(int)

        def _dfs(p, c):
            nonlocal heights

            for _, q, _ in F.arcs(p):
                _dfs(q, c + 1)

            heights[p] = max(c, heights[p])

        for q, _ in F.I:
            _dfs(q, 0)

        return heights

    # TODO: test this out and make it less complex (remove noqa: C901)
    @staticmethod
    def minimize_revuz(fsa):  # noqa: C901
        assert fsa.deterministic

        assert fsa.acyclic

        δinv = {}
        for q in fsa.Q:
            for a, j, w in fsa.arcs(q):
                if j not in δinv:
                    δinv[j] = dict()
                if a not in δinv[j]:
                    δinv[j][a] = set()
                δinv[j][a].add((q, w))

        height = fsa._heights()

        π = dd(list)
        for q in height:
            π[height[q]].append(q)

        mfsa = fsa.spawn()

        state_map = {}
        for h in range(len(π)):
            A = dd(list)
            for s in π[h]:
                sm = MinimizeState(frozenset([s]))

                ########################################
                # Bucket Sort: O(n)
                # assumes labels are single characters
                ########################################
                n_min = min([ord(str(a)) for a, _, _ in mfsa.arcs(sm)], default=())
                S = dict()
                for a, j, _ in mfsa.arcs(sm):
                    S[ord(str(a)) - n_min] = (a, j)

                arcs = []
                for i in range(len(S)):
                    if i in S:
                        arcs.append(S[i])

                ########################################
                # Alternative, works for any kind of arc labels: O(nlogn)
                ########################################
                # arcs = sorted([a, j for a, j, _ in mfsa.arcs(sm)], key=lambda x: x[0])

                labels = tuple(chain.from_iterable(arcs))
                A[labels].append(s)

            for k, v in A.items():
                if len(v) > 1:
                    for u in v:
                        t = MinimizeState(frozenset(v))
                        state_map[MinimizeState(frozenset([u]))] = t

                        # remove the state from mfsa
                        # there will only be outgoing edges from them
                        mfsa.Q.remove(MinimizeState(frozenset([u])))
                        del mfsa.δ[MinimizeState(frozenset([u]))]

                else:
                    t = MinimizeState(frozenset(v))
                    state_map[t] = t

                for u in v:
                    for a, j, w in fsa.arcs(u):
                        mfsa.add_arc(t, a, state_map[MinimizeState(frozenset([j]))], w)
                    for a, transitions in δinv.get(u, {}).items():
                        for j, w in transitions:
                            mfsa.add_arc(MinimizeState(frozenset([j])), a, t, w)

        for q, w in fsa.F:
            mfsa.add_F(state_map[MinimizeState(frozenset([q]))], w)
        for q, w in fsa.I:
            mfsa.add_I(state_map[MinimizeState(frozenset([q]))], w)

        return mfsa

    @staticmethod
    def partition(fsa, partition_symbol: Sym = ε) -> Tuple[FSA, FSA]:
        """Partition FSA into two
        (one with arcs of the partition symbol and one with all others)

        Args:
            fsa (FSA): The input FSA
            partition_symbol (Sym, optional): The symbol based on which to
            partition the input FSA

        Returns:
            Tuple[FSA, FSA]: The FSA with non-partition symbol arcs
                             and the FSA with only the partition symbol arcs
        """

        E = fsa.spawn()
        N = fsa.spawn(keep_init=True, keep_final=True)

        for q in fsa.Q:
            E.add_state(q)
            N.add_state(q)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                if a == partition_symbol:
                    E.add_arc(i, a, j, w)
                else:
                    N.add_arc(i, a, j, w)

        return N, E

    @staticmethod
    def epsremoval(fsa):
        # note that N keeps same initial and final weights
        N, E = Transformer.partition(fsa)
        W = Pathsum(E).lehmann(zero=False)

        for i in fsa.Q:
            for a, j, w in fsa.arcs(i, no_eps=True):
                for k in fsa.Q:
                    N.add_arc(i, a, k, w * W[j, k])

        # additional initial states
        for i, j in product(fsa.Q, repeat=2):
            N.add_I(j, fsa.λ[i] * W[i, j])

        return N

    @staticmethod
    def _phi_transitive_closure(
        fsa: FSA,
        q: State,
        single_hops: dd[State, dd[Sym, dd[State, Semiring]]],
    ) -> None:
        """Determines the one-hop arcs possible from a given state `q` according to
        the original transition function and the failure arcs.

        Args:
            fsa (FSA): The input FSA
            q (State): The current state
            single_hops (dd[State, dd[Sym, dd[State, Semiring]]]):
                The dictionary of one-hop arcs found so far in the backward traversal of
                the failure arcs.
        """

        # Keep the list of all taken transitions, and the failure target for `q`
        valid_transitions, fallback_target = set(), None
        for a, t, w in fsa.arcs(q):
            if a == φ:
                fallback_target = t
            else:
                # If the transition is not φ, add it directly to the expanded FSA
                valid_transitions.add(a)
                single_hops[q][a][t] = w

        # If `q` has no fallback target, we are done
        if fallback_target is None:
            return

        # If there is an outgoing failure arc,
        # add the non-existing outgoing transitions
        # according to the fallback target
        for a in single_hops[fallback_target]:
            if a in (fsa.Sigma - valid_transitions - set([φ])):
                for t, w in single_hops[fallback_target][a].items():
                    single_hops[q][a][t] = w

    @staticmethod
    def _failure_expanded_fsa(
        fsa: FSA,
        single_hops: dd[State, dd[Sym, dd[State, Semiring]]],
    ) -> FSA:
        """Construct the expanded FSA based on the one-hop arcs
        found by traversing the failure arcs.

        Args:
            fsa (FSA): The input FSA
            single_hops (dd[State, dd[Sym, dd[State, Semiring]]]):
                The one-hop transitions possible from each state based on the original
                transition function and the failure arcs

        Returns:
            FSA: The failure-arc expanded FSA
        """

        Aʼ = FSA(fsa.R)
        for q in fsa.Q:
            for a in single_hops[q]:
                for t, w in single_hops[q][a].items():
                    Aʼ.add_arc(q, a, t, w)

        for q, w in fsa.I:
            Aʼ.add_I(q, w)
        for q, w in fsa.F:
            Aʼ.add_F(q, w)

        return Aʼ

    @staticmethod
    def expand_phi_arcs(fsa: FSA) -> FSA:
        """Generates an equivalent FSA without failure (φ) transitions
           by creating the missing transitions to the fallback state.
           We assume all failure transitions have weight 1,
           that there is at most 1 outgoing failure arc per state,
           and that they form an *acyclic* subgraph in the FSA.
        Args:
            fsa (FSA): The input FSA with failure arcs according to the
                specifications above.

        Returns:
            FSA: The FSA with the failure arcs expanded.
        """

        _, Aφ = Transformer.partition(fsa, φ)

        # Hacky: preprocess the φ-only FSA to have initial states
        Iφ = set(Aφ.Q)
        for q in Aφ:
            for _, t, _ in Aφ.arcs(q):
                # Remove the target state from the states with no incoming edges
                Iφ.discard(t)

        for q in Iφ:
            Aφ.add_I(q, fsa.R.one)

        # We assume that E is *acyclic*
        assert Aφ.acyclic

        # Keep the map of the transitions accessible using failure arcs
        # from all the states in the FSA
        hops = dd(lambda: dd(lambda: dd(lambda: fsa.R.zero)))

        for q in Aφ.toposort(rev=True):
            Transformer._phi_transitive_closure(fsa, q, hops)

        # After determining all one-hop transitions considering the failure arcs,
        # create the new expanded FSA
        Aʼ = Transformer._failure_expanded_fsa(fsa, hops)

        return Aʼ

    @staticmethod
    def to_lin_alg(fsa):
        """Converts a fsa into its linear algebra representation
        (requires semiring to be a field).

        Params:
            fsa (FSA): The input fsa

        Returns:
            n: number of states
            Σ: alphabet of fsa symbols
            M: dictionary of symbols to transition matrices of the form n x n
            λ: vector of dim. n of real initial weights
            ρ: vector of dim. n of real final weights
        """
        assert fsa.R == Real
        return (fsa.num_states, fsa.Sigma, fsa.T, fsa.init_vector, fsa.final_vector)

    @staticmethod
    def from_lin_alg(n, Σ, M, λ, ρ):
        """Converts the linear algebrea representation of a fsa into
        an fsa object over the Real semiring.

        Params:
            n: number of states
            Σ: alphabet of fsa symbols
            M: dictionary of symbols to transition matrices of the form n x n
            λ: vector of dim. n of real initial weights
            ρ: vector of dim. n of real final weights

        Returns:
            a new FSA object
        """
        # create fsa
        fsa = FSA(Real)
        Q = [State(i) for i in range(n)]

        # add arcs
        for i in range(n):
            q = Q[i]
            for a in Σ:
                for j in range(n):
                    p = Q[j]
                    fsa.add_arc(q, a, p, w=M[a][i, j])

        # add initial states
        for i, w in enumerate(λ):
            fsa.add_I(Q[i], Real(w))

        # add final states
        for i, w in enumerate(ρ):
            fsa.add_F(Q[i], Real(w))

        return fsa

    @staticmethod
    def lin_alg_union(fsa1, fsa2):
        """Computes the union of two wfsa using linear algebra."""
        return Transformer.from_lin_alg(
            *Transformer._lin_alg_union(
                *Transformer.to_lin_alg(fsa1), *Transformer.to_lin_alg(fsa2)
            )
        )

    @staticmethod
    def _lin_alg_union(n1, Σ1, M1, λ1, ρ1, n2, Σ2, M2, λ2, ρ2):
        n = n1 + n2
        Σ = Σ1.union(Σ2)
        M = {}
        for a in Σ:
            M[a] = np.zeros((n, n))
            if a in M1:
                for i in range(n1):
                    for j in range(n1):
                        M[a][i, j] = M1[a][i, j]
            if a in M2:
                for i in range(n2):
                    for j in range(n2):
                        M[a][n1 + i, n1 + j] = M2[a][i, j]
        λ = np.append(λ1, λ2)
        ρ = np.append(ρ1, ρ2)
        return n, Σ, M, λ, ρ

    def lin_alg_difference(fsa1, fsa2):
        """Computes the difference of two wfsa using linear algebra."""
        return Transformer.from_lin_alg(
            *Transformer._lin_alg_difference(
                *Transformer.to_lin_alg(fsa1), *Transformer.to_lin_alg(fsa2)
            )
        )

    @staticmethod
    def _lin_alg_difference(n1, Σ1, M1, λ1, ρ1, n2, Σ2, M2, λ2, ρ2):
        return Transformer._lin_alg_union(n1, Σ1, M1, λ1, ρ1, n2, Σ2, M2, λ2, -ρ2)

    @staticmethod
    def forward_basis(fsa, vocab=False):
        """Computes a basis of the forward space of the fsa. If vocab is set to true
        also returns a basis vocabulary."""
        n, Σ, M, λ, ρ = Transformer.to_lin_alg(fsa)
        if not λ.any():
            return [λ], [""] if vocab else [λ]
        λ /= np.linalg.norm(λ)
        F = np.vstack([λ])
        V = [""]
        Q = queue.Queue()
        Q.put((λ, ""))
        while not Q.empty():
            b, string = Q.get()
            for a in Σ:
                b2 = b.dot(M[a])
                string2 = string + a.value
                if not utils.span_contains(F, b2):
                    b2 = utils.gram_schmidt(F, b2)
                    F = np.vstack([F, b2])
                    Q.put((b2, string2))
                    V.append(string2)
        return (F, V) if vocab else F

    @staticmethod
    def backward_basis(fsa, vocab=False):
        """Computes a basis of the backward space of the fsa."""
        B, V = Transformer.forward_basis(fsa.reverse(), vocab=True)
        B = np.transpose(B)
        # V = [v[::-1] for v in V]
        return (B, V) if vocab else B

    @staticmethod
    def is_zero(fsa):
        """Returns True if and only if the input fsa is zero, i.e.
        accepts no string with positive weight"""
        F = Transformer.forward_basis(fsa)
        ρ = fsa.final_vector
        v = F.dot(ρ)
        return utils.orthogonal(v, v)

    @staticmethod
    def equivalent_nfa(fsa1, fsa2):
        """Returns true if and only if fsa1 and fsa2 accept the same
        weighted language over the Real semiring."""
        assert fsa1.R == Real and fsa2.R == Real
        diff_fsa = Transformer.lin_alg_difference(fsa1, fsa2)
        return Transformer.is_zero(diff_fsa)

    @staticmethod
    def forward_conjugate(fsa):
        """Computes a forward minimal FSA equivalent to the input."""
        n, Σ, M, λ, ρ = Transformer.to_lin_alg(fsa)
        F = Transformer.forward_basis(fsa)
        n_conj = len(F)
        F_inv = F.transpose().dot(np.linalg.inv(F.dot(F.transpose())))
        M_conj = {}
        for a in Σ:
            M_conj[a] = F.dot(M[a]).dot(F_inv)
        λ_conj = λ.dot(F_inv)
        ρ_conj = F.dot(ρ)
        return Transformer.from_lin_alg(n_conj, Σ, M_conj, λ_conj, ρ_conj)

    @staticmethod
    def backward_conjugate(fsa):
        """Computes a backward minimal FSA equivalent to the input."""
        return Transformer.forward_conjugate(fsa.reverse()).reverse()

    @staticmethod
    def minimize_nfa(fsa):
        """Computes a minimal FSA equivalent to the input."""
        assert fsa.R == Real
        forward_conjugate = Transformer.forward_conjugate(fsa)
        return Transformer.backward_conjugate(forward_conjugate)

    @staticmethod
    def renormalize_decoupled_fst(T: FST) -> FST:
        """
        WARNING: THIS DOES NOT WORK!!!
        This was an attempt at the following:
        Given a decoupled FST, replace the weights such that, when composed with
        `string_fsa(x)`, the resulting FST represents a locally normalized FST.

        Args:
            T (FST): The decoupled FST.

        Returns:
            FST: The renormalized FST.
        """
        # TODO: (Anej) Check if the FST is decoupled.
        Tʼ = T.spawn()

        for q in T.Q:
            for (a, b), j, w in T.arcs(q):
                if q.label == "sep":
                    Tʼ.add_arc(q, a, b, j, w)
                else:
                    Tʼ.add_arc(q, a, b, j, T.R.one)

        for q, w in T.I:
            # Tʼ.add_I(q, w)
            Tʼ.add_I(q, T.R.one)
        for q, w in T.F:
            # Tʼ.add_F(q, w)
            Tʼ.add_F(q, T.R.one)

        return Tʼ

    def uniform(A: FSA) -> FSA:
        """Converts the FSA into a FSA whose topology is the same but the weights are
        uniform over the next transitions.

        Returns:
            FSA: The uniform FSA.
        """

        B = A.spawn()

        for q in A.Q:
            out_arcs = list(A.arcs(q))
            if A.ρ[q] != A.R.zero:
                Z = A.R(len(out_arcs) + 1)
                B.add_F(q, A.R.one / Z)
            else:
                Z = A.R(len(out_arcs))
            for a, j, _ in out_arcs:
                B.add_arc(q, a, j, A.R.one / Z)

        initial_states = [q for q, _ in A.I]
        for q, w in A.I:
            B.add_I(q, A.R.one / A.R(len(initial_states)))

        return B
