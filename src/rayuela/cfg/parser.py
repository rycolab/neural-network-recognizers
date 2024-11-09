from collections import defaultdict as dd
from itertools import product
from typing import List, Type

import numpy as np

from rayuela.base.datastructures import PriorityQueue
from rayuela.base.misc import straight
from rayuela.base.semiring import Real, Semiring, Tropical, product_semiring_builder
from rayuela.base.symbol import Concatenation, Expr, Sym, Union
from rayuela.cfg.cfg import CFG
from rayuela.cfg.nonterminal import NT, S
from rayuela.cfg.production import Production
from rayuela.cfg.transformer import Transformer
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.base.state import State


class Parser:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.R = self.cfg.R

    def sum(self, input, strategy="cky"):
        if strategy == "cky":
            return self.cky(input)
        elif strategy == "agenda":
            return self.agenda(input)
        elif strategy == "earley":
            return self.earley(input)
        else:
            raise NotImplementedError

    def _prune(self, input):
        """semiring version of CKY"""
        N = len(input)

        # handle unaries outside of main loops
        W = Pathsum(self.cfg.unary_fsa).lehmann(zero=False)
        chain = (
            lambda X, Y: W[State(X), State(Y)]
            if (State(X), State(Y)) in W
            else self.R.zero
        )

        active = set([])

        # initialization
        β = self.R.chart()

        # pre-terminals
        for p, w in self.cfg.terminal:
            (head, body) = p
            for i in range(N):
                if body[0] == input[i]:
                    β[head, i, i + 1] += w
                    if w != self.R.zero:
                        active.add(p)

        # three for loops
        for span in range(1, N + 2):
            for i in range(N - span + 2):
                k = i + span
                for j in range(i + 1, k):
                    for p, w in self.cfg.binary:
                        X, Y, Z = p.head, p.body[0], p.body[1]
                        β[X, i, k] += β[Y, i, j] * β[Z, j, k] * w
                        if β[Y, i, j] * β[Z, j, k] * w != self.R.zero:
                            active.add(p)

                # include unary chains (not part of standard CKY)
                U = []
                for X in self.cfg.V:
                    for Y in self.cfg.V:
                        U.append(((Y, i, k), β[(X, i, k)] * chain(X, Y)))
                for item, w in U:
                    β[item] += w

        return active

    def _cky(self, input, unary=False):
        """semiring version of CKY"""
        N = len(input)

        # handle unaries outside of main loops

        if unary:
            self.cfg.make_unary_fsa()
            W = Pathsum(self.cfg.unary_fsa).lehmann(zero=False)
            chain = (
                lambda X, Y: W[State(X), State(Y)]
                if (State(X), State(Y)) in W
                else self.R.zero
            )

        # initialization
        β = self.R.chart()

        # pre-terminals
        for (head, body), w in self.cfg.terminal:
            for i in range(N):
                if body[0] == input[i]:
                    β[head, i, i + 1] += w

        # three for loops
        for span in range(1, N + 2):
            for i in range(N - span + 1):
                k = i + span
                for j in range(i + 1, k):
                    for p, w in self.cfg.binary:
                        X, Y, Z = p.head, p.body[0], p.body[1]
                        β[X, i, k] += β[Y, i, j] * β[Z, j, k] * w

                if unary:
                    # include unary chains (not part of standard CKY)
                    U = []
                    for X in self.cfg.V:
                        for Y in self.cfg.V:
                            U.append(((Y, i, k), β[(X, i, k)] * chain(X, Y)))
                    for item, w in U:
                        β[item] += w

        return β

    def cky(self, input, unary=True):
        return self._cky(input, unary=unary)[(self.cfg.S, 0, len(input))]

    def _agenda(self, input):
        N = len(input)

        # handle unaries outside of main loops
        W = Pathsum(self.cfg.unary_fsa).lehmann(zero=False)
        chain = lambda X, Y: W[State(X), State(Y)]

        # initialization
        β = self.R.chart()
        agenda = PriorityQueue(R=self.R)
        popped = set([])  # we need popped because of defaultdict

        # base case
        for (head, body), w in self.cfg.terminal:
            for i in range(N):
                if body[0] == input[i]:
                    agenda.push((head, i, i + 1), w)

        #  main loop
        while agenda:
            item, w = agenda.pop()
            popped.add(item)
            β[item] += w

            # unary chains
            (X, i, k) = item
            for Y in self.cfg.V:
                if (Y, i, k) not in popped:
                    agenda.push((Y, i, k), β[(X, i, k)] * chain(X, Y))

            # attach right
            (Y, i, j) = item
            for k in range(j + 1, N + 1):
                for p, w in self.cfg.binary:
                    X, Y_, Z = p.head, p.body[0], p.body[1]
                    v = β[Y, i, j] * β[Z, j, k] * w
                    if Y == Y_ and (Z, j, k) in β:
                        agenda.push((X, i, k), v)

            # attach left
            (Z, j, k) = item
            for i in range(0, j):
                for p, w in self.cfg.binary:
                    X, Y, Z_ = p.head, p.body[0], p.body[1]
                    v = β[Y, i, j] * β[Z, j, k] * w
                    if Z == Z_ and (Y, i, j) in β:
                        agenda.push((X, i, k), v)

        return β

    def agenda(self, input):
        return self._agenda(input)[(self.cfg.S, 0, len(input))]

    def bar_hillel(self, fsa, unary=True):
        """intersects the grammar with an FSA and parses"""

        # handle unaries outside of main loops
        W = Pathsum(self.cfg.unary_fsa).allpairs(strategy=Strategy.LEHMANN, zero=False)
        chain = (
            lambda X, Y: W[State(X), State(Y)]
            if (State(X), State(Y)) in W
            else self.R.zero
        )

        # initialization
        N = fsa.num_states
        β = dd(lambda: self.R.zero)

        # memoize topological order
        O = {}
        for n, q in enumerate(fsa.toposort()):
            O[q] = n

        # pre-terminals
        for p in fsa.toposort():
            for a, q, v in fsa.arcs(p):
                for (head, body), w in self.cfg.terminal:
                    if body[0] == a:
                        β[head, O[p], O[q]] += v * w

        # the three main for loops
        for span in range(1, N):
            for i in range(N - span):
                k = i + span
                for j in range(i + 1, k):
                    for (head, body), w in self.cfg.binary:
                        X, Y, Z = head, body[0], body[1]
                        β[X, i, k] += β[Y, i, j] * β[Z, j, k] * w

                if unary:
                    # include unary chains (not part of standard CKY)
                    U = []
                    for X in self.cfg.V:
                        for Y in self.cfg.V:
                            U.append(((Y, i, k), β[(X, i, k)] * chain(X, Y)))

                    for item, w in U:
                        β[item] += w

        res = self.R.zero
        for p, v in fsa.I:
            for q, w in fsa.F:
                res += v * β[self.cfg.S, O[p], O[q]] * w

        return res

    def brzozowski(self, input):
        T = Transformer()
        fsa = straight(input, self.R)
        ncfg = T.left_quotient(self.cfg, fsa)

        ecfg, _ = ncfg.eps_partition()

        return ecfg.treesum()

    def plc(self):
        """Computes the left-corner probabilities of the grammar.
        Assumes CNF and Real semiring."""
        assert self.R == Real
        assert self.cfg.in_cnf()

        V = self.cfg.ordered_V
        V_idx = {X: i for i, X in enumerate(V)}
        P = np.zeros((len(V), len(V)))
        for p, w in self.cfg.binary:
            X, Y = V_idx[p.head], V_idx[p.body[0]]
            P[X, Y] += w.value
        return np.linalg.inv(np.eye(len(V), len(V)) - P)

    def lri(self, input, strategy=None):
        """Computes all prefix probabilities of a string under the grammar.
        Assumes CNF and Real semiring."""
        assert self.cfg.in_cnf()
        if strategy is None:
            return self._lri(input)
        elif strategy == "fast":
            return self._lri_fast(input)
        else:
            raise NotImplementedError

    def _lri(self, input):
        """Original LRI formulation as introduced by Jelinek and Lafferty (1991) -
        https://aclanthology.org/J91-3004/"""
        N = len(input)
        V = self.cfg.ordered_V
        V_idx = {X: i for i, X in enumerate(V)}
        ppre = self.R.chart()
        # precompute β using CKY
        β = self._cky(input, unary=False)
        # precompute E
        E = self.R.chart()
        P_L = self.plc()
        for X in self.cfg.V:
            for Y in self.cfg.V:
                E[X, Y] = Real(P_L[V_idx[X], V_idx[Y]])
        # precompute E2
        E2 = self.R.chart()
        for X in self.cfg.V:
            for p, w in self.cfg.binary:
                Y2, Y, Z = p.head, p.body[0], p.body[1]
                E2[X, Y, Z] += E[X, Y2] * self.cfg._P[Production(Y2, (Y, Z))]
        # compute base case
        for X in self.cfg.V:
            for i in range(N):
                for p, w in self.cfg.terminal:
                    Y, v = p.head, p.body[0]
                    if v == input[i]:
                        ppre[X, i, i + 1] += E[X, Y] * w
        # compute prefix probability
        for l in range(1, N + 2):
            for i in range(N - l + 1):
                k = i + l
                for j in range(i + 1, k):
                    for X in self.cfg.V:
                        for Y in self.cfg.V:
                            for Z in self.cfg.V:
                                ppre[X, i, k] += (
                                    E2[X, Y, Z] * β[Y, i, j] * ppre[Z, j, k]
                                )
        return ppre

    def _lri_fast(self, input):
        """Faster prefix parsing algorithm as proposed by Nowak and Cotterell (2023)"""
        N = len(input)
        V = self.cfg.ordered_V
        V_idx = {X: i for i, X in enumerate(V)}
        ppre = self.R.chart()
        # precompute β using CKY
        β = self._cky(input, unary=False)
        # precompute E
        E = self.R.chart()
        P_L = self.plc()
        for X in self.cfg.V:
            for Y in self.cfg.V:
                E[X, Y] = Real(P_L[V_idx[X], V_idx[Y]])
        # precompute γ and δ
        γ = self.R.chart()
        δ = self.R.chart()
        for i in range(N):
            for j in range(N):
                for p, w in self.cfg.binary:
                    X, Y, Z = p.head, p.body[0], p.body[1]
                    γ[i, j, X, Z] += w * β[Y, i, j]
                for X in self.cfg.V:
                    for Y in self.cfg.V:
                        for Z in self.cfg.V:
                            δ[i, j, X, Z] += E[X, Y] * γ[i, j, Y, Z]
        # compute base case
        for X in self.cfg.V:
            for i in range(N):
                for p, w in self.cfg.terminal:
                    Y, v = p.head, p.body[0]
                    if v == input[i]:
                        ppre[X, i, i + 1] += E[X, Y] * w
        # compute prefix probability
        for l in range(1, N + 2):
            for i in range(N - l + 1):
                k = i + l
                for j in range(i + 1, k):
                    for X in self.cfg.V:
                        for Z in self.cfg.V:
                            ppre[X, i, k] += δ[i, j, X, Z] * ppre[Z, j, k]
        return ppre


class EarleyItem:
    def __init__(self, i, k, head, body=(), dot=0, star=False, dotopt=False):
        self.i, self.k = i, k
        self.head, self.body = head, body
        assert self.i <= self.k, "inadmissible span"
        self.dot = dot
        self.star = star

    @property
    def end(self):
        return self.dot >= len(self.body)

    @property
    def next(self):
        if not self.end:
            return self.body[self.dot]
        return None

    def scan(self, a):
        assert self.body[self.dot] == a, "heads incompatible"
        # return EarleyItem(
        #     self.i, self.k + 1, self.head, self.body[self.dot :], dot=self.dot + 1
        # )
        return EarleyItem(self.i, self.k + 1, self.head, self.body, dot=self.dot + 1)

    def complete(self, item):
        assert self.k == item.i, "indices incompatible"
        assert self.body[self.dot] == item.head, "heads incompatible"
        # return EarleyItem(
        #     self.i, item.k, self.head, self.body[self.dot :], dot=self.dot + 1
        # )
        return EarleyItem(self.i, item.k, self.head, self.body, dot=self.dot + 1)

    def bwd(self):
        if self.dot < len(self.body):
            return (self.body[self.dot], self.k)
        return None

    def sig(self):
        return (self.head, self.i)

    def right(self):
        if self.dot < len(self.body):
            return (self.body[self.dot], self.k)
        return None

    def left(self):
        return (self.head, self.i)

    def __str__(self):
        body = []
        for n, X in enumerate(self.body):
            if n == self.dot:
                body.append("•")
            body.append(str(X))
        if self.dot == len(self.body):
            body.append("•")
        body = " ".join(body)

        return f"[{self.i}, {self.k}, {self.head} → {body}]"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            isinstance(other, EarleyItem)
            and self.i == other.i
            and self.k == other.k
            and self.head == other.head
            and self.body == other.body
            and self.dot == other.dot
        )

    def __hash__(self):
        return hash((self.i, self.k, self.dot, self.head, self.body))


class EarleyParser(Parser):
    def __init__(self, cfg):
        super().__init__(cfg)

    def parse_n2(self, input):
        β = self.R.chart()
        N = len(input)

        for k in range(N):
            # predict

            for i in range(k):
                for p, v in self.cfg.P_byhead(item.next):
                    xseitem = EarleyItem(k, k, p.head, p.body)

                # scan

                # complete
                for j in range(i + 1, k):
                    for l, r in product(β[i, j], β[j, k]):
                        β[i, k, l.complete(r)] += β[i, j] * β[j, k]

        return β

    def _agenda(self, input):
        N = len(input)
        zero, one = self.R.zero, self.R.one

        # initialization
        β = self.R.chart()
        agenda = PriorityQueue(R=Tropical)

        # add initial items
        for p, w in self.cfg.P_byhead(S):
            eitem = EarleyItem(0, 0, p.head, p.body)
            agenda.push(eitem, w)
            β[eitem] += w

        #  main loop
        visited = set([])
        gss_right, gss_left = dd(set), dd(set)

        while agenda:
            item, w = agenda.pop()
            i, k = item.i, item.k
            visited.add(item)

            gss_right[item.right()].add(item)
            gss_left[item.left()].add(item)

            # scan
            if k < N:
                eitem = EarleyItem(k, k + 1, input[k], ())
                if eitem not in visited:
                    agenda.push(eitem, one)
                    β[eitem] = one
                    visited.add(eitem)

            # predict
            if not item.end and isinstance(item.next, NT):
                for p, v in self.cfg.P_byhead(item.next):
                    eitem = EarleyItem(k, k, p.head, p.body)
                    β[eitem] = v

                    if eitem not in visited:
                        agenda.push(eitem, v)

            # complete left
            for eitem in gss_left[item.right()]:
                if eitem.end:
                    eitemʼ = item.complete(eitem)
                    β[eitemʼ] += β[item] * β[eitem]

                    if eitemʼ not in visited:
                        agenda.push(eitemʼ, β[eitemʼ])

            # complete right
            if item.end:
                for eitem in gss_right[item.left()]:
                    eitemʼ = eitem.complete(item)
                    β[eitemʼ] += β[eitem] * β[item]

                    if eitemʼ not in visited:
                        agenda.push(eitemʼ, β[eitemʼ])

            # unary rules
            # TODO: add unary rules

        return β

    def _earley(self, input):
        zero, one = self.R.zero, self.R.one

        # initialization
        N = len(input)
        β = self.R.chart()

        # unary chains
        W = Pathsum(self.cfg.unary_fsa).lehmann(zero=True)
        chain = (
            lambda X, Y: W[State(X), State(Y)]
            if (State(X), State(Y)) in W
            else one
            if X == Y
            else zero
        )
        agendas = []
        for n in range(N + 1):
            agendas.append(PriorityQueue(R=Tropical))

        # add initial items
        # TODO: can cut off loop early (done for unit-testing)
        for n in range(N + 1):
            for p, w in self.cfg.P_byhead(S, unary=True):
                for Y in self.cfg.V:
                    if chain(p.head, Y) != zero:
                        eitem = EarleyItem(n, n, Y, p.body)
                        β[eitem] += w * chain(p.head, Y)
                        agendas[n].push(eitem, n)

        #  main loop
        visited = set([])
        gss = dd(set)
        U = self.R.chart()

        predicted = set([])

        for n in range(N + 1):
            # scan
            if n < N:
                eitem = EarleyItem(n, n + 1, input[n], ())
                agendas[n + 1].push(eitem, 0)
                β[eitem] = one

            while agendas[n]:
                item, _ = agendas[n].pop()
                i, k = item.i, item.k
                gss[item.bwd()].add(item)

                assert n == k
                visited.add(item)

                # predict
                if not item.end and isinstance(item.next, NT):
                    for Y in self.cfg.V:
                        for p, w in self.cfg.P_byhead(Y, unary=False):
                            eitem = EarleyItem(k, k, item.next, p.body)

                            if (Y, eitem) not in predicted:
                                β[eitem] += w * chain(Y, item.next)
                                predicted.add((Y, eitem))
                                if eitem not in visited:
                                    agendas[k].push(eitem, -eitem.i)

                # compete
                elif item.end:
                    for eitem in gss[item.sig()]:
                        eitemʼ = eitem.complete(item)
                        β[eitemʼ] += β[eitem] * β[item]

                        if eitemʼ not in visited:
                            agendas[eitemʼ.k].push(eitemʼ, -eitemʼ.i)

        return β

    def _earley_fast(self, input):
        zero, one = self.R.zero, self.R.one

        # initialization
        N = len(input)
        β = self.R.chart()

        # unary chains
        W = Pathsum(self.cfg.unary_fsa).lehmann(zero=True)
        chain = (
            lambda X, Y: W[State(X), State(Y)]
            if (State(X), State(Y)) in W
            else one
            if X == Y
            else zero
        )
        agendas = []
        for n in range(N + 1):
            agendas.append(PriorityQueue(R=Tropical))

        # add initial items
        # TODO: can cut off loop early (done for unit-testing)
        for n in range(N + 1):
            for p, w in self.cfg.P_byhead(S, unary=True):
                for Y in self.cfg.V:
                    if chain(p.head, Y) != zero:
                        eitem = EarleyItem(n, n, Y, p.body)
                        β[eitem] += w * chain(p.head, Y)
                        agendas[n].push(eitem, n)

        #  main loop
        visited = set([])
        gss = dd(set)
        U = self.R.chart()

        predicted = set([])

        for n in range(N + 1):
            # scan
            if n < N:
                eitem = EarleyItem(n, n + 1, input[n], ())
                agendas[n + 1].push(eitem, 0)
                β[eitem] = one

            while agendas[n]:
                item, _ = agendas[n].pop()
                i, k = item.i, item.k
                gss[item.bwd()].add(item)

                assert n == k
                visited.add(item)

                # predict 1
                if not item.end and isinstance(item.next, NT):
                    for Y in self.cfg.V:
                        for p, w in self.cfg.P_byhead(Y, unary=False):
                            eitem = EarleyItem(k, k, item.next, p.body)

                            if (Y, eitem) not in predicted:
                                β[eitem] += w * chain(Y, item.next)
                                predicted.add((Y, eitem))
                                if eitem not in visited:
                                    agendas[k].push(eitem, -eitem.i)
                    # predict 2

                # complete 1
                elif item.end:
                    for eitem in gss[item.sig()]:
                        eitemʼ = eitem.complete(item)
                        β[eitemʼ] += β[eitem] * β[item]

                        if eitemʼ not in visited:
                            agendas[eitemʼ.k].push(eitemʼ, -eitemʼ.i)

                # complete 2

        return β

    def earley_chart(self, input, strategy="earley"):
        β = None
        if strategy == "earley":
            β = self._earley(input)
        elif strategy == "agenda":
            β = self._agenda(input)
        else:
            raise NotImplementedError

        chart = self.R.chart()
        for item, w in β.items():
            if item.end:
                chart[item.head, item.i, item.k] += w

        return chart

    def earley(self, input, strategy="earley"):
        β = None
        if strategy == "earley":
            β = self._earley(input)
        elif strategy == "earley_fast":
            β = self._earley_fast(input)
        elif strategy == "agenda":
            β = self._agenda(input)
        else:
            raise NotImplementedError

        total = self.cfg.R.zero
        for item, w in β.items():
            if item.end and item.head == S and item.i == 0 and item.k == len(input):
                total += β[item]

        return total


def compute_weights_of_all_parses(T: Expr, R: Type[Semiring]) -> List[Semiring]:
    """Computes the weights of all individual parses captured in the `parses` Expr
    object, which stores the forest of parses.

    Args:
        T (Expr): The forest of parses in which the Expr values are tuples of
            from the (Expr, R) product semiring, where R is the semiring weighting
            the parses.

    Returns:
        List[Semiring]: The weights of all individual parses.
    """

    def _compute(T: Expr):
        if isinstance(T, Sym):
            weights = [R(T.value)]
        elif isinstance(T, Concatenation):
            Tx_ = _compute(T.x)
            Ty_ = _compute(T.y)

            weights = [tx * ty for (tx, ty) in product(Tx_, Ty_)]
        elif isinstance(T, Union):
            weights = _compute(T.x) + _compute(T.y)

        return weights

    weights = _compute(T)
    return weights
