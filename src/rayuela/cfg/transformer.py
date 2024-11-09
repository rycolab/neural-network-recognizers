import itertools as it
import time
from itertools import chain, product
from math import prod

from rayuela.base.misc import symify
from rayuela.base.semiring import Boolean, Integer, product_semiring_builder
from rayuela.base.symbol import Expr, Sym, ε
from rayuela.cfg.cfg import CFG
from rayuela.cfg.misc import binarized, powerset, preterminal, unary
from rayuela.cfg.nonterminal import NT, Delta, Other, S, Slash, Triplet
from rayuela.cfg.production import Production
from rayuela.cfg.treesum import Treesum
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.base.state import State


class Transformer:
    def __init__(self):
        self.counter = 0

    def _gen_nt(self):
        self.counter += 1
        return NT(f"@{self.counter}")

    def booleanize(self, cfg):
        one = Boolean(True)
        ncfg = CFG(R=Boolean)
        ncfg.S = cfg.S
        for p, w in cfg.P:
            if w != cfg.R.zero:
                ncfg.add(one, p.head, *p.body)
        return ncfg

    def unaryremove(self, cfg):
        # compute unary chain weights
        W = Pathsum(cfg.unary_fsa).lehmann(zero=True)

        def chain(X, Y):
            if (State(X), State(Y)) in W:
                return W[State(X), State(Y)]
            elif X == Y:
                return cfg.R.one
            else:
                return cfg.R.zero

        ncfg = CFG(R=cfg.R)
        for p, w in cfg.P:
            X, body = p
            if not unary(p):
                for Y in cfg.V:
                    ncfg.add(chain(X, Y) * w, Y, *body)

        ncfg.make_unary_fsa()
        return ncfg

    def nullaryremove(self, cfg):
        zero, one = cfg.R.zero, cfg.R.one
        ecfg, _ = cfg.eps_partition()
        U = Treesum(ecfg).table()

        rcfg = cfg.spawn()
        rcfg.add(U[cfg.S], S, ε)

        for p, w in cfg.P:
            head, body = p
            if len(body) == 1 and body[0] == ε:
                continue

            for B in it.product([0, 1], repeat=len(body)):
                v, lst = w, []
                for i, b in enumerate(B):
                    if b == 1:
                        v *= U[body[i]]
                    elif body[i] != ε:
                        lst.append(body[i])

                # excludes the all zero case
                if len(lst) > 0:
                    rcfg.add(v, head, *lst)

        rcfg.make_unary_fsa()
        return rcfg

    def brzozowski_derivative(self, cfg, a, idx=0):
        """non-pruned derivative grammar"""

        one = cfg.R.one
        ncfg = cfg.spawn()

        ecfg, _ = cfg.eps_partition()
        U = Treesum(ecfg).table()

        for p, w in cfg.P:
            ncfg.add(w, p.head, *p.body)

        for p, w in cfg.P:
            if len(p.body) == 0:
                pass

            delta = one
            for k, X in enumerate(p.body):
                if X in cfg.V:
                    ncfg.add(
                        delta * w,
                        Delta(p.head, a, idx),
                        Delta(p.body[k], a, idx),
                        *p.body[(k + 1) :],
                    )
                elif X == a:
                    ncfg.add(delta * w, Delta(p.head, a, idx), *p.body[(k + 1) :])

                delta *= U[X]

        ncfg.make_unary_fsa()
        return ncfg

    def derivative(self, cfg, s, pruned=True, normal_form=False):
        """function that computes the derivative of a grammar (Heriksen et al., 2019)
        if pruned =True, then it uses the pruned implementaion to compute the grammar"""

        string = symify(s)
        cfgj_1 = cfg
        S = cfg.S

        global time_8
        global time_9
        global time_10

        # queue=list(reversed(string))
        for a in string:
            time_8 = time.time()
            cfgj_1 = self.pruned_derivative(cfg, cfgj_1, a, normal_form=normal_form)
            time_9 = time.time()
            S = Delta(S, a, 0)

        return cfgj_1, S

    def pruned_derivative(self, cfg, cfgj_1, a, idx=0, normal_form=False):
        # TODO: define a derivative grammar object

        """Implements the pruned weighted derivative.
        Dictionary of the variable:
        Inputs:
        cfg     ---> original grammar G
        cfgj_1  ---> j-1 th derivative grammar
        a       ---> j-th character of the input string s
        Outputs:
        cfgj    ---> j-th derivative grammar

        Where G_{j}=< N U N_{j}, P U P_{j} , S/s_1...s_j, Σ >
        """

        one = cfg.R.one
        cfgj = cfg.spawn()
        ecfgj_1, _ = cfgj_1.eps_partition()

        # If we are using the speed up version, the entries of the nullability chart are FSA States
        _ta = lambda X: State(X) if normal_form else X

        if normal_form:
            # print("check")
            U = ecfgj_1.make_chain_fsa().backward(strategy=Strategy.VITERBI)
        else:
            U = Treesum(
                ecfgj_1
            ).table()  # we can parse no matter the form of the grammar

        for p, w in cfg.P:  # Add to grammar Gj the original production rules
            cfgj.add(w, p.head, *p.body)

        for p, w in cfgj_1.P:
            if len(p.body) == 0:
                pass
            delta = one

            head = Delta(p.head, a, idx)

            for k, X in enumerate(p.body):
                if isinstance(X, NT):  # Add nonterminal rules
                    tail = Delta(p.body[k], a, idx)
                    cfgj.add(
                        delta * w,
                        head,
                        tail,
                        *p.body[(k + 1) :],
                    )

                elif X == a:  # Add terminal rules
                    cfgj.add(delta * w, head, *p.body[(k + 1) :])
                delta *= U[_ta(X)]

        return cfgj

    def left_quotient(self, cfg, FSA):
        assert cfg.R == FSA.R
        R = cfg.R
        fsa = FSA.copy()
        pfg = cfg.spawn()
        qstar = State("q*")
        QQ = fsa.Q | {qstar}
        S = cfg.S
        SS = Other(cfg.S)

        # (1a)
        for qi, wi in fsa.I:
            pfg.add(~wi, S, Triplet(qi, SS, qstar))

        # (1b)
        for q0, q1, q2 in product(QQ, repeat=3):
            pfg.add(
                cfg.R.one, Triplet(q0, SS, q2), Triplet(q0, SS, q1), Triplet(q1, ε, q2)
            )

        # (1c)
        for q0, q1 in product(QQ, repeat=2):
            pfg.add(cfg.R.one, Triplet(q0, SS, q1), Triplet(q0, S, q1))

        # (1d) -- non nullary production rules
        for (X, Ys), w in cfg.P:
            M = len(Ys)
            if M >= 1 and Ys[0] != ε:
                for qs in product(QQ, repeat=M + 1):
                    pfg.add(
                        w,
                        Triplet(qs[0], X, qs[M]),
                        *[Triplet(qs[i], Ys[i], qs[i + 1]) for i in range(M)],
                    )

        # (1e) -- nullary productions
        for (X, Ys), w in cfg.P:
            M = len(Ys)
            if M == 1 and Ys[0] == ε:
                for q0 in QQ:
                    pfg.add(w, Triplet(q0, X, q0), ε)

        # (1f)
        for q0 in fsa.Q:
            for a, q1, w in fsa.arcs(q0):
                pfg.add(~w, Triplet(q0, a, q1), ε)

        # (1g)
        for q0, q1, q2 in product(QQ, repeat=3):
            for a in cfg.Sigma:
                pfg.add(
                    cfg.R.one,
                    Triplet(q0, a, q2),
                    Triplet(q0, ε, q1),
                    Triplet(q1, a, q2),
                )

        # (1h)
        for qf, wf in fsa.F:
            pfg.add(~wf, Triplet(qf, ε, qstar), ε)

        # (1i)
        for a in cfg.Sigma:
            pfg.add(fsa.R.one, Triplet(qstar, a, qstar), a)

        return pfg

    def separate_terminals(self, cfg) -> CFG:
        ncfg = cfg.spawn()
        for p, w in cfg.P:
            I = self._terminals(cfg, p)
            if len(I) == 0:
                ncfg.add(w, p.head, *p.body)
            else:
                for (head, body), v in self._fold(cfg, p, w, I):
                    ncfg.add(v, head, *body)

        ncfg.make_unary_fsa()
        return ncfg

    def _terminals(self, cfg, p):
        I = []
        for i, x in enumerate(p.body):
            if x in cfg.Sigma:
                I.append((i, i))
        return I

    def binarize(self, cfg):
        ncfg = cfg.spawn()

        stack = []
        for p, w in cfg.P:
            stack.append((p, w))

        while stack:
            (p, w) = stack.pop()
            (head, body) = p
            if preterminal(p):
                ncfg.add(w, head, *body)
            elif binarized(p):
                ncfg.add(w, head, *body)
            else:
                for (head, body), v in self._fold(cfg, p, w, [(0, 1)]):
                    stack.append((Production(head, body), v))

        ncfg.make_unary_fsa()
        return ncfg

    def _fold(self, cfg, p, w, I):
        # basic sanity checks
        for i, j in I:
            assert i >= 0 and j >= i and j < len(p.body)

        # new productions
        P, heads = [], []
        for i, j in I:
            head = self._gen_nt()
            heads.append(head)
            body = p.body[i : j + 1]
            P.append(((head, body), cfg.R.one))

        # new "head" production
        body = tuple()
        start = 0
        for (end, n), head in zip(I, heads):
            body += p.body[start:end] + (head,)
            start = n + 1
        body += p.body[start:]
        P.append(((p.head, body), w))

        return P

    def fold(self, cfg, p, w, I):
        ncfg = cfg.spawn()
        add = ncfg.add

        for q, v in cfg.P:
            if p != q:
                add(v, q.head, *q.body)

        for (
            (head, body),
            v,
        ) in self._fold(cfg, p, w, I):
            add(v, head, *body)

        ncfg.make_unary_fsa()
        return ncfg

    def cnf(self, cfg):
        # remove terminals
        ncfg = self.separate_terminals(cfg)

        # remove nullary rules
        ncfg = self.nullaryremove(ncfg)

        # remove unary rules
        ncfg = self.unaryremove(ncfg)

        # binarize
        ncfg = self.binarize(ncfg)

        return ncfg.trim()

    def elim(self, cfg, p):
        ncfg = cfg.spawn()
        u = cfg._P[p]
        for pʼ, w in cfg.P:
            if p == pʼ:
                continue

            # TODO: find cleaner way
            matches = []
            for i, elem in enumerate(pʼ.body):
                if p.head == elem:
                    matches.append(i)

            ps = set(powerset(matches))
            if len(ps) > 1:
                for s in ps:
                    s = set(s)
                    if len(s) == 0:
                        continue
                    nbody = []
                    for i, elem in enumerate(pʼ.body):
                        if i in s:
                            nbody.extend(list(p.body))
                        else:
                            nbody.append(elem)

                    ncfg.add(w * u, pʼ.head, *nbody)

            ncfg.add(w, pʼ.head, *pʼ.body)

        ncfg.make_unary_fsa()
        return ncfg

    def unfold2(self, cfg, p, i):
        ncfg = cfg.spawn()
        for pʼ, w in cfg.P:
            if pʼ == p:
                continue
            ncfg.add(w, pʼ.head, *pʼ.body)

        new = []
        for pʼ, w in cfg.P:
            if len(p.body) > 0 and pʼ.head == p.body[i]:
                nbody = list(p.body[:i] + pʼ.body + p.body[i + 1 :])
                if ε in nbody:
                    nbody.remove(ε)
                new.append((w * cfg._P[p], p.head, nbody))

        for w, head, body in new:
            ncfg.add(w, head, *body)

        return ncfg

    def speculate(self, cfg, Xs, Ys, sigma):
        R = cfg.R
        zero, one = R.zero, R.one

        scfg = CFG(R=R)
        add = scfg.add

        # base case
        for X in Xs:
            add(one, X / X, ε)

        # make slashed and other rules
        for p, w in cfg.P:
            (head, body) = p
            if p not in Ps:  # or head not in Ys:
                add(w, ~head, *body)
            else:
                for X in Xs:
                    add(w, head / X, body[0] / X, *body[1:])

                if body[0] not in Xs:
                    add(w, ~head, ~body[0], *body[1:])

        # recovery rules
        for Y in cfg.V:
            if Y not in Xs:
                add(one, Y, ~Y)

            for X in Xs:
                add(one, Y, ~X, Y / X)
            for a in cfg.Sigma:
                add(one, Y, a, Y / a)

        scfg.make_unary_fsa()
        return scfg

    def speculate_equiv(self, cfg, Xs, Ps):
        assert ε not in cfg.Sigma

        def slash(X, Y):
            # TODO: needs to be unique
            return Slash(X, Y)

        def other(X):
            if isinstance(X, Sym):
                return X
            # TODO: needs to be unique
            return Other(X)

        R = cfg.R
        zero, one = R.zero, R.one

        scfg = CFG(R=R)
        add = scfg.add

        # base case
        for X in Xs:
            add(one, slash(X, X), ε)

        # make slashed and other rules
        for p, w in cfg.P:
            (head, body) = p

            if p not in Ps:
                add(w, other(head), *body)
            else:
                for X in Xs:
                    add(w, slash(head, X), slash(body[0], X), *body[1:])

                if body[0] not in Xs:
                    add(w, other(head), other(body[0]), *body[1:])

        # recovery rules
        for Y in cfg.V:
            if Y not in Xs:
                add(one, Y, other(Y))

        for Y in cfg.V:
            for X in Xs:
                add(one, Y, other(X), slash(Y, X))

        scfg.make_unary_fsa()
        return scfg

    def lc_equiv(self, cfg, Xs, Ps):
        assert ε not in cfg.Sigma

        def slash(X, Y):
            # TODO: needs to be unique
            return Slash(X, Y)

        def other(X):
            if isinstance(X, Sym):
                return X
            # TODO: needs to be unique
            return Other(X)

        R = cfg.R
        zero, one = R.zero, R.one

        lcfg = CFG(R=R)
        add = lcfg.add

        # base case
        for X in cfg.V:
            add(one, slash(X, X), ε)

        # make slashed and other rules
        for p, w in cfg.P:
            (head, body) = p

            if p not in Ps:
                add(w, other(head), *body)
            else:
                for Y in cfg.V:
                    add(w, slash(Y, body[0]), *body[1:], slash(Y, head))

                if body[0] not in Xs:
                    add(w, other(head), other(body[0]), *body[1:])

        # recovery rules
        for Y in cfg.V:
            if Y not in Xs:
                add(one, Y, other(Y))

        for Y in cfg.V:
            for X in Xs:
                add(one, Y, other(X), slash(Y, X))

        lcfg.make_unary_fsa()
        return lcfg

    def lc_selective_johnson(self, cfg, Ps):
        """
        Adapted from Johnson and Roark (2000)
        """
        R = cfg.R
        zero, one = R.zero, R.one

        lcfg = CFG(R=R)
        add = lcfg.add

        # base case
        for Y in cfg.V:
            add(one, Y / Y, ε)

        # left-corner rules
        for p, w in cfg.P:
            (head, body) = p
            if p not in Ps:
                for X in cfg.V:
                    add(w, X, *body, X / head)
            else:
                for X in cfg.V:
                    add(w, X / body[0], *body[1:], X / head)

        # recovery rules
        for X in cfg.V:
            for a in cfg.Sigma:
                add(one, X, a, X / a)

        lcfg.make_unary_fsa()
        return lcfg

    def reverse(self, cfg):
        rcfg = cfg.spawn()
        for p, w in cfg.P:
            nbody = list(reversed(p.body))
            rcfg.add(w, p.head, *nbody)
        return rcfg

    def greibach(self, cfg):
        # convert reverse grammar to cnf
        cfg = self.cnf(self.reverse(cfg))

        # perform left-corner transform
        sigma, Ys = {}, cfg.V.copy()
        for p, w in cfg.P:
            if isinstance(p.body[0], NT):
                sigma[p] = 0
        lcfg = self.lc(cfg, Ys, sigma)

        # eliminate others
        for p, w in list(lcfg.P):
            if isinstance(p.head, Other):
                lcfg = self.elim(lcfg, p).trim()

        # eliminate non-slashed
        for p, w in list(lcfg.P):
            if (
                p.head != S
                and not isinstance(p.head, Slash)
                and not isinstance(p.head, Other)
            ):
                lcfg = self.elim(lcfg, p).trim()

        # clean-up
        lcfg = self.nullaryremove(lcfg).trim()

        # un-reverse
        gcfg = self.reverse(lcfg)

        return gcfg

    def greibach2(self, cfg, lc):
        # convert reverse grammar to cnf
        cfg = self.cnf(self.reverse(cfg))

        # perform left-corner transform
        sigma, Ys = {}, cfg.V.copy()
        for p, w in cfg.P:
            if isinstance(p.body[0], NT):
                sigma[p] = 0
        lcfg = lc(cfg)

        # eliminate others
        for p, w in list(lcfg.P):
            if isinstance(p.head, Other):
                lcfg = self.elim(lcfg, p).trim()

        # eliminate non-slashed
        for p, w in list(lcfg.P):
            if (
                p.head != S
                and not isinstance(p.head, Slash)
                and not isinstance(p.head, Other)
            ):
                lcfg = self.elim(lcfg, p).trim()

        # clean-up
        lcfg = self.nullaryremove(lcfg).trim()

        # un-reverse
        gcfg = self.reverse(lcfg)

        return gcfg

    def product(self, cfg, Ps, Xs, Ys):
        """ """
        assert ε not in cfg.Sigma
        R = cfg.R
        zero, one = R.zero, R.one

        pcfg = CFG(R=R)
        add = pcfg.add

        for p, w in cfg.P:
            (head, body) = p

            # recovery rules
            if (
                len(body) >= 2
                and isinstance(body[0], NT)
                and isinstance(body[1], NT)
                and body[0] in Xs
                and body[1] in Ys
            ):
                add(w, head, body[0] * body[1], *body[2:])
            else:
                add(w, head, *body)

            if head in Ys:
                if p not in Ps or body[0] not in Ys:
                    for A in Xs:
                        add(w, A * head, A, *body)
                else:
                    for A in Xs:
                        add(w, A * head, A * body[0], *body[1:])

        # pcfg = pcfg.trim()
        pcfg.make_unary_fsa()
        return pcfg

    def product_just_ps(self, cfg, Ps):
        """ """
        assert ε not in cfg.Sigma
        R = cfg.R
        zero, one = R.zero, R.one

        pcfg = CFG(R=R)
        add = pcfg.add

        for p, w in cfg.P:
            (head, body) = p

            if p not in Ps:
                # recovery rule
                if isinstance(body[0], NT) and isinstance(body[1], NT):
                    add(w, head, body[0] * body[1], *body[2:])
                else:
                    add(w, head, *body)

                for A in cfg.V:
                    add(w, A * head, A, *body)

            else:
                add(w, head, *body)
                for A in cfg.V:
                    add(w, A * head, A * body[0], *body[1:])

        pcfg.make_unary_fsa()
        return pcfg

    def product_terminal(self, cfg):
        """ """
        assert ε not in cfg.Sigma
        R = cfg.R
        zero, one = R.zero, R.one

        pcfg = CFG(R=R)
        add = pcfg.add

        # lift terminals to non-terminals
        for a in cfg.Sigma:
            add(one, NT(a), a)

        for p, w in cfg.P:
            (head, body) = p

            # X → x
            if isinstance(body[0], Sym) and len(p.body) == 1:
                add(w, head, NT(body[0]))
                for a in cfg.Sigma:
                    add(w, NT(a) * head, NT(a), NT(body[0]))

            # X → Y Z
            # TODO: S → b b not handled well

            else:
                if isinstance(body[0], Sym):  # and isinstance(body[1], NT):
                    # See note above
                    add(w, head, NT(body[0]) * body[1], *body[2:])
                    for a in cfg.Sigma:
                        add(w, NT(a) * head, NT(a), body[0], *body[1:])
                else:
                    add(w, head, body[0], *body[1:])
                    for a in cfg.Sigma:
                        add(w, NT(a) * head, NT(a) * body[0], *body[1:])

        pcfg.make_unary_fsa()

        return pcfg

    def locally_normalize(self, G: CFG) -> CFG:
        """Locally normalizes the grammar into a strongly equivalent CFG
        such that the weights of all productions with the same head sum to one.

        Args:
            G (CFG): The CFG to normalize.

        Returns:
            CFG: The normalized CFG.
        """

        Gʹ = G.spawn()

        β = Treesum(G).table(strategy="forwardchain")

        for p, w in G.P:
            wʹ = w * prod([β[X] for X in p.body], start=G.R.one) / β[p.head]
            Gʹ.add(wʹ, p.head, *p.body)

        return Gʹ

    def lift_to_count(self, G: CFG) -> CFG:
        """Lifts the weights of the CFG to include integer weights that count the
        number of derivations.

        Args:
            G (CFG): The CFG to lift.

        Returns:
            CFG: The lifted CFG. The weights are in the product semiring in which the
            first component corresponds to the original weights and the second one
            the the integer counts.
        """

        R = product_semiring_builder(G.R, Integer)
        Gʹ = CFG(R, G.S)

        for p, w in G.P:
            Gʹ.add(R(w, Integer.one), p.head, *p.body)

        return Gʹ

    def lift_to_expression(self, G: CFG) -> CFG:
        """Lifts the weights of the CFG to include the expressions of all the possible
        parses.

        Args:
            G (CFG): The CFG to lift.

        Returns:
            CFG: The lifted CFG. The weights are in the product semiring in which the
            first component corresponds to the original weights and the second one
            the the expressions.
        """

        R_inner = product_semiring_builder(Expr, Expr)
        R_outer = product_semiring_builder(G.R, R_inner)
        Gʹ = CFG(R_outer, G.S)

        for p, w in G.P:
            _body = "".join([str(b) for b in p.body])
            Gʹ.add(
                R_outer(w, R_inner(Sym(f"{p.head._X}→{_body}"), Sym(w.value))),
                p.head,
                *p.body,
            )

        return Gʹ
