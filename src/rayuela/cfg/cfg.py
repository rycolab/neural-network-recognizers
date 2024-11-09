import copy
from itertools import product
from math import prod
from typing import List, Sequence, Tuple, Type, Union

from frozendict import frozendict

import rayuela
from rayuela.base.misc import straight
from rayuela.base.semiring import Boolean, Semiring
from rayuela.base.symbol import Expr, Sym, ε
from rayuela.cfg.exceptions import InvalidProduction
from rayuela.cfg.nonterminal import NT, Other, S, Triplet
from rayuela.cfg.production import Production
from rayuela.cfg.treesum import Treesum
from rayuela.fsa.fsa import FSA
from rayuela.base.state import State


class CFG:
    def __init__(self, R: Type[Semiring] = Boolean, _S: NT = S):
        # A weighted context-free grammar is a 5-tuple <R, Σ, V, P, S> where
        # • R is a semiring;
        # • Σ is an alphabet of terminal symbols;
        # • V is an alphabet of non-terminal symbols;
        # • P is a finite relation V × (Σ ∪ V)* × R;
        # • S ∈ V is a distinguished started symbol.

        # semiring
        self.R = R

        # alphabet
        self.Sigma = set([])

        # non-terminals
        self.V = set([S])

        # productions
        self._P = self.R.chart()

        # unique start symbol
        self.S = _S

        # unary FSA
        self.unary_fsa = None

    @property
    def terminal(self):
        for p, w in self.P:
            (head, body) = p
            if len(body) == 1 and (
                isinstance(body[0], Sym) or isinstance(body[0], Sym)
            ):
                yield p, w

    @property
    def unary(self):
        for p, w in self.P:
            (head, body) = p
            if len(body) == 1 and isinstance(body[0], NT):
                yield p, w

    @property
    def binary(self):
        for p, w in self.P:
            (head, body) = p
            if len(body) == 2 and isinstance(body[0], NT) and isinstance(body[1], NT):
                yield p, w

    @property
    def size(self):
        size = 0
        for (_, body), _ in self.P:
            for elem in body:
                if elem != ε:
                    size += 1
            size += 1
        return size

    @property
    def num_rules(self):
        return len(self._P)

    @property
    def ordered_V(self):
        """Returns a list of nonterminals ordered by their lexicographical index"""
        V = list(self.V)
        V.sort(key=lambda a: str(a.X))
        return V

    def is_linear(self, direction: str) -> bool:
        """Returns whether the grammar is linear.

        Args:
            direction (str): The direction of the linear grammar.
            Either "left" or "right".

        Returns:
            bool: Whether the grammar is linear.
        """
        for p, _ in self.P:
            elements = p.body[1:] if direction == "left" else p.body[:-1]
            for elem in elements:
                if isinstance(elem, NT):
                    return False

        return True

    def to_fsa(self) -> FSA:
        """Returns a WFSA representing the same language if the grammar is linear.

        Returns:
            FSA: The WFSA.
        """

        assert self.is_linear("right") or self.is_linear("left")

        A = FSA(self.R)
        A.add_I(State(self.S), self.R.one)

        if self.is_linear("right"):
            for p, _ in self.P:
                if isinstance(p.body[-1], NT):
                    if len(p.body) == 2:
                        A.add_arc(
                            State(p.head),
                            Sym(p.body[0].value),
                            State(p.body[-1]),
                            self.R.one,
                        )
                    else:
                        A.add_arc(
                            State(p.head),
                            Sym(p.body[0].value),
                            State((p.head, 1)),
                            self.R.one,
                        )
                        for i in range(1, len(p.body) - 2):
                            A.add_arc(
                                State((p.head, i)),
                                Sym(p.body[i].value),
                                State((p.head, i + 1)),
                                self.R.one,
                            )
                        A.add_arc(
                            State((p.head, len(p.body) - 2)),
                            Sym(p.body[-2].value),
                            State(p.body[-1]),
                            self.R.one,
                        )
                else:
                    if len(p.body) == 1:
                        A.add_arc(
                            State(p.head),
                            Sym(p.body[0].value),
                            State((p.head, 1)),
                            self.R.one,
                        )
                        A.add_F(State((p.head, 1)), self.R.one)
                    else:
                        A.add_arc(
                            State(p.head),
                            Sym(p.body[0].value),
                            State((p.head, 1)),
                            self.R.one,
                        )
                        for i in range(1, len(p.body) - 1):
                            A.add_arc(
                                State((p.head, i)),
                                Sym(p.body[i].value),
                                State((p.head, i + 1)),
                                self.R.one,
                            )
                        A.add_arc(
                            State((p.head, len(p.body) - 1)),
                            Sym(p.body[-1].value),
                            State((p.head, len(p.body))),
                            self.R.one,
                        )
                        A.add_F(State((p.head, len(p.body))), self.R.one)

        return A

    def leftmost_nonterminal_fsa(self) -> FSA:
        """Constructs the WFSA that models the transition probabilities between
        heads of productions and the leftmost symbols in the bodies, that is, it
        constructs the WFSA with the transitions `(X) - Y/w -> (Y)` for the productions
        of the form `X -> Y ... / w`.

        Returns:
            FSA: The WFSA.
        """

        A = FSA(self.R)
        for p, w in self.P:
            assert len(p.body) > 0
            if len(p.body) > 0 and isinstance(p.body[0], NT):
                X, Y = p.head, p.body[0]
                A.add_arc(State(X), Sym(Y._X), State(Y), w)

        return A

    def leftmost_derivation_fsa(self) -> FSA:
        """Constructs the WFSA that models the leftmost derivations in the grammar.

        Returns:
            FSA: The WFSA.
        """
        from rayuela.fsa.pathsum import Pathsum

        assert self.in_nf

        ps = Pathsum(self.leftmost_nonterminal_fsa()).lehmann()

        A = FSA(self.R)

        # create states from the suffixes of the sequence of
        # nonterminals of the right-hand side of productions,
        # without the first one
        for p, _ in self.P:
            for i in range(1, len(p.body)):
                # e.g., for S → X Y Z W add
                # YZW, ZW, W
                A.Q.add(State(tuple([x._X for x in p.body[i:]])))

        # initial state is (S,)
        A.set_I(State((self.S._X,)), self.R.one)
        # final state is F
        A.set_F(State("F"), self.R.one)

        for X, Y in dict(ps).keys():
            for p, w in self.P:
                if not all([isinstance(b, NT) for b in p.body]):
                    continue

                # X ~~~~> Y
                _Y = p.head
                if _Y == Y.idx:
                    # X ~~~~> Y → Z
                    Z = p.body[0]
                    for q in A.Q:
                        if type(q.idx) == tuple and q.idx[0] == X.idx._X:
                            if len(p.body) == 1:
                                A.add_arc(q, Sym(Z._X), State("F"), ps[(X, Y)] * w)
                            else:
                                A.add_arc(
                                    q,
                                    Sym(Z._X),
                                    State(tuple([x._X for x in p.body[1:]])),
                                    ps[(X, Y)] * w,
                                )

        for q in A.Q:
            if q != State("F"):
                if type(q.idx) == tuple and len(q.idx) == 1:
                    # (A,) -- A/1 --> F
                    A.add_arc(q, Sym(q.idx[0]), State("F"), self.R.one)
                if len(q.idx) > 1:
                    # (A, B, ...) -- A/1 --> (B, ...)
                    A.add_arc(q, Sym(q.idx[0]), State(q.idx[1:]), self.R.one)
                # (A, B, ...) -- ε/1 --> F
                A.add_arc(q, ε, State("F"), self.R.one)

        return A

    def leftmost_derivation_fsa_cnf(self) -> FSA:
        """Constructs the WFSA that models the leftmost derivations in the grammar.
        Returns:
            FSA: The WFSA.
        """
        from rayuela.fsa.pathsum import Pathsum

        assert self.in_cnf()

        ps = Pathsum(self.leftmost_nonterminal_fsa()).lehmann()

        A = FSA(self.R)
        for X, Y in dict(ps).keys():
            for p, w in self.P:
                if len(p.body) == 2 and isinstance(p.body[0], NT):
                    _Y, (Z, W) = p.head, p.body
                    if _Y == Y.idx:
                        A.add_arc(
                            State(X.idx), Sym(Z._X), State(W), ps[(X, Y)] * w
                        )  # from X emit Z, go to W

        for Z in self.V:
            A.add_arc(State(Z), Sym(Z._X), State("F"), self.R.one)
            A.add_arc(State(Z), ε, State("F"), self.R.one)

        A.add_I(State(self.S), self.R.one)
        A.add_F(State("F"), self.R.one)

        return A

    @property
    def is_locally_normalized(self) -> bool:
        """Returns whether the grammar is locally normalized.

        Returns:
            bool: Whether the grammar is locally normalized.
        """
        for head in self.V:
            total = self.R.zero
            for p, w in self.P:
                if p.head == head:
                    total += w
            if total != self.R.one:
                return False

        return True

    def _compute_normalized_weight_brute(
        self,
        Q: List[State],
        S: List[Sym],
        w: Semiring,
        wʹ: Semiring,
        i: State,
        a: Sym,
        j: State,
        q: State,
        qʹ: State,
        A: FSA,
    ) -> Semiring:
        if len(Q) > 0:
            # Divide out the weights of the transitions that have to be backtracked
            _w = (
                prod(
                    [A.δ[Q[ii]][S[ii + 1]][Q[ii + 1]] for ii in range(len(Q) - 1)],
                    start=self.R.one,
                )
                * A.δ[q][S[0]][Q[0]]
            )
            # Cancel the final weight that is not relevant anymore
            _w *= A.ρ[Q[-1]]

        else:
            _w = A.ρ[q]  # Divide out the final weight that is not relevant anymore

        if _w == self.R.zero:
            return self.R.zero

        # Include the weight of the production (from the original PDA/CFG),
        # the weight of the transition that has to be taken in A,
        # and new relevant final weight
        w_ = wʹ * A.ρ[qʹ] * w / _w

        return w_

    def _compute_normalized_weight(
        self,
        path: List[Tuple[State, Sym, Semiring, State]],
        w: Semiring,
        wʹ: Semiring,
        i: State,
        a: Sym,
        j: State,
        q: State,
        qʹ: State,
        A: FSA,
    ) -> Semiring:
        if len(path) > 0:
            # Divide out the weights of the transitions that have to be backtracked
            _w = prod([w for _, _, w, _ in path], start=self.R.one)
            # Cancel the final weight that is not relevant anymore
            _w *= A.ρ[path[-1][-1]]

        else:
            _w = A.ρ[q]  # Divide out the final weight not relevant anymore

        if _w == self.R.zero:
            return self.R.zero

        # Include the weight of the production (from the original PDA/CFG),
        # the weight of the transition that has to be taken in A,
        # and new relevant final weight
        w_ = wʹ * A.ρ[qʹ] * w / _w

        return w_

    def _add_normalized_transitions(
        self,
        P: "rayuela.pda.pda.PDA",
        Pʼ: "rayuela.pda.pda.PDA",
        w: Semiring,
        i: State,
        a: Sym,
        j: State,
        head: Sequence[Union[Sym, NT]],
        body: Sequence[Union[Sym, NT]],
        A: FSA,
    ) -> None:
        for q in A.Q:
            for qʹ, wʹ in A.δ[q][Sym(head[0]._X)].items():
                for path in A.enumerate_paths(q=q, label=[Sym(b._X) for b in body]):
                    w_ = self._compute_normalized_weight(path, w, wʹ, i, a, j, q, qʹ, A)
                    if w_ == self.R.zero:
                        continue
                    qs = [b[-1] for b in path]
                    for X in P.Gamma.union({ε}):
                        body_ = zip(qs, body)
                        Pʼ.add(w_, i, a, j, ((q, X), (qʹ, head[0])), (q, X), *body_)

    def to_locally_normalized_bottom_up_pda(self) -> "rayuela.pda.PDA":
        """We assume the standardized way of converting a CFG to a bottom-up PDA.
        This method the PDA that is topologically the same as the standard bottom-up
        PDA corresponding to the CFG, but with locally normalized weights.

        Returns:
            PDA: The locally-normalized bottom-up PDA.
        """
        assert self.is_locally_normalized, "Grammar is not locally normalized"
        # assert self.in_cnf(), "Grammar is not in CNF"

        from rayuela.pda.pda import PDA

        A = self.leftmost_derivation_fsa().trim().determinize().rename_states()
        assert A.deterministic  # Sanity check

        P = self.bottom_up()

        Pʼ = PDA(self.R, _S=(-1, P.S))  # Just a special tuple final stack symbol

        qI = P.new_state()
        Pʼ.set_I(qI, self.R.one)
        for q, w in P.I:
            Pʼ.add(w, qI, ε, q, ((list(A.I)[0][0], ε),))

        qF = Pʼ.new_state()
        Pʼ.set_F(qF, self.R.one)
        for q, w in P.F:
            for qʹ, _ in A.F:
                Pʼ.add(
                    w, q, ε, qF, ((-1, P.S),), (list(A.I)[0][0], ε), (qʹ, P.S)
                )  # "Project" back to S

        for (i, a, j, head, body), w in P.arcs:
            self._add_normalized_transitions(P, Pʼ, w, i, a, j, head, body, A)

        return Pʼ.rename_states()

    def to_locally_normalized_bottom_up_pda_gnf(self) -> "rayuela.pda.PDA":
        """Implements a construction similar to the one given by
        Abney et al. (1999). Assumes the CFG is in GNF.

        Returns:
            PDA: The locally normalized bottom-up PDA.
        """
        assert self.is_locally_normalized, "Grammar is not locally normalized"
        assert self.in_gnf, "Grammar is not in GNF"

        from rayuela.pda.pda import PDA

        pda = PDA(R=self.R)
        one = self.R.one

        # add W_ε
        pda.Gamma.add(NT((ε,)))

        pda.S = NT((self.S.X,))

        # create stack symbols from the suffixes of the sequence of
        # nonterminals of the right-hand side of productions
        for p, _ in self.P:
            for i in range(1, len(p.body)):
                # e.g., for S → a X Y Z add
                # W_XYZ, W_YZ, W_Z
                pda.Gamma.add(NT(p.body[i:]))

        # p = X → a 𝝱
        for p, w in self.P:
            # W_X𝞪
            nts = {nt.X for nt in pda.Gamma if nt.X[0] == p.head}
            for nt in nts:
                if len(p.body) == 1:
                    # W_X𝞪, ε -- a/w --> W_ε, W_X𝞪
                    pda.add(w, State(nt), p.body[0], State((ε,)), (), *(NT(nt),))
                else:
                    # W_X𝞪, ε -- a/w --> W_𝝱, W_X𝞪
                    pda.add(w, State(nt), p.body[0], State(p.body[1:]), (), *(NT(nt),))

        # post-processing rules
        for nt1, nt2 in product(pda.Gamma - {NT((ε,))}, repeat=2):
            if len(nt2.X) == 1:
                pda.add(one, State((ε,)), ε, State((ε,)), (nt1, nt2), *(nt1,))
            else:
                pda.add(one, State((ε,)), ε, State(nt2.X[1:]), (nt1, nt2), *(nt1,))

        # initial and final states
        pda.set_I(State((self.S.X,)), one)
        pda.set_F(State((ε,)), one)

        print(pda)

    def w(self, p):
        return self._P[p]

    def spawn(self):
        return CFG(R=self.R, _S=self.S)

    def locally_normalize(self) -> "CFG":
        """Constructs an equivalent locally normalized CFG.

        Returns:
            CFG: The locally normalized CFG.
        """
        from rayuela.cfg.transformer import Transformer

        return Transformer().locally_normalize(self)

    def make_unary_fsa(self):
        one = self.R.one
        fsa = FSA(R=self.R)

        # add a state for every non-terminal
        for X in self.V:
            fsa.add_state(State(X))

        # add arcs between every pair of unary rules
        for (head, body), w in self.unary:
            fsa.add_arc(State(body[0]), ε, State(head), w)

        # add initial and final weight one for every state
        for q in list(fsa.Q):
            fsa.set_I(q, one)
            fsa.set_F(q, one)

        self.unary_fsa = fsa

    def make_chain_fsa(self):
        """makes fsa for computation of nullability"""
        one = self.R.one
        fsa = FSA(R=self.R)
        has_incoming_edge = self.R.chart()

        # add a state for every non-terminal
        for X in self.V:
            fsa.add_state(State(X))

        # add arcs between every pair of unary rules
        for (head, body), w in self.P:
            if len(body) == 0 or body[0] == ε:
                fsa.set_F(head, w)
            elif len(body) == 1:
                fsa.add_arc(State(head), ε, State(body[0]), w)

        # If a state doesn't have any incoming edges, we set it's initial weight to one
        for q in fsa.Q:
            has_incoming_edge[q] = False

        for q in fsa.Q:
            for a, p, w in fsa.arcs(q):
                has_incoming_edge[p] = True

        for q in fsa.Q:
            if not has_incoming_edge[q]:
                fsa.set_I(q, one)

        return fsa

    def eps_partition(self):
        """makes a new grammar can only generate epsilons"""
        ecfg = self.spawn()
        ncfg = self.spawn()

        def has_terminal(body):
            for elem in body:
                if (isinstance(elem, Sym) or isinstance(elem, Expr)) and elem != ε:
                    return True
            return False

        for p, w in self.P:
            head, body = p
            if has_terminal(body):
                ncfg.add(w, head, *body)
            elif len(body) == 1 and body[0] == ε:
                ecfg.add(w, head, *body)
            else:
                ncfg.add(w, head, *body)
                ecfg.add(w, head, *body)

        return ecfg, ncfg

    @property
    def P(self):
        for p, w in self._P.items():
            yield p, w

    def P_byhead(self, X, unary=True):
        for p, w in self._P.items():
            if X == p.head:
                if not unary and len(p.body) == 1 and isinstance(p.body[0], NT):
                    continue
                yield p, w

    def adds(self, prod: str) -> None:
        """Adds a production in a string format to the grammar.
        The production should be of the form:
        `head -> body_1 / weight_1 | body_2 / weight_2 ...`.
        The individual elements of the body should be separated by spaces.

        (Unrelated to the `from_string` method.)

        Args:
            prod (str): The production to add.
        """
        head_str, bodies = prod.split("->")

        if "|" in bodies:
            bodies = bodies.split("|")
        else:
            bodies = [bodies]

        for body in bodies:
            if "/" in body:
                body_str, weight = body.split("/")
            else:
                body_str, weight = body, "1"

            body_str = body_str.strip().split()

            head = NT(head_str.strip())
            tail = []
            for x in body_str:
                x = x.strip()
                if x.isupper():
                    x = NT(x)
                elif x.islower() or not x.isalpha():
                    if x == "eps":
                        x = ε
                    else:
                        x = Sym(x)
                tail.append(x)

            self.add(self.R(float(weight)), head, *tail)

    def add(self, w, head, *body):
        assert isinstance(self.V, set), "Cannot add to frozen CFG"
        assert isinstance(self.Sigma, set), "Cannot add to frozen CFG"
        assert isinstance(self._P, dict), "Cannot add to frozen CFG"

        if not isinstance(head, NT):
            raise InvalidProduction
        if not isinstance(w, Semiring):
            w = self.R(w)

        self.V.add(head)

        for elem in body:
            if isinstance(elem, NT):
                self.V.add(elem)
            elif (isinstance(elem, Sym) or isinstance(elem, Expr)) and elem != ε:
                self.Sigma.add(elem)
            elif elem != ε:
                raise InvalidProduction

        self._P[Production(head, body)] += w

    def get_productions(self):
        return self._P

    def freeze(self):
        self.Sigma = frozenset(self.Sigma)
        self.V = frozenset(self.V)
        self._P = frozendict(self._P)

    def copy(self):
        return copy.deepcopy(self)

    def fresh(self):
        ncfg = self.spawn()
        for p, w in self.P:
            nbody = []
            for elem in p.body:
                if isinstance(elem, NT):
                    nbody.append(NT(str(elem)))
                elif isinstance(elem, Sym) or isinstance(elem, Expr):
                    nbody.append(elem)
            ncfg.add(w, NT(str(p.head)), *nbody)
        ncfg.make_unary_fsa()

        return ncfg

    def accessible(self):
        from rayuela.cfg.transformer import Transformer
        from rayuela.cfg.treesum import Treesum

        boo = Transformer().booleanize(self)

        A = set([])
        for item, v in Treesum(boo).backwardchain().items():
            if v != Boolean.zero:
                A.add(item)

        return A

    def coaccessible(self):
        from rayuela.cfg.transformer import Transformer
        from rayuela.cfg.treesum import Treesum

        boo = Transformer().booleanize(self)
        C = set([])

        for item, v in Treesum(boo).forwardchain().items():
            if v != Boolean.zero:
                C.add(item)

        return C

    def treesum(self):
        treesum = Treesum(self)
        return treesum.sum()

    def accept(self, s):
        from rayuela.cfg.transformer import Transformer

        s = straight(s, R=Boolean)
        ncfg = Transformer().booleanize(self)
        return ncfg.intersect_fsa(s).treesum()

    def trim(self):
        return self._trim()

    def cotrim(self):
        return self._cotrim()

    def cnf(self):
        from rayuela.cfg.transformer import Transformer

        return Transformer().cnf(self)

    def elim(self, p):
        from rayuela.cfg.transformer import Transformer

        return Transformer().elim(self, p)

    def unfold(self, p, i):
        from rayuela.cfg.transformer import Transformer

        return Transformer().unfold2(self, p, i)

    def removenullary(self):
        from rayuela.cfg.transformer import Transformer

        return Transformer().nullaryremove(self)

    def _trim(self):
        A, C = self.accessible(), self.coaccessible()
        AC = A.intersection(C)

        ncfg = self.spawn()
        for p, w in self.P:
            if p.head in AC and w != self.R.zero:
                invalid = False
                for elem in p.body:
                    if isinstance(elem, NT) and elem not in AC:
                        invalid = True
                if not invalid:
                    ncfg.add(w, p.head, *p.body)

        ncfg.make_unary_fsa()
        # ncfg.freeze()
        return ncfg

    def _cotrim(self):
        C = self.coaccessible()

        ncfg = self.spawn()
        for p, w in self.P:
            if p.head in C and w != self.R.zero:
                invalid = False
                for elem in p.body:
                    if isinstance(elem, NT) and elem not in C:
                        invalid = True
                if not invalid:
                    ncfg.add(w, p.head, *p.body)

        ncfg.make_unary_fsa()
        # ncfg.freeze()
        return ncfg

    def nozero(self):
        ncfg = self.spawn()
        for p, w in self.P:
            if w != self.R.zero:
                ncfg.add(w, p.head, *p.body)

        ncfg.make_unary_fsa()
        # ncfg.freeze()
        return ncfg

    @classmethod
    def from_string(cls, string, R, comment="#"):
        cfg = CFG(R=R)
        for line in string.split("\n"):
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == comment:
                continue

            head_str, tmp = line.split("→")
            tail_str, weight = tmp.split(":")
            tail_str = tail_str.strip().split()

            head = NT(head_str.strip())
            tail = []
            for x in tail_str:
                x = x.strip()
                if x.isupper():
                    x = NT(x)
                elif x.islower() or not x.isalpha():
                    x = Sym(x)
                tail.append(x)

            cfg.add(R(float(weight)), head, *tail)

        cfg.make_unary_fsa()
        return cfg

    @classmethod
    def get_moore_cfg(cls, grammar_path, R):
        """
        Wrapper for getting cfgs from the Moore grammars
        See https://users.sussex.ac.uk/~johnca/cfg-resources/index.html
        """
        moore_grammar = []
        with open(grammar_path, "r") as f:
            moore_grammar_str = f.read().strip()
        moore_grammar_str = moore_grammar_str.replace(":", "-colon-").replace(
            "non_cyclic", "NON_CYCLIC"
        )
        moore_grammar_str = moore_grammar_str.replace("``", "-quot_ini-").replace(
            "''", "-quot_fin-"
        )
        moore_grammar = moore_grammar_str.split("\n")
        if moore_grammar[0][0] == ";":
            # clean ct-grammar-eval from initial comments
            moore_grammar = moore_grammar[5:]
        rules = []
        prev, lhs = None, None
        for line in moore_grammar:
            line = line.strip()
            if prev is None or prev == "":
                lhs = "S" if line == "SIGMA" else line
            elif line != "" and lhs != line:
                rules.append(f"{lhs} → {line}:1")
            prev = line
        cfg = cls.from_string("\n".join(rules), R)
        return cfg

    def to_fsta(self):
        """converts a grammar to a finite-state tree automaton"""
        from rayuela.fsta.fsta import FSTA

        fsta = FSTA(R=self.R)

        # add rank > 1 rules
        for p, w in self.P:
            fsta.add(w, p.head, State(p.head), [State(X) for X in p.body])

        # add rank 0 rules
        for a in self.Sigma:
            fsta.add(self.R.one, a, State(a), [])

        return fsta

    # TODO: Change this to a property
    def in_cnf(self):
        """check if grammar is in cnf"""
        for p, w in self.P:
            (head, body) = p
            if head == self.S and len(body) == 1 and body[0] == ε:
                # S → ε
                continue
            elif (
                head in self.V
                and len(body) == 2
                and all([elem in self.V and elem != self.S for elem in body])
            ):
                # A → B C
                continue
            elif (
                head in self.V
                and len(body) == 1
                and body[0] in self.Sigma
                and body[0] != ε
            ):
                # A → a
                continue
            else:
                return False
        return True

    @property
    def in_gnf(self) -> bool:
        """Checks if the grammar is in Greibach Normal Form (GNF)."""
        for p, _ in self.P:
            (_, body) = p
            if len(body) == 0:
                return False
            if isinstance(body[0], NT):
                return False
            for elem in body[1:]:
                if not isinstance(elem, NT) or elem == S:
                    return False
        return True

    @property
    def in_nf(self) -> bool:
        """Checks if the grammar is in a normal forms that assumes productions
        of one the forms S → ε, A → a or A → B C D ..."""
        for p, w in self.P:
            (head, body) = p
            if head == self.S and len(body) == 1 and body[0] == ε:
                # S → ε
                continue
            elif head in self.V and all(
                [elem in self.V and elem != self.S for elem in body]
            ):
                # A → B C
                continue
            elif (
                head in self.V
                and len(body) == 1
                and body[0] in self.Sigma
                and body[0] != ε
            ):
                # A → a
                continue
            else:
                return False
        return True

    def shift_reduce(self):
        return self.bottom_up()

    def bottom_up(self):
        from rayuela.pda.pda import PDA

        pda = PDA(R=self.R)

        pda.set_I(State(0), self.R.one)
        pda.set_F(State(0), self.R.one)

        for p, w in self.P:
            # A → x
            if len(p.body) == 1 and p.body[0] in self.Sigma:
                pda.add(w, State(0), p.body[0], State(0), (p.head,))
            # A → B C ...
            elif all([X in self.V for X in p.body]):
                pda.add(w, State(0), ε, State(0), (p.head,), *p.body)
            else:
                raise AssertionError(
                    "Grammar mixes terminals and non-terminals in a production"
                )

        return pda

    def top_down(self):
        from rayuela.pda.pda import PDA

        pda = PDA(R=self.R)

        pda.set_I(State(0), self.R.one)
        pda.set_F(State(0), self.R.one)

        for p, w in self.P:
            # A → x
            if len(p.body) == 1 and p.body[0] in self.Sigma:
                pda.add(w, State(0), p.body[0], State(0), (), p.head)
            # A → B C ...
            elif all([X in self.V for X in p.body]):
                pda.add(w, State(0), ε, State(0), p.body, p.head)
            else:
                raise AssertionError(
                    "Grammar mixes terminals and non-terminals in a production"
                )

        return pda

    def cyclic(self, reverse=True):
        """
        Returns True if grammar is cyclic and (reverse) topological ordering
        if it is acyclic
        """

        def has_cycles(X):
            nonlocal counter
            𝜷[X] = Boolean.one
            started[X] = counter
            counter += 1
            X_productions = (p for p, w in self.P if p[0] == X and w != self.R.zero)
            for p in X_productions:
                _, body = p
                for elem in body:
                    if elem in self.Sigma:
                        continue
                    elif 𝜷[elem] == Boolean.one:  # cycle detected
                        return True
                    elif has_cycles(elem):  # propagate cycle
                        return True
            𝜷[X] = Boolean.zero
            return False

        𝜷 = Boolean.chart()
        started = {}
        counter = 0
        cyclic = has_cycles(self.S)
        if reverse:
            sort = [k for k, v in sorted(started.items(), key=lambda item: item[1])]
        else:
            sort = [
                k
                for k, v in sorted(
                    started.items(), key=lambda item: item[1], reverse=True
                )
            ]
        return cyclic, sort

    def intersect_fsa(self, fsa):  # noqa: C901
        """
        Intersects cfg with fsa and returns the resulting parse-forest grammar
        Semiringified weighted case from Nederhof and Satta (2003)
        """
        assert self.R == fsa.R

        pfg = self.spawn()

        def get_intersecting_rule(head, body, qs):
            NTs = []
            new_head = Triplet(qs[0], head, qs[-1])

            for i in range(len(qs) - 1):
                NTs.append(Triplet(qs[i], body[i], qs[i + 1]))

            if len(NTs) == 0:
                NTs.append(ε)
            return new_head, NTs

        # rules from cfg
        for p, w in self.P:
            (head, body) = p
            if len(body) == 1 and body[0] == ε:
                for q in product(fsa.Q, repeat=1):
                    h, b = get_intersecting_rule(head, body, q)
                    pfg.add(w, h, *b)
            else:
                for qs in product(fsa.Q, repeat=len(body) + 1):
                    h, b = get_intersecting_rule(head, body, qs)
                    pfg.add(w, h, *b)

        # S rules
        for qi, wi in fsa.I:
            for qf, wf in fsa.F:
                b = Triplet(qi, self.S, qf)
                pfg.add(wi * wf, self.S, b)

        # terminal rules
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                h = Triplet(i, a, j)
                pfg.add(w, h, a)

        return pfg

    def intersect_fsa_ε(self, fsa):  # noqa: C901
        """
        Intersects cfg with fsa and returns the resulting parse-forest grammar
        Semiringified weighted case from Nederhof and Satta (2003)
        """
        assert self.R == fsa.R

        pfg = self.spawn()

        def get_intersecting_rule(head, body, qs):
            NTs = []
            new_head = Triplet(qs[0], head, qs[-1])

            for i in range(len(qs) - 1):
                NTs.append(Triplet(qs[i], body[i], qs[i + 1]))

            if len(NTs) == 0:
                NTs.append(ε)
            return new_head, NTs

        # rules from cfg
        for p, w in self.P:
            (head, body) = p
            if len(body) == 1 and body[0] == ε:
                for q in product(fsa.Q, repeat=1):
                    h, b = get_intersecting_rule(head, body, q)
                    pfg.add(w, h, *b)
            else:
                for qs in product(fsa.Q, repeat=len(body) + 1):
                    h, b = get_intersecting_rule(head, body, qs)
                    pfg.add(w, h, *b)

        # S rules
        for qi, wi in fsa.I:
            for qf, wf in fsa.F:
                b = Triplet(qi, self.S, qf)
                pfg.add(wi * wf, self.S, Other(b))

        for qs in product(fsa.Q, repeat=3):
            head, (start, epsilon) = get_intersecting_rule(
                self.S,
                (
                    self.S,
                    ε,
                ),
                qs,
            )
            pfg.add(self.R.one, Other(head), Other(start), epsilon)

        for qs in product(fsa.Q, repeat=2):
            (q1, q2) = qs
            head = Triplet(q1, self.S, q2)
            pfg.add(self.R.one, Other(head), head)

        # terminal rules
        for i in fsa.Q:
            for a, j, w in fsa.arcs(i):
                h = Triplet(i, a, j)
                pfg.add(w, h, a)

        # ADDING PRE-TERMINAL RULES TO HANDLE ε-ΤRANSITIONS

        for a in self.Sigma:
            # SENSE TERMINALS ()
            for qs in product(fsa.Q, repeat=3):
                h, b = get_intersecting_rule(
                    a,
                    (
                        ε,
                        a,
                    ),
                    qs,
                )
                pfg.add(self.R.one, h, *b)

        return pfg

    def to_latex(self):
        """
        Prints production rules in latex syntax
        """
        latex = []
        for p, w in self.P:
            latex.append(
                f"& \weightedproduction{{\\text{{{str(p.head)}}}}}"
                + f"{{\\text{{{' '.join([str(child) for child in p.body])}}}}}"
                + f"{{\\text{{{str(w)}}}}}"
            )
        latex = "\\\\ \n".join(latex)
        print(latex)

    def __str__(self):
        return "\n".join(
            f"{p}\t{w}"
            for (p, w) in sorted(
                self.P,
                key=lambda x: (len(str(x[0].head)), str(x[0].head), len(str(x[0]))),
            )
            if w != self.R.zero
        )

    def _repr_html_(self):  # noqa: C901
        """
        When returned from a Jupyter cell, this will generate the CFG visualization.
        """
        self_str = str(self)
        return self_str.replace("\n", "<br>").replace("\t", " : ")
