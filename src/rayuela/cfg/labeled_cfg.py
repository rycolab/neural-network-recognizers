from collections import defaultdict as dd

from rayuela.base.semiring import Boolean
from rayuela.base.symbol import Sym, ε
from rayuela.base.state import State

from rayuela.cfg.cfg import CFG
from rayuela.cfg.exceptions import InvalidProduction
from rayuela.cfg.nonterminal import NT, S
from rayuela.cfg.production import Production


class LabeledCFG(CFG):
    def __init__(self, R=Boolean):
        super().__init__(R)

        # labels
        self.labels = dd(None)

    def spawn(self):
        return LabeledCFG(R=self.R)

    def delabel(self):
        ncfg = CFG(R=self.R)
        for p, w in self.P:
            ncfg.add(w, p.head, *p.body)
        return ncfg

    def bottom_up(self):
        from rayuela.pda.labeled_pda import LabeledPDA

        pda = LabeledPDA(R=self.R)

        pda.set_I(State(0), self.R.one)
        pda.set_F(State(0), self.R.one)
        pda.set_labels(self.labels)

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

    def add(self, w, label, head, *body):
        if not isinstance(head, NT):
            raise InvalidProduction
        self.V.add(head)

        for elem in body:
            if isinstance(elem, NT):
                self.V.add(elem)
            elif isinstance(elem, Sym):
                self.Sigma.add(elem)
            else:
                raise InvalidProduction

        p = Production(head, body)
        self._P[p] += w
        self.labels[p] = label

    def trim(self):
        # TODO: make sure this shares more code
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
                    ncfg.add(w, self.labels[p], p.head, *p.body)

        ncfg.make_unary_fsa()
        ncfg.freeze()
        return ncfg

    def __dep(self, label):
        if label is None:
            return ""
        else:
            (n1, n2, direction) = label
            if direction == "r":
                return f"{n2}→{n1}"
            elif direction == "l":
                return f"{n1}←{n2}"
            else:
                raise AssertionError("Invalid Direction")

    def __str__(self):
        return "\n".join(
            f"{p}\t{w}\t{self.__dep(self.labels[p])}"
            for (p, w) in sorted(
                self.P,
                key=lambda x: (len(str(x[0].head)), str(x[0].head), len(str(x[0]))),
            )
        )
