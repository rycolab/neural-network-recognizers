# import random

from rayuela.base.semiring import Real, Boolean, Tropical, String
from rayuela.base.symbol import Sym
# from rayuela.base.misc import _random_weight as rw


from rayuela.cfg.cfg import CFG
from rayuela.cfg.nonterminal import S, NT
from rayuela.cfg.production import Production

# symbols
a, b, c = Sym("a"), Sym("b"), Sym("c")
A, B, C = NT("A"), NT("B"), NT("C")


class CFGExamples():

    def broke_earley(self, R=Tropical):
        zero, one = R.zero, R.one

        cfg = CFG.from_string("""
            A → a	37
            A → B B	36
            A → B A	14
            B → a	36
            B → B B	5
            B → A B	5
            S → a	3
            S → B A	34
            S → A B	37
            S → A A	32
        """.strip(), Tropical)

        cfg.make_unary_fsa()

        return cfg

    def broke_earley2(self, R=Tropical):
        zero, one = R.zero, R.one

        cfg = CFG.from_string("""
            A → b	39
            A → a	15
            A → A A	17
            A → B B	2
            B → b	28
            B → a	18
            B → A A	4
            B → B B	45
            S → b	21
            S → a	42
            S → A B	9
        """.strip(), Tropical)

        cfg.make_unary_fsa()

        return cfg

    def non_cnf(self, R=Real):
        cfg = CFG.from_string("""
            A → b	1.0
            A → B a	1.0
            B → b	1.0
            B → B a	1.0
            B → A A B	1.0
            S → B a B A	1.0
            """.strip(), R
        )

        cfg.make_unary_fsa()

        return cfg

    def grammar3(self, R=Tropical):
        cfg = CFG.from_string("""
            A → b	2.0
            A → a	4.0
            A → A A	1.0
            S → b	1.0
            S → a	4.0
            S → A A	2.0""".strip(), R
        )

        cfg.make_unary_fsa()

        return cfg

    def grammar4(self, R=Tropical):
        cfg = CFG.from_string("""A → a	33
            A → b	19
            A → B A	20
            A → A B	10
            B → a	45
            B → b	43
            B → B A	15
            S → a	19
            S → b	0
            S → B A	37
            S → A A	3
        """.strip(), R)

        cfg.make_unary_fsa()

        return cfg

    def grammar5(self, R=Tropical):


        cfg = CFG.from_string(
        """
            A → S	4
            A → A	31
            A → B	23
            A → a	22
            A → A A	47
            A → B A	29
            B → S	42
            B → A	31
            B → B	2
            B → a	31
            S → a	4
            S → A B	0
            S → B B	36""".strip(), R)
        cfg.make_unary_fsa()
        return cfg

    def grammar6(self, R=Tropical):

        cfg = CFG.from_string(
        """
            A → A	27
            A → a	1
            S → A	43
            S → a	45
        """.strip(), R)
        cfg.make_unary_fsa()
        return cfg

    def grammar7(self, R=Tropical):

        cfg = CFG.from_string(
        """
            A → B	8
            A → A	50
            A → a	18
            A → B A	22
            A → A B	27
            B → B	15
            B → A	29
            B → a	0
            B → B A	22
            S → B	46
            S → A	27
            S → a	47
            S → B A	41
            S → A B	37
        """.strip(), R)
        cfg.make_unary_fsa()
        return cfg

    def grammar8(self, R=Tropical):

        cfg = CFG.from_string(
        """
        A → A	43
        A → a	6
        A → A A	17
        S → A	21
        S → a	20
        """.strip(), R)
        cfg.make_unary_fsa()
        return cfg


    def grammar9(self, R=Tropical):

        cfg = CFG.from_string(
        """
        #A → A	0.1
        #A → B	0.1
        A → a	0.1
        #B → A	0.1
        #B → B	0.1
        B → a	0.1
        B → A A	0.1
        #S → A	0.1
        #S → B	0.1
        S → a	0.1
        S → A B	0.1
        S → B B	0.1
        """.strip(), R)
        cfg.make_unary_fsa()
        return cfg

    def grammar10(self, R=Tropical):

        cfg = CFG.from_string(
        """
        A → B	0.1
        A → A	0.1
        A → a	0.1
        A → A B	0.1
        B → B	0.1
        B → A	0.1
        B → a	0.1
        B → B B	0.1
        B → A A	0.1
        S → B	0.1
        S → A	0.1
        S → a	0.1
        S → B B	0.1
        S → A B	0.1
        """.strip(), R)
        cfg.make_unary_fsa()
        return cfg

    def example_catalan(self, R=Real):

        zero, one = R.zero, R.one

        cfg = CFG(R=R)
        Sigma = set([a])
        NTs = set([S, A])

        cfg.add(rw(R), S, A, A)
        cfg.add(rw(R), A, A, A)
        cfg.add(rw(R), A, a)

        #cfg.add(one, S, A, A)
        #cfg.add(one, A, A, A)
        #cfg.add(one, A, a)

        cfg.make_unary_fsa()

        return cfg

    def example1(self, R, rand=False):
        # A permissive grammar over a specific alphabet and  non-terminal set.

        Sigma = set([a, b])
        NTs = set([S, A, B])
        zero, one = R.zero, R.one
        cfg = CFG(R=R)
        add = cfg.add

        for X in NTs:
            for x in Sigma:
                w = rw(R) if rand else one
                add(w, X, x)

        for X in NTs:
            for Y in NTs:
                if Y == S:
                    continue
                w = rw(R) if rand else one
                add(w, X, Y)
                for Z in NTs:
                    if Z == S:
                        continue
                    w = rw(R) if rand else one
                    add(w, X, Y, Z)

        cfg.make_unary_fsa()
        return cfg


    def example2(self, R, rand=False):

        cfg = CFG(R=R)
        return cfg


    def example_test_abney(self, R):
        cfg = CFG(R=R)

        one = R.one

        X, Y, Z, U = NT("X"), NT("Y"), NT("Z"), NT("U")
        a, b, c, d, e = Sym("a"), Sym("b"), Sym("c"), Sym("d"), Sym("e")

        cfg.add(one, S, a, X, Y, Z)
        cfg.add(one, X, b)
        cfg.add(one, Y, c, U)
        cfg.add(one, U, d)
        cfg.add(one, Z, e)

        return cfg


if __name__ == '__main__':
    cfg = CFGExamples().example_test_abney(Boolean)
    cfg.to_locally_normalized_bottom_up_pda_gnf()