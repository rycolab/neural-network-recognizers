import random
from itertools import product
from typing import List, Optional, Sequence, Set, Type, Union

from rayuela.base.alphabet import Alphabet
from rayuela.base.misc import _random_weight as rw
from rayuela.base.semiring import Semiring
from rayuela.base.symbol import ε
from rayuela.cfg.cfg import CFG
from rayuela.cfg.nonterminal import NT, S
from rayuela.cfg.transformer import Transformer


def random_cfg(  # noqa: C901
    Sigma: Alphabet,
    V: Union[Sequence[NT], Set[NT]],
    R: Type[Semiring],
    bias: float = 0.25,
    nonterminal_only: bool = False,
    body_length: int = 2,
    productions_per_nonterminal: int = 5,
    locally_normalized: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> CFG:
    random.seed(seed)

    cfg = CFG(R=R)

    if nonterminal_only:
        available = list(V)
    else:
        available = list(Sigma.union(V).difference({S}))

    for X in V:
        for _ in range(productions_per_nonterminal):
            body = []

            # Generate X → Y Z, X → x, or X → ε (old)
            for _ in range(random.randint(2, body_length)):
                # sym = random.choice(list(Sigma.union(V).difference({S})))
                sym = random.choice(available)

                if body and sym == ε:
                    # If there's already a symbol and the new symbol is ε, break
                    break

                body.append(sym)

                if sym == ε:
                    break

            cfg.add(rw(R, **kwargs), X, *body)

    for X in V:
        for a in Sigma:
            cfg.add(rw(R, **kwargs), X, a)

    cfg.make_unary_fsa()

    if locally_normalized:
        cfg = cfg.locally_normalize()

    return cfg


def random_cfg_cnf(  # noqa: C901
    Sigma: Alphabet,
    V: Union[Sequence[NT], Set[NT]],
    R: Type[Semiring],
    bias: float = 0.5,
    seed: Optional[int] = None,
    locally_normalized: bool = False,
    **kwargs,
) -> CFG:
    random.seed(seed)

    cfg = CFG(R=R)

    if isinstance(V, Sequence):
        V = set(V)

    V.add(S)
    nts = set(V) - {S}
    s_count = 0
    for X, Y in product(V, nts):
        for Z in nts:
            if random.random() < bias:
                cfg.add(rw(R, **kwargs), X, Y, Z)
                if X == S:
                    s_count += 1

    if s_count == 0:
        # ensure grammar has at least 1 production rule with S as LHS
        A = random.choice(tuple(nts))
        B = random.choice(tuple(nts))
        cfg.add(rw(R, **kwargs), S, A, B)
        bias = 0.1
        for Y in nts:
            for Z in nts:
                if random.random() < bias:
                    cfg.add(rw(R, **kwargs), S, Y, Z)

    # for X, Y in product(V, V.difference(set([S]))):
    #    cfg.add(rw(R, **kwargs), X, Y)

    for X in V:
        for a in Sigma:  # .#union([ε]):
            if not (a == ε and X != S):
                cfg.add(rw(R, **kwargs), X, a)

    cfg.make_unary_fsa()

    cfg = cfg.trim()

    if locally_normalized:
        cfg = cfg.locally_normalize()

    return cfg


def add_unaries(cfg: CFG, bias: float = 0.5, **kwargs) -> CFG:
    nts = set(cfg.V) - {S}
    A = random.choice(tuple(nts))
    B = A
    if len(nts) > 1:
        while B == A:
            B = random.choice(tuple(nts))
    cfg.add(rw(cfg.R, **kwargs), A, B)
    for X in nts:
        for Y in nts:
            if X != Y:
                if random.random() < bias:
                    cfg.add(rw(cfg.R, **kwargs), X, Y)
    return cfg


def add_nullaries(cfg: CFG, **kwargs) -> CFG:
    for X in cfg.V:
        cfg.add(rw(cfg.R, **kwargs), X, ε)
    return cfg


def random_palindrome_cfg(
    Sigma: Alphabet,
    R: Type[Semiring],
    seed: Optional[int] = None,
    to_cnf: bool = False,
    locally_normalized: bool = False,
    set_weights: Optional[List[float]] = None,
    **kwargs,
) -> CFG:
    random.seed(seed)

    G = CFG(R=R)

    A = NT("A")

    # Old
    # G.add(rw(R, **kwargs), S, ε)
    # G.add(rw(R, **kwargs), S, A)

    # for a in Sigma:
    #     G.add(rw(R, **kwargs), A, a, A, a)
    #     G.add(rw(R, **kwargs), A, a)

    # if to_cnf:
    #     G = Transformer().cnf(G)

    # if locally_normalized:
    #     G = G.locally_normalize()

    ws = (
        [rw(R, **kwargs), rw(R, **kwargs)]
        if set_weights is None
        else [R(w) for w in set_weights[:2]]
    )
    if locally_normalized:
        ws = [w / sum(ws, R.zero) for w in ws]

    G.add(ws[0], S, ε)
    G.add(ws[1], S, A)

    ws = (
        [rw(R, **kwargs) for _ in range(2 * len(Sigma))]
        if set_weights is None
        else [R(w) for w in set_weights[2:]]
    )
    if locally_normalized:
        ws = [w / sum(ws, R.zero) for w in ws]

    for ii, a in enumerate(Sigma):
        G.add(ws[2 * ii], A, a, A, a)
        G.add(ws[2 * ii + 1], A, a)

    if to_cnf:
        G = Transformer().cnf(G)

    return G
