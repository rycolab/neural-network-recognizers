from itertools import product
from typing import List, Optional, Tuple, Type, Union

import numpy as np

from rayuela.base.alphabet import Alphabet
from rayuela.base.misc import _random_weight as rw
from rayuela.base.semiring import Real, Semiring
from rayuela.base.symbol import Expr, Sym, ε, φ
from rayuela.fsa.fsa import FSA, State
from rayuela.fsa.fst import FST


def _add_transition(
    i: int,
    a: Union[Sym, Tuple[Sym, Sym]],
    j: int,
    used_symbols: List[Union[Sym, Tuple[Sym, Sym]]],
    A: Union[FSA, FST],
    bias: float = 0.25,
    acyclic: bool = False,
    deterministic: bool = True,
    rng=None,
    **kwargs,
) -> None:
    """Adds a transition to the random machine."""

    if rng is None:
        rng = np.random.default_rng()

    if (deterministic or a == φ) and a in used_symbols:
        # always add at most one failure arc
        # or at most of any symbol if machine should be deterministic
        return False

    bias = bias if a != φ else kwargs.get("phi_bias", bias)

    if rng.random() < bias:
        w = rw(A.R, rng=rng, **kwargs) if a != φ else A.R.one
        if acyclic or a == φ:
            # Make sure that the failure arcs *always* form an acyclic subgraph
            if i < j:
                if isinstance(A, FST):
                    assert isinstance(a, tuple)
                    A.add_arc(State(i), a[0], a[1], State(j), w)
                else:
                    assert isinstance(a, Sym) or isinstance(a, Expr)
                    A.add_arc(State(i), a, State(j), w)
                used_symbols.append(a)
                return True
            else:
                return False
        else:
            if isinstance(A, FST):
                assert isinstance(a, tuple)
                A.add_arc(State(i), a[0], a[1], State(j), w)
            else:
                assert isinstance(a, Sym) or isinstance(a, Expr)
                A.add_arc(State(i), a, State(j), w)
            used_symbols.append(a)
            return True
    else:
        return False


def _add_initial_and_final(
    fsa: Union[FSA, FST], num_initial: int, num_final: int, rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    if len(fsa.Q) > 0:
        Is = rng.choice(list(fsa.Q), num_initial)
        for q in Is:
            fsa.set_I(q, rw(fsa.R, rng=rng, divide_by=num_initial))
        Fs = rng.choice(list(fsa.Q), num_final)
        for q in Fs:
            fsa.set_F(q, rw(fsa.R, rng=rng, divide_by=num_final))


def _add_pfsa_arcs(
    num_states: int,
    alphabet: Alphabet,
    i: int,
    A: FSA,
    bias: float,
    deterministic: bool,
    rng=None,
) -> None:
    if rng is None:
        rng = np.random.default_rng()
    if deterministic:
        out_neighbourhood_size = min(
            sum([int(rng.random() < bias) for _ in range(num_states)]), len(alphabet)
        )
        α = [rng.random() for _ in range(out_neighbourhood_size)]
        α = [t / sum(α) for t in α]
        alphabet = list(alphabet)
        rng.shuffle(alphabet)
        S = rng.choice(range(num_states), out_neighbourhood_size)
        for ii, j in enumerate(S):
            A.add_arc(State(i), alphabet[ii], State(j), Real(α[ii]))
    else:
        out_neighbourhood_size = sum(
            [int(rng.random() < bias) for _ in range(num_states * len(alphabet))]
        )
        α = [rng.random() for _ in range(out_neighbourhood_size)]
        α = [t / sum(α) for t in α]
        S = rng.choice(
            list(product(range(num_states), alphabet)), out_neighbourhood_size
        )
        for ii, (j, a) in enumerate(S):
            A.add_arc(State(i), a, State(j), Real(α[ii]))


def _add_fsa_arcs(
    num_states: int,
    alphabet: Alphabet,
    i: int,
    used_symbols: List[Union[Sym, Tuple[Sym, Sym]]],
    A: Union[FSA, FST],
    bias: float,
    acyclic: bool,
    deterministic: bool,
    rng=None,
    **kwargs,
) -> None:
    if rng is None:
        rng = np.random.default_rng()
    for j in rng.choice(range(num_states), num_states):
        for a in alphabet:
            _add_transition(
                i=i,
                a=a,
                j=j,
                used_symbols=used_symbols,
                A=A,
                bias=bias,
                acyclic=acyclic,
                deterministic=deterministic,
                rng=rng,
                **kwargs,
            )


def _random_machine(
    Σ: Alphabet,
    R: Type[Semiring],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = False,
    no_phi: bool = True,
    acyclic: bool = False,
    deterministic: bool = True,
    num_initial: int = 1,
    num_final: int = 1,
    fst: bool = False,
    probabilistic: bool = False,
    rng=None,
    **kwargs,
) -> Union[FSA, FST]:
    if rng is None:
        rng = np.random.default_rng()

    if fst:
        fsa = FST(R=R)
    else:
        fsa = FSA(R=R)

    if not no_eps:
        Σ.add(ε)
    else:
        Σ.discard(ε)

    if not no_phi:
        Σ.add(φ)
    else:
        Σ.discard(φ)

    if fst:
        alphabet = [(a, b) for a in Σ for b in kwargs.get("Delta", Σ)]
    else:
        alphabet = Σ

    if deterministic:
        assert num_initial == 1

    for i in range(num_states):
        used_symbols = []
        if probabilistic:
            _add_pfsa_arcs(
                num_states=num_states,
                alphabet=alphabet,
                i=i,
                A=fsa,
                bias=bias,
                deterministic=deterministic,
                rng=rng,
            )
        else:
            _add_fsa_arcs(
                num_states=num_states,
                alphabet=alphabet,
                i=i,
                used_symbols=used_symbols,
                A=fsa,
                bias=bias,
                acyclic=acyclic,
                deterministic=deterministic,
                rng=rng,
                **kwargs,
            )

    _add_initial_and_final(fsa, num_initial, num_final, rng=rng)

    return fsa


def _reweigh(fsa: Union[FSA, FST]):
    """Reweights the automaton to form a probability one."""
    Z = fsa.R.zero
    for q, w in fsa.I:
        Z += w

    for q, w in fsa.I:
        fsa.set_I(q, w / Z)

    for q, wf in fsa.F:
        Z = wf
        for a, j, w in fsa.arcs(q):
            Z += w
        for a, j, w in fsa.arcs(q):
            fsa.set_arc(q, a, j, w / Z)
        fsa.set_F(q, wf / Z)


def random_pfsa(
    Sigma: Union[Alphabet, str],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = True,
    acyclic: bool = False,
    deterministic: bool = True,
    minimal: bool = False,
    num_initial: int = 1,
    num_final: int = 1,
    seed: Optional[int] = None,
    **kwargs,
) -> FSA:
    """Generates a random probabilistic FSA.

    Args:
        Sigma (Union[Alphabet, str]): The alphabet of the FSA.
        num_states (int): The number of states of the FSA.
        bias (float, optional): The average connectivity of the FSA. Defaults to 0.25.
        no_eps (bool, optional): Whether to have epsilon transitions. Defaults to True.
        acyclic (bool, optional): Acyclic or not. Defaults to False.
        deterministic (bool, optional): Deterministic or not. Defaults to True.
        minimal (bool, optional): Whether to minimize the deterministic PFSA.
            WARNING: Unless you're constructing a deterministic machine, this will
            most likely not converge since most random machines are non-determinizable.
            If you want a minimal deterministic machine, set deterministic to True.
            Defaults to False.
        num_initial (int, optional): Number of initial states. Defaults to 1.
        num_final (int, optional): Number of final states. Defaults to 1.
        seed (Optional[int], optional): The seed of the random generator.
            Defaults to None.

    Returns:
        FSA: The random probabilistic FSA.
    """
    return random_machine(
        Sigma=Sigma,
        R=Real,
        num_states=num_states,
        bias=bias,
        no_eps=no_eps,
        acyclic=acyclic,
        deterministic=deterministic,
        minimal=minimal,
        probabilistic=True,
        num_initial=num_initial,
        num_final=num_final,
        seed=seed,
        **kwargs,
    )


def random_machine(
    Sigma: Union[Alphabet, str],
    R: Type[Semiring],
    num_states: int,
    bias: float = 0.25,
    no_eps: bool = False,
    no_phi: bool = True,
    acyclic: bool = False,
    deterministic: bool = True,
    minimal: bool = False,
    trimmed: bool = True,
    pushed: bool = False,
    num_initial: int = 1,
    num_final: int = 1,
    probabilistic: bool = False,
    fst: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> Union[FSA, FST]:
    """
    Creates a random WFSA or WFST.
    It takes a number of parameters that control the properties of the machine.

    Args:
        Sigma (Alphabet): The alphabet of the WFSA.
        R (Type[Semiring]): The semiring of the WFSA.
        num_states (int): The number of states of the WFSA.
        bias (float, optional): The probability of realising an edge between
            any pair of states (q, q') with a specific symbol. Defaults to 0.25.
        no_eps (bool, optional): When true, the WFSA contains no ε transitions.
            Defaults to False.
        no_phi (bool, optional): When true, the WFSA contains no φ transitions.
            Defaults to True.
        eigen (bool, optional): _description_. Defaults to False.
        acyclic (bool, optional): When true, the WFSA will be acyclic by design.
            Defaults to False.
        deterministic (bool, optional): When true, the WFSA will be deterministic.
            Defaults to True.
        minimal (bool, optional): When true, minimises the determinized machine.
            WARNING: Unless you're constructing a deterministic machine, this will
            most likely not converge since most random machines are non-determinizable.
            If you want a minimal deterministic machine, set deterministic to True.
            Defaults to False.
        trimmed (bool, optional): When true, trims the machine to make it smaller.
            Defaults to True.
        pushed (bool, optional): When true, pushes the machine to make it locally
            normalised. Defaults to False.
        num_initial: The number of initial states. Each will be assigned a random
            initial weight. Defaults to 1.
        num_final: The number of final states. Each will be assigned a random
            final weight. Defaults to 1.
        probabilistic: Whether to generate a probabilistic WFSA. Requires the real
            semiring. If true, the WFSA will be pushed and the initial states will
            be reweighted such that the initial weights form a probability distribution.
        fst (bool, optional): Whether to create a random _transducer_.
            Defaults to False.
        seed (int, optional): The seed for the random number generator.
        kwargs: Arguments for random weight generation

    Returns:
        Union[FSA, FST]: A random WFSA of WFST.
    """
    assert R is Real or not probabilistic

    if isinstance(Sigma, str):
        Sigma = Alphabet(Sigma)

    rng = np.random.default_rng(seed)
    # random.seed(seed)

    fsa = None
    while fsa is None or not fsa.num_states:
        fsa = _random_machine(
            Sigma,
            R,
            num_states,
            bias=bias,
            no_eps=no_eps,
            no_phi=no_phi,
            acyclic=acyclic,
            deterministic=deterministic,
            num_initial=num_initial,
            num_final=num_final,
            fst=fst,
            probabilistic=probabilistic,
            rng=rng,
            **kwargs,
        )

        # Trim the machine to make it smaller
        if trimmed:
            fsa = fsa.trim()

    if pushed:
        fsa = fsa.push()

    if minimal:
        fsa = fsa.determinize().minimize()

    if probabilistic:
        _reweigh(fsa)

    return fsa


def random_ngram(
    Sigma: Union[Alphabet, str], n: int, seed: Optional[int] = None, **kwargs
) -> FSA:
    from rayuela.base.symbol import BOS

    if isinstance(Sigma, str):
        Sigma = Alphabet(Sigma).symbols

    rng = np.random.default_rng(seed)

    fsa = FSA(R=Real)

    qI = State((BOS,) * (n - 1))
    fsa.set_I(qI, Real(1))
    α = [rng.random() for _ in range(len(Sigma) + 1)]
    α = [t / sum(α) for t in α]
    for ii, a in enumerate(Sigma):
        fsa.add_arc(qI, a, State((BOS,) * (n - 2) + (a,)), Real(α[ii]))
    fsa.add_F(qI, Real(α[-1]))

    for ll in range(n - 2, 0, -1):
        for ngr in product(Sigma, repeat=ll):
            ngr = (BOS,) * (n - ll - 1) + ngr
            q = State(ngr)
            α = [rng.random() for _ in range(len(Sigma) + 1)]
            α = [t / sum(α) for t in α]
            for ii, a in enumerate(Sigma):
                fsa.add_arc(q, a, State(ngr[1:] + (a,)), Real(α[ii]))
            fsa.add_F(q, Real(α[-1]))

    # Full states
    for ngr in product(Sigma, repeat=n - 1):
        q = State(ngr)
        α = [rng.random() for _ in range(len(Sigma) + 1)]
        α = [t / sum(α) for t in α]
        for ii, a in enumerate(Sigma):
            fsa.add_arc(q, a, State(ngr[1:] + (a,)), Real(α[ii]))
        fsa.add_F(q, Real(α[-1]))

    return fsa


def random_trellis(Sigma, R, num_states=2, last=3) -> FSA:  # noqa: C901
    """
    Generates a random trellis.
    Useful for testing Viterbi.
    """
    fsa = FSA(R=R)

    # beginning
    for n in range(num_states):
        for a in Sigma:
            if a == ε:
                continue
            # fsa.add_arc(State("init"), a, State((1, n)), rw(R, rng=rng))
            fsa.add_arc(State("init"), a, State((1, n)), R.one)

    fsa.set_I(State("init"), w=R.one)

    # middle
    for t in range(2, last + 1):
        for i in range(num_states):
            for j in range(num_states):
                for a in Sigma:
                    if a == ε:
                        continue
                    # fsa.add_arc(State(((t - 1), i)), a, State((t, j)), rw(R, rng=rng))
                    fsa.add_arc(State(((t - 1), i)), a, State((t, j)), R.one)

    # end
    for n in range(num_states):
        for a in Sigma:
            if a == ε:
                continue
            # fsa.add_arc(State((last, n)), a, State("end"), rw(R, rng=rng))
            fsa.add_arc(State((last, n)), a, State("end"), R.one)

    # add final state
    # fsa.set_F(State("end"), rw(R, rng=rng))
    fsa.set_F(State("end"), R.one)

    return fsa
