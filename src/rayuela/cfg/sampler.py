from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import trange

from rayuela.base.semiring import Real
from rayuela.base.string import String
from rayuela.base.symbol import EOS, Sym, ε
from rayuela.cfg.cfg import CFG
from rayuela.cfg.nonterminal import NT, S
from rayuela.cfg.parser import Parser
from rayuela.cfg.production import Production


class AutoregressiveSampler:
    """This class implements the autoregressive sampling algorithm for CFG's.
    Due to the use of the LRI parser, requires the CFG to be in CNF.
    """

    def __init__(
        self, G: CFG, normalize: bool = True, T: float = 1, seed: Optional[int] = None
    ):
        """

        Args:
            G (CFG): The CFG to sample from.
            normalize (bool, optional): Whether to normalize the CFG before sampling.
                                   Defaults to True.
            T (float, optional): The temperature to use for the sampling.
                                   Defaults to 1.
        """
        assert G.R == Real
        assert G.in_cnf()
        self.normalize = normalize if not G.is_locally_normalized else False
        self.T = T
        self.G = G.locally_normalize() if self.normalize else G
        self.parser = Parser(self.G)
        self.rng = np.random.default_rng(seed)

    def _draw(self, options: Dict[Sym, float]) -> Sym:
        p = np.asarray(list(float(w) for w in options.values()))
        choices = np.asarray(list(options.keys()))
        # print(f"p = {p}")
        return choices[self.rng.choice(len(choices), p=p)]

    def sample(
        self,
        K: int = 1,
        to_string: bool = False,
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda x: x,
    ) -> List[Union[Sequence[Sym], str]]:
        """Synonym for ancestral(K, to_string)"""
        return self.ancestral(K=K, to_string=to_string, transform=transform)

    def ancestral(
        self,
        K: int = 1,
        to_string: bool = False,
        transform: Callable[
            [Tuple[Union[str, String], Dict[Sym, float]]], Dict[Sym, float]
        ] = lambda x: x,
    ) -> List[Union[Sequence[Sym], str]]:
        """Generates K samples from the FSA.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.
            to_string (bool, optional): Whether to return the samples as strings or
                as sequences of symbols. Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.

        Returns:
            List[Sequence[Sym]]: The list of the K samples, where each sample is a
                sequence of symbols in the string.
        """
        return [self._ancestral(to_string, transform) for _ in trange(K)]

    def _w(self, w: float) -> float:
        return w ** (1 / self.T)

    def _get_probabilities(self, y: String) -> Dict[Sym, Real]:
        π = (
            float(self.parser.lri(y, strategy="fast")[(S, 0, len(y))])
            if y.y != [ε]
            else 1
        )

        p = dict()
        for a in self.G.Sigma:
            p[a] = (
                float(self.parser.lri(y + a, strategy="fast")[(S, 0, len(y + a))]) / π
            )

        p[EOS] = float(self.parser.cky(y)) / π

        p = {a: self._w(w) for (a, w) in p.items()}

        Z = sum(list(p.values()))

        p = {a: w / Z for (a, w) in p.items()}

        return p

    def _ancestral(
        self,
        to_string: bool,
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda x: x,
    ) -> List[Union[Sequence[Sym], str]]:
        y = String([ε])
        a = None

        while a != EOS:
            # print(f'y = {"".join([str(s) for s in y])}')
            p = transform(y, self._get_probabilities(y))
            a = self._draw(p)
            # print("-----------------------")
            y += a

        return y if not to_string else "".join([str(s) for s in y])


Derivation = Sequence[Union[Sym, NT]]


class ProductionSampler:
    """This class implements the sampling algorithm for CFG's in which the individual
    productions in the derivation of a string are sampled.
    """

    def __init__(
        self, G: CFG, normalize: bool = True, T: float = 1, seed: Optional[int] = None
    ):
        """

        Args:
            G (CFG): The CFG to sample from.
            normalize (bool, optional): Whether to normalize the CFG before sampling.
                                   Defaults to True.
            T (float, optional): The temperature to use for the sampling.
                                   Defaults to 1.
        """
        assert G.R == Real
        self.normalize = normalize if not G.is_locally_normalized else False
        self.T = T
        self.G = G.locally_normalize() if self.normalize else G
        self.rng = np.random.default_rng(seed)

    def _draw(self, options: Dict[Production, float]) -> Derivation:
        p = np.asarray(list(float(w) for w in options.values()))
        choices = list(options.keys())
        print(f"p = {p}")
        return list(choices[self.rng.choice(len(choices), p=p)].body)

    def sample(
        self,
        K: int = 1,
        to_string: bool = False,
        transform: Callable[
            [Tuple[Union[str, String], Dict[Sym, float]]], Dict[Sym, float]
        ] = lambda x: x,
        max_derivation_length: Optional[int] = None,
    ) -> List[Union[Sequence[Sym], str]]:
        """Synonym for ancestral"""
        return self.ancestral(
            K=K,
            to_string=to_string,
            transform=transform,
            max_derivation_length=max_derivation_length,
        )

    def ancestral(
        self,
        K: int = 1,
        to_string: bool = False,
        transform: Callable[
            [Tuple[Union[str, String], Dict[Sym, float]]], Dict[Sym, float]
        ] = lambda x: x,
        max_derivation_length: Optional[int] = None,
    ) -> List[Union[Sequence[Sym], str]]:
        """Generates K samples from the FSA.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.
            to_string (bool, optional): Whether to return the samples as strings or
                as sequences of symbols. Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.
            max_derivation_length (Optional[int], optional): The maximum length of the
                derivation.

        Returns:
            List[Sequence[Sym]]: The list of the K samples, where each sample is a
                sequence of symbols in the string.
        """
        ys = []
        for _ in range(K):
            y = self._ancestral(to_string, transform, max_derivation_length)
            if y is not None:
                ys.append(y)
        return ys

    def _w(self, w: Real) -> float:
        return float(w) ** (1 / self.T)

    def _get_probabilities(self, X: NT) -> Dict[Sym, Real]:
        p = dict()
        for P, w in self.G.P_byhead(X):
            p[P] = self._w(w)

        Z = sum(list(p.values()))

        p = {a: w / Z for (a, w) in p.items()}

        return p

    def _ancestral(
        self,
        to_string: bool,
        transform: Callable[
            [Tuple[Union[str, String], Dict[Sym, float]]], Dict[Sym, float]
        ] = lambda x: x,
        max_derivation_length: Optional[int] = None,
    ) -> List[Union[Sequence[Sym], str]]:
        d = [S]

        while any(isinstance(Y, NT) for Y in d) and (
            max_derivation_length is None or len(d) < max_derivation_length
        ):
            print(f"max_derivation_length = {max_derivation_length}")
            print(f'd = {"".join([str(s) for s in d])}, len(d) = {len(d)}')
            ix = [ii for (ii, Y) in enumerate(d) if isinstance(Y, NT)][0]
            X = d[ix]
            # p = transform(y, self._get_probabilities(y))
            p = self._get_probabilities(X)
            print(f"p = {p}")
            Xs = self._draw(p)
            d = d[:ix] + Xs + d[ix + 1 :]
            print(f'd` = {"".join([str(s) for s in d])}, len(d`) = {len(d)}')
            print("-----------------------")

        if any(isinstance(Y, NT) for Y in d):
            return None
        else:
            return String(d) if not to_string else "".join([str(s) for s in d])
