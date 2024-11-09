from itertools import permutations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm, trange

from rayuela.base.semiring import Real, Semiring
from rayuela.base.state import State
from rayuela.base.symbol import EOS, Sym, ε
from rayuela.fsa.fsa import FSA


class Sampler:
    """This class implements the ancestral sampling algorithm for FSA's.
    It works by sampling a path through the FSA and then returning the
    sequence of symbols that were traversed proportional to the weight
    of the transitions that were taken.
    """

    def __init__(self, fsa: FSA, T: float = 1, seed: Optional[int] = None):
        """

        Args:
            fsa (FSA): The FSA to sample from.
            T (float, optional): The temperature to use for the sampling.
                Defaults to 1.
            seed (Optional[int], optional): The seed to use for the random number
                generator. Defaults to None.
        """
        assert fsa.R == Real
        self.T = T
        self.A = fsa.push() if not fsa.probabilistic else fsa
        self.rng = np.random.default_rng(seed)

    def _draw(
        self, options: Dict[Union[Sym, State], float]
    ) -> Tuple[Union[Sym, State], Semiring]:
        p = np.asarray(list(float(w) for w in options.values()))
        return list(options.items())[self.rng.choice(len(p), p=p)]

    def _draw_unif(self, options):
        p = list(w for w in options.values())
        Z = len([prob for prob in p if prob != self.A.R.zero])
        # keep 0 probabilities, else uniform
        p = np.asarray([1 / Z if prob != self.A.R.zero else 0 for prob in p])
        choices = np.asarray(list(options.keys()))
        return choices[self.rng.choice(len(choices), p=p)]

    def sample_negative(
        self,
        K: int = 1,
        to_string: bool = True,
        sep: str = " ",
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda _, p: p,
        lm: bool = False,
        eps_free: bool = False,
        max: int = 10000,
    ) -> Sequence[Union[Sequence[Sym], str]]:
        """Generates K samples of strings not from the PFSA.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.
            to_string (bool, optional): Whether to return the samples as strings
                (alternative: as sequences of symbols). Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling if using
                lm=True.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.
            lm (bool, optional): Whether to sample using the language model
                approach. Defaults to False.
            eps_free (bool, optional): A flag that indicates whether the FSA is
                epsilon-free. If yes, it allows for faster sampling. Defaults to False.
            max: the max number of iterations to wait for a negative dataset
                - prevents an endless loop

        Returns:
            Sequence[Sequence[Sym]]: The sequence of the samples, where each sample is a
                sequence of symbols in the string.
        """

        # check for impossible FSAs to permute
        if len(self.A.Sigma) <= 1:
            raise Exception("Can't Peturb FSA's with an alphabet of size 1")

        negative_samples = []
        cnt = 0

        pbar = tqdm(total=K)
        while len(negative_samples) < K:
            if lm:
                fsa_str = self._ancestral_lm(to_string, sep, transform)
            else:
                fsa_str = self._ancestral(to_string, sep)

            # can't peturb string of size <= 1 or one with all same characters.
            if len(set(fsa_str)) > 1:
                candidate = self.gen_negative(fsa_str, eps_free=eps_free)
                # check if candidate possible
                if candidate:
                    negative_samples.append(candidate)
                    pbar.update(1)

            # prevent endless sampling loop
            cnt += 1
            if cnt == max:
                pbar.close()
                print("WARNING: Max iterations reached, returning what we have")
                return negative_samples

        pbar.close()

        return negative_samples

    def gen_negative(self, fsa_str, eps_free: bool = False):
        """
        Itterates through a unique peturbed candidates,
        return the first negative sample.
        Args:
            fsa_str: a sampled string from the fsa

        Returns:
            fsa_str with at least one pair of characters swapped
        """

        for c in permutations(fsa_str):
            candidate = "".join(c)
            if self.A.accept(candidate, eps_free=eps_free) == self.A.R.zero:
                return candidate

        return False

    def sample(
        self,
        K: int = 1,
        to_string: bool = True,
        sep: str = " ",
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda _, p: p,
        lm: bool = False,
        return_weights: bool = False,
    ) -> Sequence[Union[Sequence[Sym], str]]:
        """Generates K samples of strings from the PFSA.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.
            to_string (bool, optional): Whether to return the samples as strings
                (alternative: as sequences of symbols). Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling if using
                lm=True.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.
            lm (bool, optional): Whether to sample using the language model
                approach. Defaults to False.

        Returns:
            Sequence[Sequence[Sym]]: The sequence of the samples, where each sample is a
                sequence of symbols in the string. When distinct is True, it outputs a
                set of at most K unique samples, otherwise a list of non-unique samples.
        """
        if lm:
            return [self._ancestral_lm(to_string, sep, transform) for _ in trange(K)]
        else:
            if self.T == 1:
                return [
                    self._ancestral(to_string, sep, return_weights) for _ in trange(K)
                ]
            else:
                return [self._temperature_ancestral(to_string, sep) for _ in trange(K)]

    def _w(self, w: Real) -> float:
        return float(w) ** (1 / self.T)

    def _ancestral(self, to_string: bool, sep: str = " ", return_weights: bool = False):
        y = []
        q, path_weight = self._draw({p: w for p, w in self.A.I})
        path_weight = np.log(path_weight.value)

        while q != 0:
            options = {(a, qʼ): w for a, qʼ, w in self.A.arcs(q)}
            if self.A.ρ[q] != self.A.R.zero:
                options[(EOS, 0)] = self.A.ρ[q]  # Signals the end of generation

            (a, q), w = self._draw(options)
            path_weight += np.log(w.value)
            if a != EOS:
                y.append(a.value)

        if to_string:
            y = sep.join(y)

        if return_weights:
            return y, path_weight

        return y

    def _temperature_ancestral(self, to_string: bool, sep: str = " "):
        y = []
        Z = sum([self._w(w) for _, w in self.A.I])
        q, _ = self._draw({p: self._w(w) / Z for p, w in self.A.I})

        while q != 0:
            Z = sum([self._w(w) for _, _, w in self.A.arcs(q)]) + self._w(self.A.ρ[q])
            options = {(a, qʼ): self._w(w) / Z for a, qʼ, w in self.A.arcs(q)}
            options[(EOS, 0)] = (
                self._w(self.A.ρ[q]) / Z
            )  # Signals the end of generation

            (a, q), _ = self._draw(options)
            if a != EOS:
                y.append(a)

        return y if not to_string else sep.join([str(s) for s in y])

    def _ancestral_lm(
        self,
        to_string: bool,
        sep: str = " ",
        transform: Callable[[Dict[Sym, float]], Dict[Sym, float]] = lambda x: x,
    ) -> List[Union[Sequence[Sym], str]]:
        """Generates a sample from the FSA as an autoregressive language model.

        Args:
            to_string (bool, optional): Whether to return the sample as a string or
                as a sequence of symbols. Defaults to False.
            transform (Callable[[Dict[Sym, float]], Dict[Sym, float]], optional): A
                function to transform the probabilities before sampling.
                It can depend on the string generated so far. Defaults to the
                identity function of the probabilities.

        Returns:
            Union[Sequence[Sym], str]: The sample, either as a sequence of symbols
                or as a string.
        """
        # Choose the initial state to start sampling from
        Z = sum([self._w(w) for _, w in self.A.I])
        pq_a = {ε: {q: self._w(w) / Z for q, w in self.A.I}}  # For convenience

        # The generated string
        a = ε
        y = [a]

        while a != EOS:
            pq_a = pq_a[a]
            q, _ = self._draw(pq_a)

            Zq = sum([self._w(w) for _, _, w in self.A.arcs(q)]) + self._w(self.A.ρ[q])
            Zqa = {
                b: sum(self._w(w) for c, _, w in self.A.arcs(q) if c == b)
                for b in self.A.Sigma
            }

            # Marginal probabilities of next symbols given the current state
            pa = {
                b: sum(self._w(w) for c, _, w in self.A.arcs(q) if c == b) / Zq
                for b in self.A.Sigma
            }
            pa[EOS] = self._w(self.A.ρ[q]) / Zq

            # Conditional probabilities of states given the next symbol
            pq_a = {
                b: {
                    qʼ: sum(
                        self._w(w)
                        for c, qʼʼ, w in self.A.arcs(q)
                        if c == b and qʼ == qʼʼ
                    )
                    / Zqa[b]
                    for qʼ in self.A.Q
                }
                for b in self.A.Sigma
                if Zqa[b] > 0
            }

            a, _ = self._draw(transform(y, pa))
            y.append(a)

        return y if not to_string else sep.join([str(s) for s in y])

    def sample_all_next(
        self,
        K: int = 1,
        to_string: bool = True,
        sep: str = " ",
        return_weights: bool = False,
    ):
        """Generates K samples of strings from the PFSA, together with the set of
        all valid symbols at each step in the generation.

        Args:
            K (int, optional): The number of samples to generate. Defaults to 1.

        Returns:
            Sequence[Sequence[Sym]]: The sequence of the samples, where each sample is a
                tuple consisting of sequence of symbols in the string and a sequence of
                sets of valid symbols for each position.
        """
        assert self.A.probabilistic
        r = range if K == 1 else trange
        return [
            self._ancestral_all_next(
                to_string=to_string, sep=sep, return_weights=return_weights
            )
            for _ in r(K)
        ]

    def _ancestral_all_next(
        self, to_string: bool = True, sep: str = " ", return_weights: bool = False
    ):
        """
        Generates a string, together with the set of all valid symbols at each step
        in the generation.

        Returns:
            tuple[list[str]], list[list[str]]]: The list of symbols in the sampled
            string and the set of valid symbols at each position.
            The next symbol is drawn uniformly at random.
        """
        y = []
        y_all_valid = []
        q, path_weight = self._draw({p: w for p, w in self.A.I})
        path_weight = np.log(path_weight.value)

        while q != 0:
            options = {(a, qʼ): w for a, qʼ, w in self.A.arcs(q)}
            if self.A.ρ[q] != self.A.R.zero:
                options[(EOS, 0)] = self.A.ρ[q]  # Signals the end of generation

            next_syms = [a.value for (a, _), _ in options.items()]
            y_all_valid.append(next_syms if not to_string else sep.join(next_syms))

            (a, q), w = self._draw(options)
            path_weight += np.log(w.value)
            if a != EOS:
                y.append(a.value)

        if to_string:
            y = sep.join(y)

        if return_weights:
            return y, y_all_valid, path_weight

        return y, y_all_valid
