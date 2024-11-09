import dataclasses
import math
import random

import torch

from rayuela.base.semiring import Tropical as RayuelaTropical
from rayuela.base.symbol import Sym as RayuelaSym
from rayuela.base.state import State as RayuelaState
from rayuela.fsa.fsa import FSA

from recognizers.automata.automaton import Symbol
from recognizers.string_sampling.finite_automaton_weight_pushing import (
    NormalizedCountingFiniteAutomaton
)
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    String,
    Parse,
    ValidNextSymbolSet,
    ValidNextSymbolList,
    EmptyLanguageError
)
from recognizers.string_sampling.edit_distance import compute_edit_distance

@dataclasses.dataclass
class CacheEntry:
    label: bool
    edit_distance: int | None = None

def to_tropical_rayuela_fsa(automaton):
    result = FSA(R=RayuelaTropical)
    result.set_I(RayuelaState(automaton.initial_state))
    for (q, a), r in automaton.transitions.items():
        result.set_arc(RayuelaState(q), RayuelaSym(a), RayuelaState(r))
    for q in automaton.accept_states:
        result.set_F(RayuelaState(q))
    return result

class FiniteAutomatonLanguage(WeightedLanguage):

    def __init__(self,
        automaton: NormalizedCountingFiniteAutomaton,
        alphabet: list[str] | None,
        dtype: torch.dtype,
        device: torch.device
    ):
        super().__init__()
        self.cache = {}
        self.automaton = automaton
        self.tropical_rayuela_fsa = to_tropical_rayuela_fsa(automaton)
        self.dtype = dtype
        self.device = device
        if alphabet is not None:
            self._symbol_to_str = lambda s: alphabet[s]
        else:
            self._symbol_to_str = str

    def alphabet_size(self) -> int:
        return self.automaton.alphabet_size

    def symbol_to_str(self, symbol: Symbol) -> str:
        return self._symbol_to_str(symbol)

    def with_length_range(self,
        length_range: tuple[int, int]
    ) -> LengthRestrictedWeightedLanguage:
        return LengthRestrictedFiniteAutomatonLanguage(self, length_range)

    def sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolSet | None]:
        s, log_probability, next_symbols = self.automaton.sample(
            length,
            generator,
            include_log_probability,
            include_next_symbols
        )
        cache_entry = self.cache.get(s)
        if cache_entry is None:
            cache_entry = self.cache[s] = CacheEntry(label=True)
        return s, log_probability, next_symbols

    def is_negative(self,
        s: String,
        include_edit_distance: bool
    ) -> tuple[bool, int | None]:
        cache_entry = self.cache.get(s)
        if cache_entry is None:
            label = self.uncached_label(s)
            cache_entry = self.cache[s] = CacheEntry(label=label)
        if include_edit_distance and cache_entry.edit_distance is None:
            if cache_entry.label:
                cache_entry.edit_distance = 0
            else:
                cache_entry.edit_distance = self.uncached_edit_distance(s, self.dtype, self.device)
        return not cache_entry.label, cache_entry.edit_distance

    def uncached_label(self, s: String) -> bool:
        return self.automaton.accepts(s)

    def uncached_edit_distance(self,
        s: String,
        dtype: torch.dtype,
        device: torch.device
    ) -> int:
        return compute_edit_distance(self.tropical_rayuela_fsa, s, dtype, device)

class LengthRestrictedFiniteAutomatonLanguage(LengthRestrictedWeightedLanguage):

    def __init__(self, parent: FiniteAutomatonLanguage, length_range: tuple[int, int]):
        super().__init__()
        self.parent = parent
        self.min_length, self.max_length = length_range
        if self.max_length > self.parent.automaton.max_length:
            raise ValueError(
                f'the prepared automaton is prepared for sampling strings up '
                f'to length {self.parent.automaton.max_length}, but '
                f'{self.max_length} is required'
            )
        # Figure out which lengths are possible within this length range.
        self.valid_lengths = parent.automaton.valid_lengths(length_range)
        if not self.valid_lengths:
            raise EmptyLanguageError(
                f'no lengths are valid within the length range {length_range}'
            )
        # Precompute the log probability of selecting a length.
        self.log_num_lengths = math.log(len(self.valid_lengths))

    def supports_log_probability(self) -> bool:
        return True

    def supports_next_symbols(self) -> bool:
        return True

    def supports_edit_distance(self) -> bool:
        return True

    def sample(self,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, Parse]:
        length = self.sample_length(generator)
        s, log_probability, next_symbols = self.parent.sample(
            length,
            generator,
            include_log_probability,
            include_next_symbols
        )
        if log_probability is not None:
            # Renormalize the probability according to the length selected.
            log_probability = (
                log_probability
                - self.log_num_lengths
                - self.parent.automaton.total_length_weight(length)
            )
        return s, Parse(log_probability, next_symbols)

    def sample_length(self, generator):
        return generator.choice(self.valid_lengths)

    def is_negative(self,
        s: String,
        include_edit_distance: bool
    ) -> tuple[bool, int | None]:
        if not (self.min_length <= len(s) <= self.max_length):
            raise ValueError
        return self.parent.is_negative(s, include_edit_distance)
