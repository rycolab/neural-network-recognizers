import unittest
import random

import torch
from numpy.ma.testutils import assert_equal

from rayuela.fsa.examples import EditDistanceExamples

from recognizers.string_sampling.finite_automaton_language import FiniteAutomatonLanguage
from recognizers.string_sampling.prepare_sampler import (
    push_finite_automaton_weights,
    lift_finite_automaton
)
from recognizers.string_sampling.sample_dataset import (
    generate_negative_example,
    generate_positive_example
)

from recognizers.hand_picked_languages.save_automaton import get_automaton


class TestEqRayuela(unittest.TestCase):

    def test_handpicked_languages(self) -> None:
        dtype = torch.float32
        device = torch.device("cpu")
        max_length = 500
        num_samples = 5
        generator = random.Random(123)

        language_names = [
            'repeat-01',
            'dyck-2-3'
        ]

        for name in language_names:
            automaton, alphabet = get_automaton(name)
            prepared_automaton = push_finite_automaton_weights(
                lift_finite_automaton(
                    automaton,
                    max_length,
                    dtype,
                    device
                ),
                dtype,
                device
            )

            language = FiniteAutomatonLanguage(prepared_automaton, alphabet, dtype, device)
            alphabet = [language.symbol_to_str(c) for c in range(language.alphabet_size())]
            alphabet_size = len(alphabet)
            lr_language = language.with_length_range((0, 40))

            for i in range(num_samples):
                s, edit_distance, _ = generate_negative_example(
                    lr_language,
                    (0, 40),
                    alphabet_size,
                    0.5,
                    False,
                    True,
                    generator,
                    None
                )
                edit_distance_rayuela = round(EditDistanceExamples.edit_distance(language.tropical_rayuela_fsa, s).value)
                assert_equal(edit_distance, edit_distance_rayuela)

                s, _ = generate_positive_example(
                    lr_language,
                    False,
                    False,
                    generator,
                    None
                )
                edit_distance = 0
                edit_distance_rayuela = round(EditDistanceExamples.edit_distance(language.tropical_rayuela_fsa, s).value)
                assert_equal(edit_distance, edit_distance_rayuela)


if __name__ == '__main__':
    unittest.main()
