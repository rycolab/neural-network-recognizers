import argparse
import contextlib
import enum
import math
import pathlib
import random
import shutil
import sys
from collections.abc import Set, MutableSet

import numpy
import torch

from rau.tools.torch.model_interface import parse_device
from recognizers.tools.jsonl import write_json_line
from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    LengthRestrictedWeightedLanguage,
    String,
    Parse,
    EmptyLanguageError
)
from recognizers.string_sampling.finite_automaton_weight_pushing import (
    NormalizedCountingFiniteAutomaton
)
from recognizers.string_sampling.finite_automaton_language import (
    FiniteAutomatonLanguage
)
from recognizers.hand_picked_languages.majority import Majority
from recognizers.hand_picked_languages.stack_manipulation import StackManipulation
from recognizers.hand_picked_languages.marked_reversal import MarkedReversal
from recognizers.hand_picked_languages.unmarked_reversal import UnmarkedReversal
from recognizers.hand_picked_languages.marked_copy import MarkedCopy
from recognizers.hand_picked_languages.missing_duplicate_string import MissingDuplicateString
from recognizers.hand_picked_languages.odds_first import OddsFirst
from recognizers.hand_picked_languages.binary_addition import BinaryAddition
from recognizers.hand_picked_languages.binary_multiplication import BinaryMultiplication
from recognizers.hand_picked_languages.compute_sqrt import ComputeSqrt
from recognizers.hand_picked_languages.bucket_sort import BucketSort

def generate_files(
    language: LengthRestrictedWeightedLanguage,
    length_range: tuple[int, int],
    alphabet: list[int],
    num_samples: int,
    only_negative: bool,
    perturbation_probability: float,
    strict_num_edits_distribution: bool,
    include_log_probability: bool,
    include_next_symbols: bool,
    include_edit_distance: bool,
    generator: random.Random,
    output_dir: pathlib.Path,
    generated_strings_output: MutableSet[String] | None,
    generated_strings_max_length: int | None,
    excluded_strings: Set[String] | None
) -> None:
    include_log_probability = include_log_probability and language.supports_log_probability()
    include_next_symbols = include_next_symbols and language.supports_next_symbols()
    include_edit_distance = include_edit_distance and language.supports_edit_distance()
    alphabet_size = len(alphabet)
    output_dir.mkdir(parents=True, exist_ok=True)
    strings_file = output_dir / 'main.tok'
    print(f'writing {strings_file}')
    labels_file = output_dir / 'labels.txt'
    print(f'writing {labels_file}')
    num_edits_file = output_dir / 'num-edits.txt'
    print(f'writing {num_edits_file}')
    with strings_file.open('w') as strings_fout, \
         labels_file.open('w') as labels_fout, \
         num_edits_file.open('w') as num_edits_fout, \
         contextlib.ExitStack() as exit_stack:
        if include_log_probability:
            log_probabilities_file = output_dir / 'log-probabilities.txt'
            print(f'writing {log_probabilities_file}')
            log_probabilities_fout = exit_stack.enter_context(log_probabilities_file.open('w'))
        if include_next_symbols:
            next_symbols_file = output_dir / 'next-symbols.jsonl'
            print(f'writing {next_symbols_file}')
            next_symbols_fout = exit_stack.enter_context(next_symbols_file.open('w'))
        if include_edit_distance:
            edit_distances_file = output_dir / 'edit-distances.txt'
            print(f'writing {edit_distances_file}')
            edit_distances_fout = exit_stack.enter_context(edit_distances_file.open('w'))
        for i in range(num_samples):
            s, label, parse, edit_distance, num_edits = generate_example(
                language,
                length_range,
                alphabet_size,
                only_negative,
                perturbation_probability,
                strict_num_edits_distribution,
                include_log_probability,
                include_next_symbols,
                include_edit_distance,
                generator,
                excluded_strings
            )
            print(' '.join(alphabet[c] for c in s), file=strings_fout)
            print(label, file=labels_fout)
            if label:
                if include_log_probability:
                    print(parse.log_probability, file=log_probabilities_fout)
                if include_next_symbols:
                    check_next_symbols(s, parse.next_symbols)
                    next_symbols_data = []
                    for next_symbols_i in parse.next_symbols:
                        has_eos = ReservedSymbol.EOS in next_symbols_i
                        non_eos_symbols = sorted(alphabet[c] for c in next_symbols_i if c != ReservedSymbol.EOS)
                        next_symbols_data.append({
                            's' : ' '.join(non_eos_symbols),
                            'e' : has_eos
                        })
                    write_json_line(next_symbols_data, next_symbols_fout)
            else:
                print(num_edits if num_edits is not None else '', file=num_edits_fout)
                if include_edit_distance:
                    assert isinstance(edit_distance, int) and edit_distance > 0
                    print(edit_distance, file=edit_distances_fout)
            if (
                generated_strings_output is not None and
                (generated_strings_max_length is None or len(s) <= generated_strings_max_length)
            ):
                generated_strings_output.add(s)

def check_next_symbols(s, next_symbols):
    if len(next_symbols) != len(s) + 1:
        raise ValueError(
            f'a list of next symbol sets is the wrong length '
            f'(expected {len(s) + 1}, got {len(next_symbols)}); '
            f'string: {s}; '
            f'next symbol sets: {next_symbols}'
        )
    for i, (s_i, next_symbols_i) in enumerate(zip(s, next_symbols[:-1], strict=True)):
        if s_i not in next_symbols_i:
            raise ValueError(
                f'a symbol in a string does not appear in the set '
                f'of valid next symbols; '
                f'at position {i}, symbol {s_i} is not in the set {next_symbols_i}; '
                f'string: {s}; '
                f'next symbol sets: {next_symbols}'
            )
    if ReservedSymbol.EOS not in next_symbols[-1]:
        raise ValueError(
            f'ReservedSymbol.EOS does not appear in the last set of valid next '
            f'symbols; '
            f'string: {s}; '
            f'next symbol sets: {next_symbols}'
        )

def generate_example(
    language,
    length_range,
    alphabet_size,
    only_negative,
    perturbation_probability,
    strict_num_edits_distribution,
    include_log_probability,
    include_next_symbols,
    include_edit_distance,
    generator,
    excluded_strings
):
    if only_negative:
        label = 0
    else:
        label = generator.randrange(2)
    if label:
        s, parse = generate_positive_example(
            language,
            include_log_probability,
            include_next_symbols,
            generator,
            excluded_strings
        )
        if not (length_range[0] <= len(s) <= length_range[1]):
            raise ValueError(
                f'a positive example of length {len(s)} was generated outside '
                f'the desired length range: {s}'
            )
        edit_distance = None
        num_edits = None
    else:
        s, edit_distance, num_edits = generate_negative_example(
            language,
            length_range,
            alphabet_size,
            perturbation_probability,
            strict_num_edits_distribution,
            include_edit_distance,
            generator,
            excluded_strings
        )
        parse = None
    return s, label, parse, edit_distance, num_edits

class CannotGenerateExampleError(ValueError):
    pass

def generate_positive_example(
    language,
    include_log_probability,
    include_next_symbols,
    generator,
    excluded_strings
):
    for _ in range(100):
        s, parse = language.sample(
            generator=generator,
            include_log_probability=include_log_probability,
            include_next_symbols=include_next_symbols
        )
        if excluded_strings is None or s not in excluded_strings:
            return s, parse
    else:
        raise CannotGenerateExampleError(
            'unable to generate a held-out positive example after 100 tries'
        )

def generate_negative_example(
    language,
    length_range,
    alphabet_size,
    perturbation_probability,
    strict_num_edits_distribution,
    include_edit_distance,
    generator,
    excluded_strings
):
    if strict_num_edits_distribution:
        # Decide whether to propose only random or perturbed strings.
        if generator.random() < perturbation_probability:
            # Sample a number of edits, and keep sampling strings with exactly
            # that many edits until it's negative.
            num_edits = generate_random_num_edits(generator)
            def propose_negative_example():
                s = generate_perturbed_string_with_num_edits(
                    language,
                    length_range,
                    alphabet_size,
                    num_edits,
                    generator
                )
                return s, num_edits
        else:
            # Sample random strings.
            def propose_negative_example():
                s = generate_random_string(length_range, alphabet_size, generator)
                return s, None
    else:
        # Decide which type of negative sampling to do on every retry.
        def propose_negative_example():
            if generator.random() < perturbation_probability:
                return generate_perturbed_string(
                    language,
                    length_range,
                    alphabet_size,
                    generator
                )
            else:
                s = generate_random_string(length_range, alphabet_size, generator)
                return s, None
    for _ in range(100):
        s, num_edits = propose_negative_example()
        if excluded_strings is None or s not in excluded_strings:
            is_negative, edit_distance = language.is_negative(
                s,
                include_edit_distance=include_edit_distance
            )
            if is_negative:
                return s, edit_distance, num_edits
    else:
        raise CannotGenerateExampleError(
            'unable to generate a negative example after 100 tries'
        )

def generate_random_string(length_range, alphabet_size, generator):
    lo, hi = length_range
    length = generator.randint(lo, hi)
    return tuple(generator.randrange(alphabet_size) for _ in range(length))

def generate_perturbed_string(language, length_range, alphabet_size, generator):
    num_edits = generate_random_num_edits(generator)
    s = generate_perturbed_string_with_num_edits(
        language,
        length_range,
        alphabet_size,
        num_edits,
        generator
    )
    return s, num_edits

def generate_random_num_edits(generator):
    return python_to_numpy_generator(generator).geometric(0.5)

def python_to_numpy_generator(generator):
    return numpy.random.default_rng(generator.getrandbits(32))

def generate_perturbed_string_with_num_edits(
    language,
    length_range,
    alphabet_size,
    num_edits,
    generator
):
    s, _ = language.sample(
        generator=generator,
        include_log_probability=False,
        include_next_symbols=False
    )
    r = apply_random_edits(s, num_edits, length_range, alphabet_size, generator)
    return tuple(r)

def apply_random_edits(s, num_edits, length_range, alphabet_size, generator):
    r = list(s)
    for _ in range(num_edits):
        apply_random_edit(r, length_range, alphabet_size, generator)
    return r

class Edit(enum.IntEnum):
    INSERT = enum.auto()
    REPLACE = enum.auto()
    DELETE = enum.auto()

def apply_random_edit(s, length_range, alphabet_size, generator):
    lo, hi = length_range
    choices = []
    if len(s) < hi:
        choices.append(Edit.INSERT)
    if len(s) > 0 and alphabet_size > 1:
        choices.append(Edit.REPLACE)
    if len(s) > lo:
        choices.append(Edit.DELETE)
    match generator.choice(choices):
        case Edit.INSERT:
            apply_random_insertion(s, alphabet_size, generator)
        case Edit.REPLACE:
            apply_random_replacement(s, alphabet_size, generator)
        case Edit.DELETE:
            apply_random_deletion(s, generator)
        case _:
            raise ValueError

def apply_random_insertion(s, alphabet_size, generator):
    i = generator.randrange(len(s) + 1)
    symbol = generator.randrange(alphabet_size)
    s.insert(i, symbol)

def apply_random_replacement(s, alphabet_size, generator):
    i = generator.randrange(len(s))
    old_symbol = s[i]
    new_symbol = generator.randrange(alphabet_size - 1)
    new_symbol += int(new_symbol >= old_symbol)
    s[i] = new_symbol

def apply_random_deletion(s, generator):
    i = generator.randrange(len(s))
    del s[i]

def get_hand_coded_language(name):
    match name:
        case 'majority':
            return Majority()
        case 'stack-manipulation':
            return StackManipulation()
        case 'marked-reversal':
            return MarkedReversal()
        case 'unmarked-reversal':
            return UnmarkedReversal()
        case 'marked-copy':
            return MarkedCopy()
        case 'missing-duplicate-string':
            return MissingDuplicateString()
        case 'odds-first':
            return OddsFirst()
        case 'binary-addition':
            return BinaryAddition()
        case 'binary-multiplication':
            return BinaryMultiplication()
        case 'compute-sqrt':
            return ComputeSqrt()
        case 'bucket-sort':
            return BucketSort()
        case _:
            raise ValueError(f'invalid hand-coded language name: {name}')

def get_automaton_language(filename, dtype, device):
    data = torch.load(filename, map_location=torch.device('cpu'))
    automaton = data['sampler']
    alphabet = data.pop('alphabet', None)
    match automaton:
        case NormalizedCountingFiniteAutomaton():
            return FiniteAutomatonLanguage(automaton, alphabet, dtype, device)
        case _:
            raise ValueError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=pathlib.Path, required=True,
        help='Output directory where the data files will be written.')
    parser.add_argument('--random-seed', type=int, required=True,
        help='Random seed used for random sampling.')
    parser.add_argument('--language',
        help='Name of a hand-coded language to sample from.')
    parser.add_argument('--sampler', type=pathlib.Path,
        help='A .pt file containing an automaton prepared for sampling.')
    parser.add_argument('--perturbation-probability', type=float, default=0.5,
        help='The probability that a proposed negative example (before '
             'rejection sampling) will be a perturbed positive example.')
    parser.add_argument('--strict-num-edits-distribution', action='store_true', default=False,
        help='Make sure the distribution of number of edits on negative '
             'examples is followed even after rejection sampling.')
    parser.add_argument('--include-log-probability', action='store_true', default=False,
        help='Output the log probability of each positive example.')
    parser.add_argument('--include-next-symbols', action='store_true', default=False,
        help='Output the set of valid next symbols at each position for each '
             'positive example.')
    parser.add_argument('--skip-standard-datasets', action='store_true', default=False,
        help='Don\'t generate the datasets besides test-edit-distance.')
    parser.add_argument('--skip-test-edit-distance', action='store_true', default=False,
        help='Don\'t generate the test-edit-distance dataset.')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--device')
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = parse_device(args.device)

    match (args.language is not None, args.sampler is not None):
        case (True, False):
            language = get_hand_coded_language(args.language)
        case (False, True):
            language = get_automaton_language(args.sampler, dtype, device)
        case _:
            parser.error('exactly one of --language or --sampler must be given')

    generator = random.Random(args.random_seed)
    training_length_range = (0, 40)
    validation_long_length_range = (0, 2 * training_length_range[1])
    test_length_range = (0, 500)
    test_examples_per_length = 10
    alphabet = [language.symbol_to_str(c) for c in range(language.alphabet_size())]
    try:
        training_language = language.with_length_range(training_length_range)
        validation_long_language = language.with_length_range(validation_long_length_range)
        test_language = language.with_length_range(test_length_range)
    except EmptyLanguageError as e:
        print(f'error: cannot sample positive examples: {e}', file=sys.stderr)
        sys.exit(1)

    def generate_split(
        language,
        length_range,
        num_samples,
        output_dir,
        only_negative=False,
        include_edit_distance=False,
        generated_strings_output=None,
        generated_strings_max_length=None,
        excluded_strings=None
    ):
        generate_files(
            language=language,
            length_range=length_range,
            alphabet=alphabet,
            num_samples=num_samples,
            only_negative=only_negative,
            perturbation_probability=args.perturbation_probability,
            strict_num_edits_distribution=args.strict_num_edits_distribution,
            include_log_probability=args.include_log_probability,
            include_next_symbols=args.include_next_symbols,
            include_edit_distance=include_edit_distance,
            generator=generator,
            output_dir=output_dir,
            generated_strings_output=generated_strings_output,
            generated_strings_max_length=generated_strings_max_length,
            excluded_strings=excluded_strings
        )

    if not args.skip_standard_datasets:
        excluded_strings = set()
        generate_split(
            language=training_language,
            length_range=training_length_range,
            num_samples=10000,
            output_dir=args.output,
            generated_strings_output=excluded_strings
        )
        generate_split(
            language=training_language,
            length_range=training_length_range,
            num_samples=1000,
            output_dir=args.output / 'datasets' / 'validation-short',
            generated_strings_output=excluded_strings
        )
        generate_split(
            language=validation_long_language,
            length_range=validation_long_length_range,
            num_samples=1000,
            output_dir=args.output / 'datasets' / 'validation-long',
            generated_strings_output=excluded_strings,
            generated_strings_max_length=training_length_range[1]
        )
        generate_split(
            language=test_language,
            length_range=test_length_range,
            num_samples=(test_length_range[1] - test_length_range[0] + 1) * test_examples_per_length,
            output_dir=args.output / 'datasets' / 'test'
        )
        test_short_held_out_dir = args.output / 'datasets' / 'test-short-held-out'
        try:
            generate_split(
                language=training_language,
                length_range=training_length_range,
                num_samples=1000,
                output_dir=test_short_held_out_dir,
                excluded_strings=excluded_strings
            )
        except CannotGenerateExampleError as e:
            print(e)
            print('skipping test-short-held-out')
            print(f'removing {test_short_held_out_dir}')
            shutil.rmtree(test_short_held_out_dir)
    if not args.skip_test_edit_distance and test_language.supports_edit_distance():
        generate_split(
            language=test_language,
            length_range=test_length_range,
            num_samples=50,
            output_dir=args.output / 'datasets' / 'test-edit-distance',
            only_negative=True,
            include_edit_distance=True
        )

if __name__ == '__main__':
    main()
