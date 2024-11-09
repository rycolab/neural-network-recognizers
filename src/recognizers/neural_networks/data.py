import dataclasses
import itertools
import pathlib
from collections.abc import Iterable

import torch

from rau.tasks.common.data import load_prepared_data_file
from rau.vocab import ToStringVocabulary

from .vocabulary import VocabularyData, load_vocabulary_data_from_file

@dataclasses.dataclass
class VocabularyContainer:
    input_vocab: ToStringVocabulary
    output_vocab: ToStringVocabulary

def add_data_arguments(parser, validation=True):
    group = parser.add_argument_group('Dataset options')
    group.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing prepared training data. The file '
             '<training-data>/main.prepared will be used as the training '
             'data, and the file <training-data>/main.vocab will be used as '
             'the vocabulary.')
    if validation:
        group.add_argument('--validation-data', default='validation',
            help='Name of the dataset in the prepared training data directory '
                 'that will be used as validation data. The file '
                 '<training-data>/datasets/<validation-data>/main.prepared '
                 'will be used as the validation data. The default name is '
                 '"validation".')

def load_prepared_data_from_directory(directory, model_interface):
    strings_data = load_prepared_data_file(directory / 'main.prepared')
    labels_data = load_prepared_labels_file(directory / 'labels.prepared')
    if model_interface.use_next_symbols_head:
        next_symbols_data = load_prepared_next_symbols_file(
            directory / 'next-symbols.prepared',
            labels_data
        )
    else:
        next_symbols_data = itertools.repeat(None, len(labels_data))
    return list(zip(strings_data, zip(labels_data, next_symbols_data, strict=True), strict=True))

def load_prepared_labels_file(path: pathlib.Path) -> list[bool]:
    return torch.load(path)

def load_prepared_next_symbols_file(
    path: pathlib.Path,
    labels: list[bool]
) -> Iterable[list[list[int]]]:
    data = torch.load(path)
    data_it = iter(data)
    for label in labels:
        if label:
            yield next(data_it)
        else:
            yield None
    try:
        next(data_it)
    except StopIteration:
        pass
    else:
        raise ValueError('there are more lists of valid token sets than positive labels')

def load_vocabulary_data(args, parser) -> VocabularyData:
    return load_vocabulary_data_from_file(args.training_data / 'main.vocab')

def load_prepared_data(args, parser, vocabulary_data, model_interface, builder=None):
    training_data = load_prepared_data_from_directory(
        args.training_data,
        model_interface
    )
    if hasattr(args, 'validation_data'):
        validation_data = load_prepared_data_from_directory(
            args.training_data / 'datasets' / args.validation_data,
            model_interface
        )
    else:
        validation_data = None
    input_vocab, output_vocab = model_interface.get_vocabularies(
        vocabulary_data,
        builder
    )
    return (
        training_data,
        validation_data,
        VocabularyContainer(input_vocab, output_vocab)
    )

def load_vocabularies(args, parser, model_interface, builder=None):
    input_vocab, output_vocab = model_interface.get_vocabularies(
        load_vocabulary_data(args, parser),
        builder
    )
    return VocabularyContainer(input_vocab, output_vocab)
