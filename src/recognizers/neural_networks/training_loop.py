import dataclasses
from typing import Optional

import torch

from rau.tasks.common.data import Dataset
from rau.tasks.common.training_loop import (
    add_training_loop_arguments as common_add_training_loop_arguments,
    get_training_loop_kwargs as common_get_training_loop_kwargs,
    TrainingLoop
)
from rau.tools.torch.model_interface import ModelInterface

from .batching import group_into_batches
from .data import VocabularyContainer
from .model_interface import ModelInput

def add_training_loop_arguments(parser):
    group = common_add_training_loop_arguments(parser,
        max_tokens_per_batch_help=
        'The maximum number of tokens allowed per batch. This puts a limit on '
        'the number of elements included in a single batch tensor, including '
        'BOS, EOS, and padding tokens. If a single example exceeds the limit, '
        'it is not discarded, but included in a batch by itself.'
    )
    group.add_argument('--language-modeling-loss-coefficient', type=float, default=1.0)
    group.add_argument('--next-symbols-loss-coefficient', type=float, default=1.0)

def get_training_loop_kwargs(parser, args):
    result = common_get_training_loop_kwargs(parser, args)
    for name in [
        'language_modeling_loss_coefficient',
        'next_symbols_loss_coefficient'
    ]:
        result[name] = getattr(args, name)
    return result

Example = tuple[torch.Tensor, tuple[bool, Optional[torch.Tensor]]]
PreparedBatch = tuple[
    ModelInput,
    tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
]

@dataclasses.dataclass
class RecognitionTrainingLoop(TrainingLoop[
    Example,
    PreparedBatch,
    VocabularyContainer
]):

    language_modeling_loss_coefficient: float
    next_symbols_loss_coefficient: float

    def get_validation_metric_name(self):
        return 'recognition_cross_entropy'

    def get_validation_metric_mode(self):
        return 'min'

    def generate_batches(self, examples, max_tokens):
        return generate_batches(examples, max_tokens)

    def get_prepared_batch_info(self, prepared_batch):
        (
            model_input,
            (
                recognition_expected_tensor,
                language_modeling_expected_tensor,
                next_symbols_expected_tensor,
                next_symbols_padding_mask,
                positive_output_lengths
            )
        ) = prepared_batch
        return dict(
            input_size=tuple(model_input.input_sequence.size()),
            recognition_output_size=tuple(recognition_expected_tensor.size()),
            language_modeling_output_size=(
                tuple(language_modeling_expected_tensor.size())
                if language_modeling_expected_tensor is not None
                else None
            ),
            next_symbols_output_size=(
                tuple(next_symbols_expected_tensor.size())
                if next_symbols_expected_tensor is not None
                else None
            )
        )

    def log_failed_batch(self, vocabulary, batch, info, console_logger, event_logger):
        if info is not None:
            console_logger.info(f'  input size: {info.get("input_size")}')
            console_logger.info(f'  recognition output size: {info.get("recognition_output_size")}')
            console_logger.info(f'  language modeling output size: {info.get("language_modeling_output_size")}')
            console_logger.info(f'  next symbols output size: {info.get("next_symbols_output_size")}')
        tokens = sum(len(x[0]) for x in batch)
        console_logger.info(f'  tokens: {tokens}')
        lengths = [len(x[0]) for x in batch]
        console_logger.info(f"  sequence lengths: {lengths}")
        token_strs = [
            [vocabulary.input_vocab.to_string(a) for a in x[0]]
            for x in batch
        ]
        sequences_str = '\n'.join(' '.join(x) for x in token_strs)
        console_logger.info(f'  sequences:\n{sequences_str}')
        return dict(
            **info,
            examples=token_strs
        )

    def get_loss(self, model, model_interface, prepared_batch):
        loss_terms = get_loss_terms(
            model,
            model_interface,
            prepared_batch,
            numerator_reduction='none',
            denominator_reduction='sum',
            label_smoothing_factor=self.label_smoothing_factor,
            include_accuracy=False
        )
        # Assign coefficients to the loss terms.
        if 'language_modeling_cross_entropy' in loss_terms:
            loss_terms['language_modeling_cross_entropy'] += (self.language_modeling_loss_coefficient,)
        if 'next_symbols_cross_entropy' in loss_terms:
            loss_terms['next_symbols_cross_entropy'] += (self.next_symbols_loss_coefficient,)
        return loss_terms

    def evaluate_batch(self, model, model_interface, prepared_batch):
        result = get_loss_terms(
            model,
            model_interface,
            prepared_batch,
            numerator_reduction='sum',
            denominator_reduction='sum',
            label_smoothing_factor=0.0,
            include_accuracy=True
        )
        return { k : (n.item(), d) for k, (n, d) in result.items() }

def generate_batches(examples, max_tokens):
    return group_into_batches(examples, lambda b, n: b * n <= max_tokens)

def get_loss_terms(
    model,
    model_interface,
    prepared_batch,
    numerator_reduction,
    denominator_reduction,
    label_smoothing_factor,
    include_accuracy
):
    """
    :param numerator_reduction: This can be none or sum. If none, then all
        numerators are returned as a 1-D tensor of values, with one value per
        example. If sum, then they are returned as 0-D tensors with a single
        value, as if the none version had been summed.
    :param denominator_reduction: This can be none or sum. If none, then all
        denominators are returned as a 1-D tensor of values, with one value per
        example. A value of `None` is equivalent to a tensor of all 1's. If
        sum, then they are returned as floats or ints, as if the none version
        had been summed.
    """
    (
        model_input,
        (
            expected_recognition_output,
            expected_language_modeling_output,
            expected_next_symbols_output,
            next_symbols_padding_mask,
            positive_output_lengths
        )
    ) = prepared_batch
    (
        recognition_logits,
        language_modeling_logits,
        next_symbols_logits
    ) = model_interface.get_logits(
        model,
        model_input
    )
    result = {}
    # Compute the recognition loss using binary cross-entropy.
    recognition_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        recognition_logits,
        expected_recognition_output,
        reduction=numerator_reduction
    )
    match denominator_reduction:
        case 'none':
            num_examples_denominator = None
        case 'sum':
            num_examples_denominator = len(recognition_logits)
        case _:
            raise ValueError
    result['recognition_cross_entropy'] = (
        recognition_loss,
        num_examples_denominator
    )
    if include_accuracy:
        # The model accepts iff the logit is >= 0 (the probability is >= 0.5).
        recognition_predictions = recognition_logits >= 0.0
        recognition_accuracy = recognition_predictions == expected_recognition_output
        match numerator_reduction:
            case 'none':
                pass
            case 'sum':
                recognition_accuracy = torch.sum(recognition_accuracy)
            case _:
                raise ValueError
        result['recognition_accuracy'] = (
            recognition_accuracy,
            num_examples_denominator
        )
    if language_modeling_logits is not None:
        # Compute the language modeling loss using cross-entropy.
        pad_index = model_interface.output_padding_index
        language_modeling_ce = torch.nn.functional.cross_entropy(
            language_modeling_logits.permute(0, 2, 1),
            expected_language_modeling_output,
            ignore_index=pad_index,
            reduction='none',
            label_smoothing=label_smoothing_factor
        )
        # Average over timesteps.
        language_modeling_mean_ce = torch.sum(language_modeling_ce, dim=1) / positive_output_lengths
        match numerator_reduction:
            case 'sum':
                language_modeling_loss = torch.sum(language_modeling_mean_ce)
            case 'none':
                language_modeling_loss = language_modeling_mean_ce
            case _:
                raise ValueError
        match denominator_reduction:
            case 'none':
                num_positive_denominator = None
            case 'sum':
                num_positive_denominator = len(expected_language_modeling_output)
            case _:
                raise ValueError
        result['language_modeling_cross_entropy'] = (
            language_modeling_loss,
            num_positive_denominator
        )
    if next_symbols_logits is not None:
        # Compute the valid symbols loss using binary cross-entropy.
        pad_index = model_interface.output_padding_index
        next_symbols_unmasked_ce = torch.nn.functional.binary_cross_entropy_with_logits(
            next_symbols_logits,
            expected_next_symbols_output,
            reduction='none'
        )
        # Average over alphabet symbols and mask out padding positions.
        next_symbols_alphabet_mean_ce = torch.mean(
            next_symbols_unmasked_ce,
            dim=2
        ) * next_symbols_padding_mask
        # Average over timesteps.
        next_symbols_mean_ce = torch.sum(next_symbols_alphabet_mean_ce, dim=1) / positive_output_lengths
        match numerator_reduction:
            case 'sum':
                next_symbols_loss = torch.sum(next_symbols_mean_ce)
            case 'none':
                next_symbols_loss = next_symbols_mean_ce
            case _:
                raise ValueError
        match denominator_reduction:
            case 'none':
                num_positive_denominator = None
            case 'sum':
                num_positive_denominator = len(expected_next_symbols_output)
            case _:
                raise ValueError
        result['next_symbols_cross_entropy'] = (
            next_symbols_loss,
            num_positive_denominator
        )
        if include_accuracy:
            next_symbols_predictions = next_symbols_logits >= 0.0
            next_symbols_accuracy = (
                (next_symbols_predictions == expected_next_symbols_output) *
                next_symbols_padding_mask[:, :, None]
            )
            next_symbols_set_accuracy = torch.all(next_symbols_accuracy, dim=2)
            next_symbols_string_accuracy = torch.all(next_symbols_set_accuracy, dim=1)
            match numerator_reduction:
                case 'none':
                    next_symbols_symbol_accuracy = torch.sum(next_symbols_accuracy, dim=(1, 2))
                    next_symbols_set_accuracy = torch.sum(next_symbols_set_accuracy, dim=1)
                case 'sum':
                    next_symbols_symbol_accuracy = torch.sum(next_symbols_accuracy)
                    next_symbols_set_accuracy = torch.sum(next_symbols_set_accuracy)
                    next_symbols_string_accuracy = torch.sum(next_symbols_string_accuracy)
                case _:
                    raise ValueError
            vocab_size = next_symbols_logits.size(2)
            match denominator_reduction:
                case 'none':
                    num_next_symbols_denominator = positive_output_lengths * vocab_size
                    num_next_symbols_sets_denominator = positive_output_lengths
                case 'sum':
                    num_next_symbols_sets_denominator = torch.sum(positive_output_lengths).item()
                    num_next_symbols_denominator = num_next_symbols_sets_denominator * vocab_size
                case _:
                    raise ValueError
            result['next_symbols_symbol_accuracy'] = (
                next_symbols_symbol_accuracy,
                num_next_symbols_denominator
            )
            result['next_symbols_set_accuracy'] = (
                next_symbols_set_accuracy,
                num_next_symbols_sets_denominator
            )
            result['next_symbols_string_accuracy'] = (
                next_symbols_string_accuracy,
                num_positive_denominator
            )
    return result
