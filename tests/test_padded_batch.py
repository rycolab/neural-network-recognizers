import argparse
import random

import pytest
import torch

from recognizers.neural_networks.vocabulary import VocabularyData
from recognizers.neural_networks.model_interface import RecognitionModelInterface
from recognizers.neural_networks.training_loop import (
    RecognitionTrainingLoop,
    add_training_loop_arguments,
    get_training_loop_kwargs,
)

@pytest.mark.parametrize('use_lm', [False, True])
@pytest.mark.parametrize('use_ns', [False, True])
@pytest.mark.parametrize('architecture', ['transformer', 'rnn', 'lstm'])
def test_single_matches_batched(use_lm, use_ns, architecture):
    argv = [
        '--device', 'cpu',
        '--parameter-seed', '123',
        '--architecture', architecture,
        '--num-layers', '5',
        '--dropout', '0',
        '--init-scale', '0.1',
        '--max-epochs', '1',
        '--random-shuffling-seed', '1',
        '--max-tokens-per-batch', '1',
        '--optimizer', 'SGD',
        '--initial-learning-rate', '0.01',
        '--label-smoothing-factor', '0',
        '--gradient-clipping-threshold', '5',
        '--early-stopping-patience', '10',
        '--learning-rate-patience', '5',
        '--learning-rate-decay-factor', '0.5',
        '--examples-per-checkpoint', '1'
    ]
    match architecture:
        case 'transformer':
            argv.extend([
                '--d-model', '32',
                '--num-heads', '4',
                '--feedforward-size', '64',
            ])
        case 'rnn' | 'lstm':
            argv.extend([
                '--hidden-units', '32'
            ])
    if use_lm:
        argv.extend([
            '--use-language-modeling-head',
            '--language-modeling-loss-coefficient', '0.123'
        ])
    if use_ns:
        argv.extend([
            '--use-next-symbols-head',
            '--next-symbols-loss-coefficient', '0.345'
        ])

    model_interface = RecognitionModelInterface(use_load=False, use_output=False)
    parser = argparse.ArgumentParser()
    model_interface.add_arguments(parser)
    model_interface.add_forward_arguments(parser)
    add_training_loop_arguments(parser)
    args = parser.parse_args(argv)

    device = model_interface.get_device(args)
    training_loop = RecognitionTrainingLoop(**get_training_loop_kwargs(parser, args))
    vocabulary_data = VocabularyData(
        tokens=['0', '1'],
        allow_unk=False
    )
    saver = model_interface.construct_saver(args, vocabulary_data)

    generator = random.Random(123)
    lengths = [0, 1, 3, 7, 10, 13, 23]
    batch = []
    for length in lengths:
        for label in (True, False):
            string = torch.tensor([generator.randrange(2) for _ in range(length)])
            if label and model_interface.use_next_symbols_head:
                next_symbols = []
                for _ in range(length + 1):
                    next_symbols_i = []
                    for symbol in range(model_interface.output_vocabulary_size):
                        if generator.randrange(2):
                            next_symbols_i.append(symbol)
                    next_symbols.append(next_symbols_i)
            else:
                next_symbols = None
            batch.append((string, (label, next_symbols)))
    generator.shuffle(batch)
    num_examples = len(batch)

    # Run forward and backward passes on the whole batch.
    saver.model.zero_grad()
    _, batched_loss, _, _, _ = training_loop.get_prepared_batch_and_loss(
        saver,
        model_interface,
        batch
    )
    batched_loss.backward()
    batched_grads = { name : param.grad.clone() for name, param in saver.model.named_parameters() }

    # Run forward and backward passes on the individual examples and accumulate
    # their gradient.
    saver.model.zero_grad()
    for example in batch:
        _, example_loss, _, _, _ = training_loop.get_prepared_batch_and_loss(
            saver,
            model_interface,
            [example]
        )
        example_loss = example_loss / num_examples
        example_loss.backward()
    single_grads = { name : param.grad.clone() for name, param in saver.model.named_parameters() }

    names = batched_grads.keys()
    assert single_grads.keys() == names
    for name in names:
        batched_grad = batched_grads[name]
        single_grad = single_grads[name]
        torch.testing.assert_close(
            batched_grad,
            single_grad,
            msg=f'gradients not equal for parameter {name}'
        )

def clone_state_dict(x):
    return { k : v.clone() for k, v in x.items() }
