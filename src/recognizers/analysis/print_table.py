import json
import math
import sys

import numpy

from recognizers.analysis.print_table_util import (
    run_main,
    Column,
    format_text,
    format_float,
    format_mean_and_variance
)

def get_best_trial(cache):
    trials = cache['trials']
    if trials:
        return trials[cache['best_index']]

def get_best_index(cache):
    trials = cache['trials']
    if trials:
        return numpy.argmin([
            trial.info['train']['best_validation_scores']['recognition_cross_entropy']
            for trial in trials
        ])

def get_validation_cross_entropy(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return best_trial.info['train']['best_validation_scores']['recognition_cross_entropy']

def get_training_accuracy(cache):
    return read_accuracy(cache, 'training')

def get_validation_accuracy(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return best_trial.info['train']['best_validation_scores']['recognition_accuracy']

def get_test_short_held_out_accuracy(cache):
    return read_accuracy(cache, 'test-short-held-out')

def get_test_accuracy(cache):
    return read_accuracy(cache, 'test')

def get_all_test_accuracy(cache):
    trials = cache['trials']
    result = (read_accuracy_from_trial(trial, 'test') for trial in trials)
    return [x for x in result if x is not None]

def get_mean_test_accuracy(cache):
    scores = cache['all_test_accuracy']
    if scores:
        mean = numpy.mean(scores)
        std = numpy.std(scores)
        return mean, std

def get_max_test_accuracy(cache):
    scores = cache['all_test_accuracy']
    if scores:
        return max(scores)

def read_eval_data(trial, dataset):
    scores_path = trial.path / 'eval' / f'{dataset}.json'
    try:
        with scores_path.open() as fin:
            return json.load(fin)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        return None

def read_accuracy_from_trial(trial, dataset):
    data = read_eval_data(trial, dataset)
    if data is not None:
        return data['scores']['recognition_accuracy']

def read_accuracy(cache, dataset):
    best_trial = cache['best_trial']
    if best_trial:
        return read_accuracy_from_trial(best_trial, dataset)

def main():
    run_main(
        columns=[
            Column('Model', 'l', 'label', format_text()),
            Column('Train', 'c', 'training_accuracy', format_float(places=3), bold_max=True),
            Column('Val.\\ CE $\\downarrow$', 'c', 'validation_cross_entropy', format_float(places=3), bold_min=True),
            Column('Val.', 'c', 'validation_accuracy', format_float(places=3), bold_max=True),
            Column('S.\\ Test', 'c', 'test_short_held_out_accuracy', format_float(places=3), bold_max=True),
            Column('L.\\ Test', 'c', 'test_accuracy', format_float(places=3), bold_max=True),
            Column('L.\\ Test (Mean)', 'c', 'mean_test_accuracy', format_mean_and_variance(places=(3, 2)), bold_max=True, get_comparable_value=lambda x: x[0]),
            Column('L.\\ Test (Max)', 'c', 'max_test_accuracy', format_float(places=3), bold_max=True)
        ],
        callbacks={
            'best_trial' : get_best_trial,
            'best_index' : get_best_index,
            'training_accuracy' : get_training_accuracy,
            'validation_cross_entropy' : get_validation_cross_entropy,
            'validation_accuracy' : get_validation_accuracy,
            'test_short_held_out_accuracy' : get_test_short_held_out_accuracy,
            'test_accuracy' : get_test_accuracy,
            'all_test_accuracy' : get_all_test_accuracy,
            'mean_test_accuracy' : get_mean_test_accuracy,
            'max_test_accuracy' : get_max_test_accuracy,
        }
    )

if __name__ == '__main__':
    main()
