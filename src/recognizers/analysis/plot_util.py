import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from recognizers.analysis.print_table_util import (
    read_data_for_multiple_trials,
    Cache,
    get_runs
)
from recognizers.analysis.print_table import (
    get_best_trial,
    get_best_index,
    read_eval_data
)

def get_test_data(cache):
    best_trial = cache['best_trial']
    if best_trial:
        return read_eval_data(best_trial, 'test')

def get_test_accuracy(cache):
    test_data = cache['test_data']
    if test_data is not None:
        return test_data['scores']['recognition_accuracy']

def get_test_accuracy_by_length(cache):
    test_data = cache['test_data']
    if test_data is not None:
        return [
            (length, scores['recognition_accuracy'])
            for length, scores in test_data['scores_by_length']
        ]

def run_main(callbacks, capture_all_events=True):

    parser = argparse.ArgumentParser()
    parser.add_argument('--label', action='append', default=[])
    parser.add_argument('--inputs', type=pathlib.Path, nargs='*', action='append', default=[])
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    labels = args.label
    input_lists = args.inputs
    if len(labels) != len(input_lists):
        parser.error('must have the same number of --label and --input arguments')

    target_runs = max(len(input_list) for input_list in input_lists)
    labels_and_trials = []
    all_missing_dirs = []
    for label, input_list in zip(labels, input_lists):
        trials, missing_dirs = read_data_for_multiple_trials(input_list, capture_all_events)
        labels_and_trials.append((label, trials))
        all_missing_dirs.extend(missing_dirs)
    show_runs = not all(len(trials) == target_runs for label, trials in labels_and_trials)

    if show_runs:
        callbacks['runs'] = get_runs

    caches = []
    for label, trials in labels_and_trials:
        cache = Cache(callbacks)
        cache['label'] = label
        cache['trials'] = trials
        caches.append(cache)

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Length')

    line_no = 0
    for (label, trials), cache in zip(labels_and_trials, caches):
        if show_runs:
            runs = cache['runs']
            label += f' ({runs} runs)'
        test_accuracy = cache['test_accuracy']
        if test_accuracy is not None and test_accuracy >= 0.95:
            data = cache['test_accuracy_by_length']
            if data is not None:
                x = [x for x, y in data]
                y = [y for x, y in data]
                ax.plot(x, y, marker=(2, 0, 15 + line_no * 15), markevery=50, label=label)
                line_no += 1
    ax.axvline(x=40, linestyle='--', color='black')
    ax.axvline(x=80, linestyle='--', color='black')

    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    plt.tight_layout()
    print(f'writing file {args.output}')
    fig.savefig(args.output)

    if show_runs:
        print(f'% info: results are not complete (targeting {target_runs} runs)')
    else:
        print(f'% info: all results are complete and are aggregated from {target_runs} runs')
    for missing_dir in all_missing_dirs:
        print(f'% missing: {missing_dir}', file=sys.stderr)

def main():
    run_main({
        'best_trial' : get_best_trial,
        'best_index' : get_best_index,
        'test_data' : get_test_data,
        'test_accuracy' : get_test_accuracy,
        'test_accuracy_by_length' : get_test_accuracy_by_length
    })

if __name__ == '__main__':
    main()
