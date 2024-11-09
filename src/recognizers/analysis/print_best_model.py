import argparse
import pathlib
import sys

from recognizers.analysis.print_table_util import (
    read_data_for_multiple_trials
)
from recognizers.analysis.print_table import (
    read_accuracy_from_trial
)

def get_test_accuracy(trial):
    result = read_accuracy_from_trial(trial, 'test')
    if result is None:
        raise ValueError(f'missing test accuracy: {trial.path}')
    return result

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=pathlib.Path, nargs='+')
    args = parser.parse_args()

    trials, missing_dirs = read_data_for_multiple_trials(args.inputs, capture_all_events=False)
    for d in missing_dirs:
        print(f'missing: {d}', file=sys.stderr)
    best_trial = max(trials, key=get_test_accuracy)
    print(best_trial.path)

if __name__ == '__main__':
    main()
