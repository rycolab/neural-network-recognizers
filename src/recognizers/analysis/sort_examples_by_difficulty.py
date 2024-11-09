import argparse
import pathlib

from recognizers.tools.jsonl import load_jsonl_file

def divide(x):
    numer, denom = x
    return numer / denom

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'training':
        data_directory = args.training_data
    else:
        data_directory = args.training_data / 'datasets' / dataset
    data_file = data_directory / 'main.tok'
    labels_file = data_directory / 'labels.txt'
    scores_file = args.model / 'eval' / f'{dataset}.jsonl'
    with data_file.open() as data_fin, \
         labels_file.open() as labels_fin, \
         scores_file.open() as scores_fin:
        examples = list(zip(
            data_fin,
            labels_fin,
            load_jsonl_file(scores_fin),
            strict=True
        ))
    examples.sort(key=lambda x: divide(x[2]['recognition_cross_entropy']), reverse=True)
    for string, label, scores in examples:
        string = ' '.join(string.split())
        label = int(label.strip())
        recognition_cross_entropy = divide(scores['recognition_cross_entropy'])
        accuracy = 'correct' if divide(scores['recognition_accuracy']) else 'incorrect'
        print(f'{label}\t{accuracy}\t{recognition_cross_entropy}\t{string}')

if __name__ == '__main__':
    main()
