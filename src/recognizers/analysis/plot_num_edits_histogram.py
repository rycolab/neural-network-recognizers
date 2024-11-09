import argparse
import collections
import pathlib

import matplotlib.pyplot as plt
import numpy

def load_num_edits(fin):
    for line in fin:
        line = line.strip()
        if line:
            yield int(line)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'training':
        data_directory = args.training_data
    else:
        data_directory = args.training_data / 'datasets' / dataset
    num_edits_file = data_directory / 'num-edits.txt'
    with num_edits_file.open() as fin:
        counts = collections.Counter(load_num_edits(fin))
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_ylabel('Count')
    ax.set_xlabel('Number of Edits')
    min_count = min(counts.keys())
    max_count = max(counts.keys())
    x = numpy.arange(min_count, max_count + 1)
    y = numpy.array([counts[i] for i in x])
    ax.bar(x, y)
    plt.tight_layout()
    print(f'writing file {args.output}')
    fig.savefig(args.output)

if __name__ == '__main__':
    main()
