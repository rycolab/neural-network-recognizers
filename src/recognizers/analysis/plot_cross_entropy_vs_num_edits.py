import argparse
import json
import pathlib

def load_labels(fin):
    for line in fin:
        yield bool(int(line.strip()))

def load_negative_score_lines(labels_fin, scores_fin):
    labels = load_labels(labels_fin)
    for label, scores_line in zip(labels, scores_fin, strict=True):
        if not label:
            yield scores_line

def load_num_edits(fin):
    for line in fin:
        line = line.strip()
        if line:
            yield int(line)
        else:
            yield None

def load_num_edits_and_scores(labels_fin, num_edits_fin, scores_fin):
    negative_score_lines = load_negative_score_lines(labels_fin, scores_fin)
    num_edits_list = load_num_edits(num_edits_fin)
    for score_line, num_edits in zip(negative_score_lines, num_edits_list, strict=True):
        if num_edits is not None:
            yield num_edits, json.loads(score_line)

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
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'training':
        data_directory = args.training_data
    else:
        data_directory = args.training_data / 'datasets' / dataset
    labels_file = data_directory / 'labels.txt'
    num_edits_file = data_directory / 'num-edits.txt'
    scores_file = args.model / 'eval' / f'{dataset}.jsonl'
    with labels_file.open() as labels_fin, \
         num_edits_file.open() as num_edits_fin, \
         scores_file.open() as scores_fin:
        data = list(load_num_edits_and_scores(labels_fin, num_edits_fin, scores_fin))
    x = [n for n, s in data]
    y = [divide(s['recognition_cross_entropy']) for n, s in data]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_ylabel('Recognition Cross Entropy (nats)')
    ax.set_xlabel('Number of Edits')
    ax.plot(x, y, '.', alpha=0.5)
    plt.tight_layout()
    print(f'writing {args.output}')
    fig.savefig(args.output)

if __name__ == '__main__':
    main()
