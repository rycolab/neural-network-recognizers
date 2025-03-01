import argparse
import json
import numpy as np
import math
import pathlib
import itertools
import matplotlib.pyplot as plt

from recognizers.analysis.plot_cross_entropy_vs_num_edits import (
    load_labels,
    load_negative_score_lines,
    divide
)

from rau.tools.logging import read_log_file

def load_scores(fin):
    for score_line in fin:
        yield score_line

def load_lengths(fin):
    for line in fin:
        yield len(line.rstrip().split())

def load_lengths_and_scores(strings_fin, scores_fin):
    scores_list = load_scores(scores_fin)
    lengths_list = load_lengths(strings_fin)
    for score_line, l in zip(scores_list, lengths_list, strict=True):
        yield l, json.loads(score_line)

def get_validation_ce(fin):
    with open(fin) as f:
        log = read_log_file(f)
        for event in log:
            if event.type == 'train':
                return event.data['best_validation_scores']['recognition_cross_entropy']

def get_test_accuracy(fin):
    with open(fin) as f:
        data = json.load(f)
        return data['scores']['recognition_accuracy']

def get_smoothed_points(xs, ys):
    def points_in_window(middle, points, window_size=10):
        return [(x, y) for x, y in points if middle - window_size <= x and x <= middle + window_size]
    points = list(zip(xs, ys))
    groups = [(x, points_in_window(x, points)) for x in range(0, 501, 10)]
    xs = [x for x, _ in groups]
    means = [np.mean([y for _, y in x]) for _, x in groups]
    stds = [np.std([y for _, y in x]) for _, x in groups]
    return xs, means, stds


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=pathlib.Path, required=True)
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--dataset', default='test')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--ylabel', action='store_true', default=False)
    args = parser.parse_args()

    architectures = ['transformer','lstm', 'rnn']
    setup = 'validation-long'
    losses = ['rec', 'rec+lm', 'rec+ns', 'rec+lm+ns']
    runs = [str(i) for i in range(1, 11)]
    colors = ['blue', 'green', 'red']
    architecture_abrev = ['Tf', 'LSTM', 'RNN']

    dataset = args.dataset
    if dataset == 'training':
        data_directory = args.training_data
    else:
        data_directory = args.training_data / 'datasets' / dataset
    strings_file = data_directory / 'main.tok'

    png_output = args.output.with_suffix('.png')
    plt.rcParams.update({'font.size' : 18})
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    ax.set_ylabel('Recognition Cross-Entropy $\\leftarrow$')
    ax.set_xlabel('Input Length')
    ax.set_ylim(bottom=0)

    xs, ys, stds = [], [], []

    for i in range(len(architectures)):
        architecture = architectures[i]
        acc_by_model = []

        for loss in losses:
            for run in runs:
                model = pathlib.Path('./models') / args.language / architecture / loss / setup / run
                log_file = model / 'eval' / 'test.json'
                acc = get_test_accuracy(log_file)
                acc_by_model.append((model, acc))

        acc_by_model.sort(key=lambda x: x[1], reverse=True)
        model, _ = acc_by_model[0]

        scores_file = model / 'eval' / f'{dataset}.jsonl'
        with strings_file.open() as strings_fin, \
                scores_file.open() as scores_fin:
            data = list(load_lengths_and_scores(strings_fin, scores_fin))

        x = [n for n, s in data]
        y = [divide(s['recognition_cross_entropy']) for n, s in data]
        x, y, std = get_smoothed_points(x, y)
        xs.append(x)
        ys.append(y)
        stds.append(std)
        ax.plot(x, y, color=colors[i], label="{}".format(architecture))
        ax.fill_between(x, np.array(y)-np.array(std), np.array(y)+np.array(std), color=colors[i], alpha=.3)

    ax.axhline(y=math.log(2), color='black', ls='--')
    ax.axvline(x=40, color='black', ls='--')
    ax.axvline(x=80, color='black', ls='--')

    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()
    print(f'writing {png_output}')
    fig.savefig(png_output)

    tex_output = args.output.with_suffix('.tex')
    dat_outputs = []

    for i in range(len(architectures)):
        dat_output = pathlib.Path(f'{args.output}-{architectures[i]}').with_suffix('.dat')
        dat_outputs.append(dat_output)
        print(f'writing {dat_output}')
        with dat_output.open('w') as fout:
            for xi, yi, zi in zip(xs[i], ys[i], stds[i]):
                print(f'{xi} {yi} {yi-zi} {yi+zi}', file=fout)
            print('', file=fout)
            print('', file=fout)
    print(f'writing {tex_output}')
    with tex_output.open('w') as fout:
        fout.write(
r'''\begin{tikzpicture}
\def\threshold{''')
        fout.write(str(math.log(2)))
        fout.write(r'''}
\def\thresholdsmallval{''')
        fout.write(str(40))
        fout.write(r'''}
\def\thresholdlongval{''')
        fout.write(str(80))
        fout.write(r'''}
\begin{axis}[
    legend style={at={(1,1)}, anchor=north east, font=\tiny, opacity=0.6, text opacity=1, draw=gray},
    axis lines=left,
    xlabel={Input Length},
    xmin=0,''')
        fout.write(r'''
    ylabel={Cross-Entropy $\leftarrow$},
    title={\language''')
        match str(args.language):
            case 'repeat-01':
                title = 'repeatzeroone'
            case 'dyck-2-3':
                title = 'dycktwothree'
            case _:
                title = "".join(str(args.language).split('-'))
        fout.write(title)
        fout.write(r'''{}},
    ymin=0,
    enlarge y limits=0.1,
    xtick={0, 100, 200, 300, 400, 500},
    tick label style={font=\tiny}
]''')
        for i in range(len(architectures)):
            fout.write(r'''
    \addplot[
        name path=A-''')
            fout.write(architecture_abrev[i])
            fout.write(r''',
        color=''')
            fout.write(colors[i])
            fout.write(r''',
        opacity=0.2,
        forget plot
    ] table[x index=0, y index=2] {figures/cross-entropy-vs-length/''')
            fout.write(args.language.name)
            fout.write(r'''/''')
            fout.write(dat_outputs[i].name)
            fout.write(r'''};     
    \addplot[
        name path=B-''')
            fout.write(architecture_abrev[i])
            fout.write(r''',
        color=''')
            fout.write(colors[i])
            fout.write(r''',
        opacity=0.2,
        forget plot
    ] table[x index=0, y index=3] {figures/cross-entropy-vs-length/''')
            fout.write(args.language.name)
            fout.write(r'''/''')
            fout.write(dat_outputs[i].name)
            fout.write(r'''};   
    \addplot[color=''')
            fout.write(colors[i])
            fout.write(r''',
        opacity=0.2, forget plot] fill between [of=A-''')
            fout.write(architecture_abrev[i])
            fout.write(r''' and B-''')
            fout.write(architecture_abrev[i])
            fout.write(r'''];
    \addplot[
        color=''')
            fout.write(colors[i])
            fout.write(r'''
    ] table[x index=0, y index=1] {figures/cross-entropy-vs-length/''')
            fout.write(args.language.name)
            fout.write(r'''/''')
            fout.write(dat_outputs[i].name)
            fout.write(r'''};''')
        fout.write(r'''
    \draw[dashed] (axis cs:\pgfkeysvalueof{/pgfplots/xmin}, \threshold) -- (axis cs:\pgfkeysvalueof{/pgfplots/xmax}, \threshold);
    \draw[dashed] (axis cs:\thresholdsmallval, \pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:\thresholdsmallval, \pgfkeysvalueof{/pgfplots/ymax});
    \draw[dashed] (axis cs:\thresholdlongval, \pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:\thresholdlongval, \pgfkeysvalueof{/pgfplots/ymax});
    \node[above left] at (axis cs:\pgfkeysvalueof{/pgfplots/xmax}, \threshold) {\small $\uparrow$ Incorrect};
\end{axis}
\end{tikzpicture}
''')

if __name__ == '__main__':
    main()