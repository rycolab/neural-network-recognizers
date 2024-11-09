import argparse
import json
import math
import pathlib

from recognizers.analysis.plot_cross_entropy_vs_num_edits import (
    load_labels,
    load_negative_score_lines,
    divide
)

def load_edit_distances(fin):
    for line in fin:
        yield int(line.strip())

def load_edit_distances_and_scores(labels_fin, edit_distances_fin, scores_fin):
    negative_score_lines = load_negative_score_lines(labels_fin, scores_fin)
    edit_distances_list = load_edit_distances(edit_distances_fin)
    for score_line, edit_distance in zip(negative_score_lines, edit_distances_list, strict=True):
        yield edit_distance, json.loads(score_line)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--training-data', type=pathlib.Path, required=True,
        help='A directory containing training data. The file '
             '<training-data>/datasets/<input>/main.prepared will be used as '
             'input, and the file '
             '<training-data>/main.vocab will be used as the vocabulary.')
    parser.add_argument('--dataset', default='test-edit-distance')
    parser.add_argument('--output', type=pathlib.Path)
    parser.add_argument('--tex-output', type=pathlib.Path)
    parser.add_argument('--ylabel', action='store_true', default=False)
    args = parser.parse_args()

    dataset = args.dataset
    if dataset == 'training':
        data_directory = args.training_data
    else:
        data_directory = args.training_data / 'datasets' / dataset
    labels_file = data_directory / 'labels.txt'
    edit_distances_file = data_directory / 'edit-distances.txt'
    scores_file = args.model / 'eval' / f'{dataset}.jsonl'
    with labels_file.open() as labels_fin, \
         edit_distances_file.open() as edit_distances_fin, \
         scores_file.open() as scores_fin:
         data = list(load_edit_distances_and_scores(labels_fin, edit_distances_fin, scores_fin))
    x = [n for n, s in data]
    y = [divide(s['recognition_cross_entropy']) for n, s in data]

    if args.output is not None:
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size' : 18})
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 5)
        ax.set_ylabel('Recognition Cross-Entropy $\\leftarrow$')
        ax.set_xlabel('Edit Distance')
        #ax.set_xlim(0,5)
        ax.plot(x, y, 'o', alpha=0.5, color='b', markersize=10)
        plt.tight_layout()
        print(f'writing {args.output}')
        fig.savefig(args.output)
    if args.tex_output:
        dat_output = args.tex_output.with_suffix('.dat')
        tex_output = args.tex_output.with_suffix('.tex')
        print(f'writing {dat_output}')
        with dat_output.open('w') as fout:
            print('editdistance crossentropy', file=fout)
            for xi, yi in zip(x, y):
                print(f'{xi} {yi}', file=fout)
        print(f'writing {tex_output}')
        with tex_output.open('w') as fout:
            fout.write(
r'''\begin{tikzpicture}
    \def\threshold{''')
            fout.write(str(math.log(2)))
            fout.write(r'''}
    \begin{axis}[
        axis lines=left,
        xlabel={Edit Distance},
        xmin=0,
        enlarge x limits=0.1,''')
            if args.ylabel:
                fout.write(r'''
        ylabel={Cross-Entropy $\leftarrow$},''')
            fout.write(r'''
        ymin=0,
        enlarge y limits=0.1,
    ]
        \addplot[
            mark=*,
            mark options={scale=0.5},
            only marks
        ] table {figures/edit-distance/''')
            fout.write(dat_output.name)
            fout.write(r'''};
        \draw[dashed] (axis cs:\pgfkeysvalueof{/pgfplots/xmin}, \threshold) -- (axis cs:\pgfkeysvalueof{/pgfplots/xmax}, \threshold);
        \node[above left] at (axis cs:\pgfkeysvalueof{/pgfplots/xmax}, \threshold) {$\uparrow$ Incorrect};
    \end{axis}
\end{tikzpicture}
''')

if __name__ == '__main__':
    main()
