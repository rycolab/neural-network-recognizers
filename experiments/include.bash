SINGULARITY_IMAGE_FILE=neural-network-recognizers.sif
BASE_DIR=data
FIGURES_DIR=$BASE_DIR/figures

FINITE_AUTOMATON_LANGUAGES=( \
  even-pairs \
  repeat-01 \
  parity \
  cycle-navigation \
  modular-arithmetic-simple \
  dyck-2-3 \
  first \
)
HAND_WRITTEN_LANGUAGES=( \
  majority \
  stack-manipulation \
  marked-reversal \
  unmarked-reversal \
  marked-copy \
  missing-duplicate-string \
  odds-first \
  binary-addition \
  binary-multiplication \
  compute-sqrt \
  bucket-sort \
)
LANGUAGES=( \
  "${FINITE_AUTOMATON_LANGUAGES[@]}" \
  "${HAND_WRITTEN_LANGUAGES[@]}"
)
ARCHITECTURES=(transformer rnn lstm)
LOSS_TERMS=(rec rec+lm rec+ns rec+lm+ns)
VALIDATION_SETS=(validation-{short,long})
TRIALS=({1..10})

submit_job() {
  bash experiments/submit_job.bash "$@"
}

to_latex_id() {
  sed '
    s/[-_]//g;
    s/0/zero/g;
    s/1/one/;
    s/2/two/;
    s/3/three/;
    s/4/four/;
    s/5/five/;
    s/6/six/;
    s/7/seven/;
    s/8/eight/;
    s/9/nine/;
  ' <<<"$1"
}

format_language_name() {
  echo "\\language$(to_latex_id "$1"){}"
}
