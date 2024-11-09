set -euo pipefail
. experiments/include.bash

cd src
. recognizers/functions.bash

usage() {
  echo "Usage: $0 <language> <architecture>"
}

language=${1-}
architecture=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

models=()
for loss_terms in "${LOSS_TERMS[@]}"; do
  for trial_no in "${TRIALS[@]}"; do
    models+=("$(get_model_dir "$BASE_DIR" "$language" "$architecture" "$loss_terms" validation-long "$trial_no")")
  done
done
best_model=$(python recognizers/analysis/print_best_model.py "${models[@]}")
python recognizers/analysis/sort_examples_by_difficulty.py \
  --model "$best_model" \
  --training-data "$(get_language_dir "$BASE_DIR" "$language")" \
  --dataset test
