set -euo pipefail

. recognizers/functions.bash

usage() {
  echo "Usage: $0 <language-dir> <model-dir>

Evaluate a saved neural network on a language.
"
}

language_dir=${1-}
model_dir=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
extra_args=("$@")

datasets=(test training)
# Optional datasets.
for dataset in test-short-held-out test-edit-distance; do
  if [[ -e $language_dir/datasets/$dataset ]]; then
    datasets+=("$dataset")
  fi
done

eval_dir=$model_dir/eval
mkdir -p "$eval_dir"
python recognizers/neural_networks/evaluate.py \
  --training-data "$language_dir" \
  --batching-max-tokens 1024 \
  --load-model "$model_dir" \
  --datasets "${datasets[@]}" \
  --output "$eval_dir" \
  "${extra_args[@]}"
