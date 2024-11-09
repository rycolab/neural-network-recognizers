set -euo pipefail
. experiments/include.bash

cd src
. recognizers/functions.bash

output_dir=$FIGURES_DIR/edit-distance
mkdir -p "$output_dir"
first=true
for language in repeat-01 dyck-2-3; do
  models=()
  for loss_terms in "${LOSS_TERMS[@]}"; do
    for trial_no in "${TRIALS[@]}"; do
      models+=("$(get_model_dir "$BASE_DIR" "$language" transformer "$loss_terms" validation-long "$trial_no")")
    done
  done
  best_model=$(python recognizers/analysis/print_best_model.py "${models[@]}")
  args=()
  if $first; then
    args+=(--ylabel)
    first=false
  fi
  python recognizers/analysis/plot_cross_entropy_vs_edit_distance.py \
    --model "$best_model" \
    "${args[@]}" \
    --training-data "$(get_language_dir "$BASE_DIR" "$language")" \
    --tex-output "$output_dir"/"$language"
done
