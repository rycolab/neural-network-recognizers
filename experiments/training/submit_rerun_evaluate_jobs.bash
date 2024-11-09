set -euo pipefail
. experiments/include.bash
. src/recognizers/functions.bash

for language in "${LANGUAGES[@]}"; do
  language_dir=$(get_language_dir "$BASE_DIR" "$language")
  for architecture in "${ARCHITECTURES[@]}"; do
    for loss_terms in "${LOSS_TERMS[@]}"; do
      for validation_data in "${VALIDATION_SETS[@]}"; do
        for trial_no in "${TRIALS[@]}"; do
          model_dir=$(get_model_dir "$BASE_DIR" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no")
          submit_job \
            train+"$architecture"+"$loss_terms"+"$validation_data"+"$trial_no" \
            cpu \
            --time=10:00 \
            -- \
            bash recognizers/neural_networks/evaluate.bash \
              "$language_dir" \
              "$model_dir"
        done
      done
    done
  done
done
