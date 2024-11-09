set -euo pipefail
. experiments/include.bash

for language in "${LANGUAGES[@]}"; do
  for architecture in "${ARCHITECTURES[@]}"; do
    for loss_terms in "${LOSS_TERMS[@]}"; do
      for validation_data in "${VALIDATION_SETS[@]}"; do
        for trial_no in "${TRIALS[@]}"; do
          submit_job \
            train+"$language"+"$architecture"+"${loss_terms//+/_}"+"$validation_data"+"$trial_no" \
            cpu \
            --time=4:00:00 \
            -- \
            bash recognizers/neural_networks/train_and_evaluate.bash \
              "$BASE_DIR" \
              "$language" \
              "$architecture" \
              "$loss_terms" \
              "$validation_data" \
              "$trial_no" \
              --no-progress
        done
      done
    done
  done
done
