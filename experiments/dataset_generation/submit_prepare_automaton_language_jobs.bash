set -euo pipefail
. experiments/include.bash

device=cpu

for language in "${FINITE_AUTOMATON_LANGUAGES[@]}"; do
  submit_job \
    prepare+"$language" \
    "$device" \
    --time=1:00:00 \
    --mem-per-cpu=8G \
    -- \
    bash recognizers/string_sampling/prepare_automaton_dataset.bash \
      "$BASE_DIR" \
      "$language" \
      "$device"
done
