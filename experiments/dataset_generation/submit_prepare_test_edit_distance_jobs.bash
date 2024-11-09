set -euo pipefail
. experiments/include.bash

device=gpu

for language in repeat-01 dyck-2-3; do
  submit_job \
    prepare-edit-distance+"$language" \
    "$device" \
    --time=1:00:00 \
    --mem-per-cpu=10G \
    --gres=gpumem:10g \
    -- \
    bash recognizers/string_sampling/prepare_test_edit_distance.bash \
      "$BASE_DIR" \
      "$language" \
      "$device"
done
