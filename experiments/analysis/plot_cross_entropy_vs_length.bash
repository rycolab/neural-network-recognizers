set -euo pipefail
. experiments/include.bash

cd src
. recognizers/functions.bash

mkdir -p "$FIGURES_DIR"/cross-entropy-vs-length
for langauge in "${LANGUAGES[@]}"; do
  python recognizers/analysis/plot_cross_entropy_vs_length.py \
    --language "$language" \
    --training-data "$(get_language_dir "$BASE_DIR" "$language")" \
    --output "$FIGURES_DIR"/cross-entropy-vs-length/"$language"
done
