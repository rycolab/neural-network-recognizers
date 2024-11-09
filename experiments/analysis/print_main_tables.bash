set -euo pipefail
. experiments/include.bash

cd src
. recognizers/functions.bash

mkdir -p "$FIGURES_DIR"

args=()
for language in "${LANGUAGES[@]}"; do
  args+=( \
    --language "$(format_language_name "$language")" \
    "$FIGURES_DIR"/full-tables/"$language".tex \
  )
done

python recognizers/analysis/print_main_table.py \
  "${args[@]}" \
  --main-output "$FIGURES_DIR"/main-table.tex \
  --loss-output "$FIGURES_DIR"/loss-function-table.tex
