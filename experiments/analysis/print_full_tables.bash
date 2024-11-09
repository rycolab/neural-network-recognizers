set -euo pipefail
. experiments/include.bash

cd src
. recognizers/functions.bash

mkdir -p "$FIGURES_DIR"

print_refs() {
  echo -n '\cref{'
  local first
  first=true
  for language in "${LANGUAGES[@]}"; do
    if $first; then
      first=false
    else
      echo -n ','
    fi
    echo -n "tab:full-$language"
  done
  echo -n '}'
}

print_includes() {
  for language in "${LANGUAGES[@]}"; do
    echo "\
\\begin{table}
    \\caption{Full results on the \\textbf{$(format_language_name "$language")} language.}
    \\label{tab:full-$language}
    \\begin{center}
        \\small
        \\input{figures/full-tables/$language.tex}
    \\end{center}
\\end{table}"
  done
}

refs_file=$FIGURES_DIR/full-tables-refs.tex
echo "writing $refs_file"
print_refs > "$refs_file"

includes_file=$BASE_DIR/figures/full-tables-include.tex
echo "writing $includes_file"
print_includes > "$includes_file"

format_row_name() {
  local architecture=$1
  local loss_terms=$2
  local validation_data=$3
  case $architecture in
    transformer) a='\transformerabbrev{}' ;;
    rnn) a='\rnnabbrev{}' ;;
    lstm) a='\lstmabbrev{}' ;;
    *) return 1 ;;
  esac
  case $loss_terms in
    rec) l='' ;;
    rec+lm) l='+\languagemodelingabbrev{}, ' ;;
    rec+ns) l='+\nextsymbolsabbrev{}, ' ;;
    rec+lm+ns) l='+\languagemodelingabbrev{}+\nextsymbolsabbrev{}, ' ;;
    *) return 1 ;;
  esac
  case $validation_data in
    validation-short) v='\validationshortabbrev{}' ;;
    validation-long) v='\validationlongabbrev{}' ;;
    *) return 1 ;;
  esac
  echo -n "$a ($l$v)"
}

OUTPUT_DIR=$FIGURES_DIR/full-tables
mkdir -p "$OUTPUT_DIR"
for language in "${LANGUAGES[@]}"; do
  output_file=$OUTPUT_DIR/$language.tex
  echo "writing $output_file"
  args=()
  for architecture in "${ARCHITECTURES[@]}"; do
    for loss_terms in "${LOSS_TERMS[@]}"; do
      for validation_data in "${VALIDATION_SETS[@]}"; do
        args+=(--label "$(format_row_name "$architecture" "$loss_terms" "$validation_data")" --inputs)
        for trial_no in "${TRIALS[@]}"; do
          args+=("$(get_model_dir "$BASE_DIR" "$language" "$architecture" "$loss_terms" "$validation_data" "$trial_no")")
        done
      done
    done
  done
  python recognizers/analysis/print_table.py "${args[@]}" > "$output_file"
done
