get_model_dir() {
  local base_dir=$1
  local language=$2
  local architecture=$3
  local loss_terms=$4
  local validation_data=$5
  local trial_no=$6
  printf %s "$base_dir/models/$language/$architecture/$loss_terms/$validation_data/$trial_no"
}

get_language_dir() {
  local base_dir=$1
  local language=$2
  printf %s "$base_dir/languages/$language"
}
