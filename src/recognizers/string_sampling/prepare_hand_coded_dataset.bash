set -euo pipefail

. recognizers/functions.bash

usage() {
echo "Usage: $0 <base-directory> <language>

Prepare the datasets for a hand-coded language.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare. Corresponds to the name of a directory
    under <base-directory>/languages/.
"
}

base_dir=${1-}
language=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi

language_dir=$(get_language_dir "$base_dir" "$language")

automaton=$language_dir/automaton.pt
sampler=$language_dir/sampler.pt

mkdir -p "$language_dir"
python recognizers/string_sampling/sample_dataset.py \
  --output "$language_dir" \
  --random-seed 123456789 \
  --language "$language" \
  --include-log-probability \
  --include-next-symbols \
  --skip-test-edit-distance
bash recognizers/neural_networks/prepare_language.bash "$base_dir" "$language"
