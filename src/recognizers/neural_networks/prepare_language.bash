set -euo pipefail

. recognizers/functions.bash

usage() {
  echo "Usage: $0 <base-directory> <language>

Prepare the datasets for a language.

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

# Optional datasets.
flags=()
for dataset in test-short-held-out test-edit-distance; do
  if [[ -e $language_dir/datasets/$dataset ]]; then
    flags+=(--more-data "$dataset")
  fi
done

python recognizers/neural_networks/prepare_data.py \
  --training-data "$language_dir" \
  --more-data validation-short \
  --more-data validation-long \
  --more-data test \
  "${flags[@]}" \
  --never-allow-unk \
  --use-next-symbols
