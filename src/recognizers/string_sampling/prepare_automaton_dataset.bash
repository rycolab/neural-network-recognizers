set -euo pipefail

. recognizers/functions.bash

usage() {
  echo "Usage: $0 <base-directory> <language> <device>

Prepare the datasets for a language defined with an automaton.

  <base-directory>
    Directory under which all datasets and models are stored.
  <language>
    Name of the language to prepare. Corresponds to the name of a directory
    under <base-directory>/languages/.
  <device>
    Either 'cpu' or 'gpu'.
"
}

base_dir=${1-}
language=${2-}
device=${3-}
if ! shift 3; then
  usage >&2
  exit 1
fi

case $device in
  cpu) ;;
  gpu) device=cuda ;;
  *)
    usage >&2
    exit 1
    ;;
esac

language_dir=$(get_language_dir "$base_dir" "$language")

automaton=$language_dir/automaton.pt
sampler=$language_dir/sampler.pt

mkdir -p "$language_dir"
echo "writing $automaton"
python recognizers/hand_picked_languages/save_automaton.py \
  --name "$language" \
  --output "$automaton"
echo "writing $sampler"
python recognizers/string_sampling/prepare_sampler.py \
  --input "$automaton" \
  --output "$sampler" \
  --max-length 500 \
  --device "$device"
python recognizers/string_sampling/sample_dataset.py \
  --output "$language_dir" \
  --random-seed 123456789 \
  --sampler "$sampler" \
  --include-log-probability \
  --include-next-symbols \
  --skip-test-edit-distance \
  --device "$device"
bash recognizers/neural_networks/prepare_language.bash "$base_dir" "$language"
