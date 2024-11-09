set -euo pipefail

. recognizers/functions.bash

usage() {
echo "Usage: $0 <base-directory> <language> <device>

Prepare the datasets for a language.

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
extra_args=("$@")

case $device in
  cpu) ;;
  gpu) device=cuda ;;
  *)
    usage >&2
    exit 1
    ;;
esac

language_dir=$(get_language_dir "$base_dir" "$language")
sampler=$language_dir/sampler.pt
python recognizers/string_sampling/sample_dataset.py \
  --output "$language_dir" \
  --random-seed 123456789 \
  --sampler "$sampler" \
  --include-log-probability \
  --include-next-symbols \
  --skip-standard-datasets \
  --device "$device"
python recognizers/neural_networks/prepare_data.py \
  --training-data "$language_dir" \
  --only-more-data \
  --more-data test-edit-distance \
  --never-allow-unk \
  --use-next-symbols
