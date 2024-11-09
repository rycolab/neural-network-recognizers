set -euo pipefail

. experiments/include.bash

usage() {
  echo "Usage: $0 <device> <command>...

  <device>    One of: cpu, gpu
"
}

device=${1-}
if ! shift 1; then
  usage >&2
  exit 1
fi

case $device in
  cpu) singularity_flags=() ;;
  gpu) singularity_flags=(--nv) ;;
  *)
    usage >&2
    exit 1
    ;;
esac

mkdir -p "$HOME"/.cache "$HOME"/.ssh
singularity exec \
  "${singularity_flags[@]}" \
  --bind "$HOME"/.cache \
  --bind "$HOME"/.ssh \
  "$SINGULARITY_IMAGE_FILE" \
  "$@"
