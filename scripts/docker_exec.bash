set -euo pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

usage() {
  echo "Usage: $0 [options]

Open a shell in the Docker container, optionally building the image first.

Options:
  --build   Build the Docker image from scratch first.
  --cpu     Run in CPU-only mode.
"
}

get_options=()
if [[ -f /etc/NIXOS ]]; then
  start_options=(--device=nvidia.com/gpu=all)
else
  start_options=(--gpus all --privileged)
fi
while [[ $# -gt 0 ]]; do
  case $1 in
    --build) get_options+=("$1") ;;
    --cpu) start_options=() ;;
    --) shift; break ;;
    *) usage >&2; exit 1 ;;
  esac
  shift
done

bash scripts/get_docker_dev_image.bash "${get_options[@]}"
dockerdev_ensure_dev_container_started "$DOCKER_DEV_IMAGE" \
  --x11 \
  -- \
  -v "$PWD":/app/ \
  --mount type=bind,source="$HOME"/.ssh/,destination=/home/"$USER"/.ssh/ \
  "${start_options[@]}"
dockerdev_run_in_dev_container "$DOCKER_DEV_IMAGE" "$@"
