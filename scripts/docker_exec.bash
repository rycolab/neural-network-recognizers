set -euo pipefail

. scripts/dockerdev.bash
. scripts/variables.bash

usage() {
  echo "Usage: $0 [options]

Open a shell in the Docker container, optionally pulling or building the
image first.

Options:
  --pull    Pull the public Docker image first.
  --build   Build the Docker image from scratch first.
  --cpu     Run in CPU-only mode.
"
}

get_options=()
start_options=(--gpus all --privileged)
while [[ $# -gt 0 ]]; do
  case $1 in
    --pull|--build) get_options+=("$1") ;;
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
  --mount type=bind,source="$HOME"/.ssh/id_rsa,destination=/home/"$USER"/.ssh/id_rsa \
  "${start_options[@]}"
dockerdev_run_in_dev_container "$DOCKER_DEV_IMAGE" "$@"
