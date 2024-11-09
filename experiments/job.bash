#!/bin/bash
set -euo pipefail

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

# Show some information about the job.
echo "Arguments:"
escaped_args=()
for arg in "$@"; do
  escaped_args+=("$(printf %q "$arg")")
done
printf '%s\n' "${escaped_args[*]}"
echo "Timestamp: $(date)"
echo "Host: $(hostname)"
case $device in
  cpu) ;;
  gpu)
    echo "GPUs:"
    nvidia-smi --query-gpu=name --format=csv | tail -n +2
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo 'Output of nvidia-smi:'
    nvidia-smi
    echo 'Users:'
    users
    echo 'Process tree:'
    ps -e f
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

echo '==== Start of job ===='
bash experiments/singularity_exec.bash "$device" bash -c 'cd src && exec poetry run -- "$@"' -- "$@"
