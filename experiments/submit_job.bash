set -euo pipefail

. experiments/include.bash

usage() {
  echo "Usage: $0 <job-name> <device> [sbatch-flags...] -- <command>...

  This script does nothing unless the command sbatch is available. This is a
  stub for a script that submits a command as a batch job to your scientific
  computing cluster. You should edit the end of this script so that it runs
  the appropriate job submission commands that are specific to your cluster.

  <job-name>    A string that identifies this job.
  <device>      One of: cpu, gpu.
  sbatch-flags  Any number of flags to pass to sbatch.
                This can include, for example:
                  --gpus=1            (for number of GPUs)
                  --gres=gpumem:2g    (for GPU memory requests)
                  --time=3:00:00      (for time limit requests)
                  --mem-per-cpu=2G    (for amount of RAM)
  <command>     The command to run.
"
}

job_name=${1-}
device=${2-}
if ! shift 2; then
  usage >&2
  exit 1
fi
sbatch_flags=()
while [[ $# -gt 0 && $1 != '--' ]]; do
  sbatch_flags+=("$1")
  shift
done
if [[ $# -eq 0 || $1 != '--' ]]; then
  usage >&2
  echo 'error: missing "--"' >&2
  exit 1
fi
shift
command_args=("$@")

case $device in
  cpu|gpu) ;;
  *)
    usage >&2
    echo "error: device must be one of cpu, gpu; got '$device'" >&2
    exit 1
    ;;
esac

if [[ $device = gpu ]]; then
  has_gpus_flag=false
  has_gres_flag=false
  for arg in "${sbatch_flags[@]}"; do
    case $arg in
      --gpus=*) has_gpus_flag=true ;;
      --gres=*) has_gres_flag=true ;;
    esac
  done
  if ! $has_gpus_flag; then
    sbatch_flags+=(--gpus=1)
  fi
  if ! $has_gres_flag; then
    sbatch_flags+=(--gres=gpumem:2g)
  fi
fi

result=$( \
  sbatch \
    --job-name="$job_name" \
    --output="$BASE_DIR"/job-outputs/"$job_name".txt \
    --open-mode=append \
    "${sbatch_flags[@]}" \
    experiments/job.bash "$device" "${command_args[@]}" \
)
echo "$job_name | $result"
