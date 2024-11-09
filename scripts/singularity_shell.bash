set -euo pipefail

. scripts/variables.bash

singularity shell --nv "$SINGULARITY_IMAGE".sif
