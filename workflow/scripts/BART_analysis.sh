#!/usr/bin/env bash
set -eEo pipefail  # no -u here yet

ENV_NAME="r-pipeline"
R_SCRIPT="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/workflow/scripts/BART_analysis.R"
OUT_FILE="/mnt/c/Users/u0155664/OneDrive - KU Leuven/phd/1_projects/pregnancy_outcome/analysis/bart_data/bart_data.out"

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

# temporarily allow unset vars during activation
set +u
conda activate "$ENV_NAME"
set -u  # re-enable nounset for the rest

# (Optional) confirm which R
which Rscript >&2

Rscript --vanilla "${R_SCRIPT}" "$@" 

touch "${OUT_FILE}"