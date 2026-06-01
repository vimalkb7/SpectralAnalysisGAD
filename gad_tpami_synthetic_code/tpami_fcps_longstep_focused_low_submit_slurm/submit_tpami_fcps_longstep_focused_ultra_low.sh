#!/bin/bash
set -euo pipefail

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused
mkdir -p "${RESULT_ROOT}/logs/slurm"

RUN_JOB=$(sbatch --parsable slurm/run_tpami_fcps_longstep_focused_6job_array.sbatch)
echo "Submitted ultra-low-submit focused FCPS sweep array: ${RUN_JOB}"

AGG_JOB=$(sbatch --parsable --dependency=afterany:${RUN_JOB} slurm/aggregate_tpami_fcps_longstep_focused.sbatch)
echo "Submitted aggregation job: ${AGG_JOB}"
