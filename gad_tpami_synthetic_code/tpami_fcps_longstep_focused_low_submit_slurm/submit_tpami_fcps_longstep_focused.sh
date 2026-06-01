#!/bin/bash
set -euo pipefail

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused
mkdir -p "${RESULT_ROOT}/logs/slurm"

# Use this 12-task version first. It should avoid AssocMaxSubmitJobLimit on most accounts.
RUN_JOB=$(sbatch --parsable slurm/run_tpami_fcps_longstep_focused_12job_array.sbatch)
echo "Submitted focused FCPS long-step sweep array: ${RUN_JOB}"

# afterany is intentional: aggregation should run even if some candidates fail internally.
AGG_JOB=$(sbatch --parsable --dependency=afterany:${RUN_JOB} slurm/aggregate_tpami_fcps_longstep_focused.sbatch)
echo "Submitted aggregation job: ${AGG_JOB}"
