#!/bin/bash
set -euo pipefail

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep
mkdir -p "${RESULT_ROOT}/logs/slurm"

RUN_JOB=$(sbatch --parsable slurm/run_tpami_fcps_longstep_hp_sweep_array.sbatch)
echo "Submitted FCPS long-step HP sweep array: ${RUN_JOB}"

AGG_JOB=$(sbatch --parsable --dependency=afterok:${RUN_JOB} slurm/aggregate_tpami_fcps_longstep_hp_sweep.sbatch)
echo "Submitted aggregation job: ${AGG_JOB}"
