#!/bin/bash
set -euo pipefail

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "${RESULT_ROOT}/logs/slurm" "${RESULT_ROOT}/aggregate" "${RESULT_ROOT}/preflight"

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

PRE=$(sbatch --parsable "${SCRIPT_DIR}/slurm/preflight_tpami_synthetic_exp1.sbatch")
echo "Submitted preflight job: ${PRE}"

ARR=$(sbatch --parsable --dependency=afterok:${PRE} "${SCRIPT_DIR}/slurm/run_tpami_synthetic_exp1_gnn_em_300_array.sbatch")
echo "Submitted synthetic array job: ${ARR}"

echo "The array launches 20 jobs: 2 datasets x 5 seeds x 2 schedulers."
echo "Current array concurrency is %10. Edit #SBATCH --array=0-19%10 to %20 if you want all 20 to start at once and your quota allows it."

AGG=$(sbatch --parsable --dependency=afterok:${ARR} "${SCRIPT_DIR}/slurm/aggregate_tpami_synthetic_exp1.sbatch")
echo "Submitted aggregation job: ${AGG}"

echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  sacct -j ${ARR} --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,ExitCode"
echo ""
echo "Main result folder: ${RESULT_ROOT}"
echo "Aggregated paper outputs will be in: ${RESULT_ROOT}/aggregate"
