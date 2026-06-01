#!/bin/bash
#SBATCH --job-name=tpami_syn_agg
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/logs/slurm/aggregate_%j.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/logs/slurm/aggregate_%j.err

set -euo pipefail

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

module purge || true
source ~/linuxhome/venvs/gad_real/bin/activate

CODE_DIR=/home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1
AGG_DIR=${RESULT_ROOT}/aggregate

mkdir -p "${RESULT_ROOT}/logs/slurm" "${AGG_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}"

echo "Aggregating TPAMI synthetic Exp1 results from ${RESULT_ROOT}"
FOUND=$(find "${RESULT_ROOT}" -name results_summary.csv | wc -l | tr -d ' ')
echo "Found results_summary.csv files: ${FOUND} / expected 20"
if [ "${FOUND}" -lt 20 ]; then
  echo "ERROR: Expected 20 result files before aggregation. Some array jobs failed or are still running."
  find "${RESULT_ROOT}" -name results_summary.csv | sort
  exit 1
fi

python -u aggregate_synthetic_results.py \
  --root "${RESULT_ROOT}" \
  --out "${AGG_DIR}"

echo "Aggregate outputs:"
ls -lh "${AGG_DIR}"

echo "Key files for paper:"
echo "  ${AGG_DIR}/paper_generation_table_em_100_200_300_mean_std.csv"
echo "  ${AGG_DIR}/aggregate_forward_stiffness_mean_std.csv"
echo "  ${AGG_DIR}/sampling_efficiency_stationary_gaussian_global_swd_mean.png"
echo "  ${AGG_DIR}/sampling_efficiency_block_covariance_global_swd_mean.png"
