#!/bin/bash
#SBATCH --job-name=tpami_syn_pre
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/logs/slurm/preflight_%j.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/logs/slurm/preflight_%j.err

set -euo pipefail

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

module purge || true
source ~/linuxhome/venvs/gad_real/bin/activate

CODE_DIR=/home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1

mkdir -p "${RESULT_ROOT}/logs/slurm" "${RESULT_ROOT}/preflight" "${RESULT_ROOT}/aggregate"

test -d "${CODE_DIR}" || { echo "ERROR: missing CODE_DIR=${CODE_DIR}"; exit 1; }
test -f "${CODE_DIR}/run_synthetic_scheduler_sampler_compare.py" || { echo "ERROR: missing run_synthetic_scheduler_sampler_compare.py"; exit 1; }
test -f "${CODE_DIR}/aggregate_synthetic_results.py" || { echo "ERROR: missing aggregate_synthetic_results.py"; exit 1; }

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}"

python - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

required = ["torch", "numpy", "pandas", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")

import torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

code = Path("/home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code")
result = Path("/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1")
print("CODE_DIR:", code)
print("RESULT_ROOT:", result)
print("Expected array jobs: 20 = 2 datasets x 5 seeds x 2 schedulers")
print("Datasets: stationary_gaussian, block_covariance")
print("Schedulers: FCPS, MAS-SNR")
print("Seeds: 0,1,2,3,4")
print("Sampler: GNN + Euler-Maruyama only")
print("Reverse steps: 25,50,100,200,300")
PY

python run_synthetic_scheduler_sampler_compare.py --help >/dev/null
python aggregate_synthetic_results.py --help >/dev/null

echo "Preflight OK."
