#!/usr/bin/env bash
set -euo pipefail

activate_repro_env() {
    local want_cuda="${1:-no}"
    if command -v module >/dev/null 2>&1; then
        module purge || true
        if [[ "${want_cuda}" == "yes" && -n "${CUDA_MODULE:-}" ]]; then
            module load "${CUDA_MODULE}"
        fi
    fi
    unset PYTHONHOME
    if [[ -n "${VENV_ACTIVATE:-}" ]]; then
        [[ -f "${VENV_ACTIVATE}" ]] || {
            echo "ERROR: VENV_ACTIVATE does not exist: ${VENV_ACTIVATE}" >&2
            exit 11
        }
        # shellcheck disable=SC1090
        source "${VENV_ACTIVATE}"
    fi
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MPLBACKEND=Agg
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

