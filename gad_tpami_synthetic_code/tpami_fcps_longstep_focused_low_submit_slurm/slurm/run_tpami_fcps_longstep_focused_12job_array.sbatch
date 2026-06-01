#!/bin/bash
#SBATCH --job-name=tpami_fcps_focus
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%4
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused/logs/slurm/train_%A_%a.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused/logs/slurm/train_%A_%a.err

set -euo pipefail

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

module purge || true
module load 2025/gpu || true
module load cuda/12.9 || true
source ~/linuxhome/venvs/gad_real/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

CODE_DIR=/home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused

# Low-submit focused FCPS search.
# Each Slurm task runs 5 FCPS configurations sequentially.
# Total outputs: 2 datasets x 3 seeds x 10 candidates = 60 trained models.
DATASETS=(stationary_gaussian block_covariance)
SEEDS=(0 1 2)
GROUPS=(0 1)

# Candidate grid chosen from prior c0 sweep and FCPS stability logic:
# - lower cmin than old runs to reduce early floor drift
# - alpha 2/3 prioritized because high alpha increases terminal rate
# - c0 covers under-corrupted stable control, intermediate prior match, and stronger terminal match
CAND_C0=(50 100 300 300 300 700 700 700 1200 1200)
CAND_ALPHA=(2.0 2.0 2.0 3.0 4.0 2.0 3.0 4.0 2.0 3.0)
CAND_CMIN=(0.001 0.001 0.001 0.001 0.01 0.001 0.001 0.01 0.001 0.001)

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_GROUPS=${#GROUPS[@]}
NUM_SEEDS=${#SEEDS[@]}
GROUP_ID=$((TASK_ID % NUM_GROUPS))
SEED_ID=$(((TASK_ID / NUM_GROUPS) % NUM_SEEDS))
DATASET_ID=$((TASK_ID / (NUM_GROUPS * NUM_SEEDS)))

DATASET=${DATASETS[$DATASET_ID]}
SEED=${SEEDS[$SEED_ID]}
GROUP=${GROUPS[$GROUP_ID]}

if [[ "$GROUP" == "0" ]]; then
  START=0
  END=4
else
  START=5
  END=9
fi

mkdir -p "${RESULT_ROOT}/logs/slurm"
test -d "${CODE_DIR}" || { echo "ERROR: missing CODE_DIR=${CODE_DIR}"; exit 1; }
test -f "${CODE_DIR}/run_synthetic_scheduler_sampler_compare.py" || { echo "ERROR: missing synthetic runner"; exit 1; }

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}"

echo "============================================================"
echo "TPAMI FCPS focused long-step sweep"
echo "Task ID      : ${TASK_ID}"
echo "Dataset      : ${DATASET}"
echo "Seed         : ${SEED}"
echo "Group        : ${GROUP} candidates ${START}-${END}"
echo "Test steps   : 1500,2000,2500"
echo "Val steps    : 1500"
echo "Result root  : ${RESULT_ROOT}"
echo "Started at   : $(date)"
echo "Host         : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"
nvidia-smi || true

FAILED=0
for CID in $(seq ${START} ${END}); do
  C0=${CAND_C0[$CID]}
  ALPHA=${CAND_ALPHA[$CID]}
  CMIN=${CAND_CMIN[$CID]}
  OUT_DIR=${RESULT_ROOT}/${DATASET}/c0_${C0}/alpha_${ALPHA}/cmin_${CMIN}/seed${SEED}
  RUN_LOG_DIR=${OUT_DIR}/logs
  mkdir -p "${RUN_LOG_DIR}" "${OUT_DIR}/slurm"
  cp "$0" "${OUT_DIR}/slurm/slurm_script" || true

  echo "------------------------------------------------------------"
  echo "Candidate ${CID}: dataset=${DATASET}, seed=${SEED}, c0=${C0}, alpha=${ALPHA}, cmin=${CMIN}"
  echo "Output: ${OUT_DIR}"
  echo "Started candidate at: $(date)"
  echo "------------------------------------------------------------"

  set +e
  srun -u python -u run_synthetic_scheduler_sampler_compare.py \
    --out "${OUT_DIR}" \
    --seed "${SEED}" \
    --device cuda \
    --dtype float32 \
    --n 128 \
    --k 8 \
    --gamma 1e-3 \
    --sigma 1.0 \
    --T 1.0 \
    --dataset_family "${DATASET}" \
    --num_train 10000 \
    --num_val 2000 \
    --num_test 2000 \
    --schedulers FCPS \
    --methods em \
    --steps 1500,2000,2500 \
    --sampler_init_mode algorithm2 \
    --fcps_c0 "${C0}" \
    --fcps_cmin "${CMIN}" \
    --fcps_alpha "${ALPHA}" \
    --val_method em \
    --val_steps 1500 \
    --val_num_samples 512 \
    --metric_num_projections 128 \
    --metric_num_seeds 5 \
    --num_epochs 80 \
    --batch_size 128 \
    --hidden_dim 64 \
    --num_layers 4 \
    --time_embed_dim 64 \
    --cheb_order 3 \
    --dropout 0.0 \
    --lr 3e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.0 \
    --lr_scheduler plateau \
    --lr_decay_factor 0.5 \
    --lr_patience_validations 2 \
    --lr_threshold 1e-4 \
    --val_every_epochs 20 \
    --best_model_metric val_selection_score \
    --train_loss x0 \
    --grad_clip_norm 1.0 \
    --train_t_min 1e-2 \
    --num_diag_times 51 \
    --diag_active_rel_floor 1e-6
  STATUS=$?
  set -e

  if [[ ${STATUS} -ne 0 ]]; then
    echo "Candidate ${CID} FAILED with exit code ${STATUS}" | tee "${OUT_DIR}/FAILED.status"
    FAILED=1
  else
    echo "Candidate ${CID} completed successfully at $(date)" | tee "${OUT_DIR}/COMPLETE.status"
  fi
  echo "------------------------------------------------------------"
done

if [[ ${FAILED} -ne 0 ]]; then
  echo "One or more candidates failed. Keeping array task exit code 0 so aggregation can still run."
fi

echo "Completed grouped task at: $(date)"
