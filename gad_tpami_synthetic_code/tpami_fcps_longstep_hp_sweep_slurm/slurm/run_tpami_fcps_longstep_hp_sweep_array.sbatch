#!/bin/bash
#SBATCH --job-name=tpami_fcps_longhp
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-269%12
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/logs/slurm/train_%A_%a.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/logs/slurm/train_%A_%a.err

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
RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep

# Focused scalar-scheduler hyperparameter search.
# Each job trains one FCPS-GNN and evaluates EM sampling at 1500, 2000, and 2500 steps.
DATASETS=(stationary_gaussian block_covariance)
SEEDS=(0 1 2)
C0_VALUES=(300 700 1200 1800 2300)
ALPHA_VALUES=(2.0 4.0 6.0)
CMIN_VALUES=(0.01 0.05 0.1)

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_CMIN=${#CMIN_VALUES[@]}
NUM_ALPHA=${#ALPHA_VALUES[@]}
NUM_C0=${#C0_VALUES[@]}
NUM_SEEDS=${#SEEDS[@]}

CMIN_ID=$((TASK_ID % NUM_CMIN))
ALPHA_ID=$(((TASK_ID / NUM_CMIN) % NUM_ALPHA))
C0_ID=$(((TASK_ID / (NUM_CMIN * NUM_ALPHA)) % NUM_C0))
SEED_ID=$(((TASK_ID / (NUM_CMIN * NUM_ALPHA * NUM_C0)) % NUM_SEEDS))
DATASET_ID=$((TASK_ID / (NUM_CMIN * NUM_ALPHA * NUM_C0 * NUM_SEEDS)))

DATASET=${DATASETS[$DATASET_ID]}
SEED=${SEEDS[$SEED_ID]}
C0=${C0_VALUES[$C0_ID]}
ALPHA=${ALPHA_VALUES[$ALPHA_ID]}
CMIN=${CMIN_VALUES[$CMIN_ID]}

OUT_DIR=${RESULT_ROOT}/${DATASET}/c0_${C0}/alpha_${ALPHA}/cmin_${CMIN}/seed${SEED}
RUN_LOG_DIR=${OUT_DIR}/logs
mkdir -p "${RESULT_ROOT}/logs/slurm" "${RUN_LOG_DIR}" "${OUT_DIR}/slurm"

test -d "${CODE_DIR}" || { echo "ERROR: missing CODE_DIR=${CODE_DIR}"; exit 1; }
test -f "${CODE_DIR}/run_synthetic_scheduler_sampler_compare.py" || { echo "ERROR: missing synthetic runner"; exit 1; }
cp "$0" "${OUT_DIR}/slurm/slurm_script" || true

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
cd "${CODE_DIR}"

echo "============================================================"
echo "TPAMI FCPS long-step hyperparameter sweep"
echo "Task ID     : ${TASK_ID}"
echo "Dataset     : ${DATASET}"
echo "Seed        : ${SEED}"
echo "FCPS c0     : ${C0}"
echo "FCPS alpha  : ${ALPHA}"
echo "FCPS cmin   : ${CMIN}"
echo "Steps       : 1500,2000,2500"
echo "Val steps   : 1500"
echo "Output dir  : ${OUT_DIR}"
echo "Started at  : $(date)"
echo "Host        : $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"
nvidia-smi || true

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

echo "Completed at: $(date)"
echo "Results written to ${OUT_DIR}"
