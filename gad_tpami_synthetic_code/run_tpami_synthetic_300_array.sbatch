#!/bin/bash
#SBATCH --job-name=tpami_syn300
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/tpami_syn300_%A_%a.out
#SBATCH --error=logs/tpami_syn300_%A_%a.err

set -euo pipefail

# Edit these two paths for your cluster.
CODE_DIR="${CODE_DIR:-$PWD}"
RESULT_ROOT="${RESULT_ROOT:-$PWD/tpami_synthetic_results_300}"

mkdir -p "$RESULT_ROOT" logs
cd "$CODE_DIR"

DATASETS=(stationary_gaussian block_covariance)
SEEDS=(0 1 2 3 4)

DATASET_ID=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_ID=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
DATASET="${DATASETS[$DATASET_ID]}"
SEED="${SEEDS[$SEED_ID]}"
OUT_DIR="$RESULT_ROOT/${DATASET}/seed${SEED}"

python run_synthetic_scheduler_sampler_compare.py \
  --out "$OUT_DIR" \
  --seed "$SEED" \
  --device cuda \
  --dtype float32 \
  --n 128 \
  --k 8 \
  --dataset_family "$DATASET" \
  --num_train 10000 \
  --num_val 2000 \
  --num_test 2000 \
  --schedulers FCPS,MAS-SNR \
  --methods em \
  --steps 25,50,100,200,300 \
  --val_method em \
  --val_steps 100 \
  --val_num_samples 1024 \
  --metric_num_projections 128 \
  --metric_num_seeds 5 \
  --num_epochs 80 \
  --batch_size 128 \
  --hidden_dim 64 \
  --num_layers 4 \
  --cheb_order 3 \
  --lr 3e-4 \
  --min_lr 1e-6 \
  --lr_scheduler plateau \
  --train_loss x0 \
  --train_t_min 1e-2 \
  --match_terminal_energy
