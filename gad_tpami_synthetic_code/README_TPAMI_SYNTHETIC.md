# TPAMI synthetic MAS-SNR experiments up to 300 reverse steps

This folder contains the patched synthetic experiment code for the concise TPAMI synthetic section.

## What changed

1. Fixed the training-time mismatch bug: the same diffusion time `t` is now used to corrupt `x0 -> xt` and to condition the GNN denoiser.
2. Main defaults are now GNN + Euler--Maruyama only: `--methods em` and `--steps 25,50,100,200,300`.
3. Added forward diagnostics for the theory: SNR dispersion, effective-variance/rho dispersion, terminal-prior match, and stiffness proxy.
4. Added robust stability columns for generation: finite-sample fraction, non-finite flag, NaN/Inf counts, and max absolute generated value.
5. Added compact paper CSVs and plots:
   - `forward_diagnostics_summary_all.csv`
   - `paper_forward_stiffness_table.csv`
   - `results_summary.csv`
   - `paper_generation_table_em_100_200_300.csv`
   - `quality_vs_steps_global_swd.png`
   - `quality_vs_steps_cov_rel_err.png`
   - `quality_vs_steps_band_ratio_l1.png`

## Single run example

```bash
python run_synthetic_scheduler_sampler_compare.py \
  --out tpami_synthetic_results_300/stationary_gaussian/seed0 \
  --seed 0 \
  --device cuda \
  --dataset_family stationary_gaussian \
  --schedulers FCPS,MAS-SNR \
  --methods em \
  --steps 25,50,100,200,300 \
  --num_train 10000 --num_val 2000 --num_test 2000 \
  --num_epochs 80 --batch_size 128 \
  --val_method em --val_steps 100 --val_num_samples 1024 \
  --match_terminal_energy
```

For the non-stationary synthetic distribution, replace:

```bash
--dataset_family stationary_gaussian
```

with:

```bash
--dataset_family block_covariance
```

## Array run for both datasets and 5 seeds

Edit `CODE_DIR` and `RESULT_ROOT` in `run_tpami_synthetic_300_array.sbatch` if needed, then run:

```bash
sbatch run_tpami_synthetic_300_array.sbatch
```

This launches 10 jobs:

- 2 datasets: `stationary_gaussian`, `block_covariance`
- 5 seeds: `0,1,2,3,4`
- 2 schedulers inside each job: `FCPS`, `MAS-SNR`
- EM sampling steps: `25,50,100,200,300`

## Aggregate after all jobs finish

```bash
python aggregate_synthetic_results.py \
  --root tpami_synthetic_results_300 \
  --out tpami_synthetic_results_300/aggregate
```

Use these files in the paper:

- `aggregate_generation_mean_std.csv`
- `paper_generation_table_em_100_200_300_mean_std.csv`
- `aggregate_forward_stiffness_mean_std.csv`
- `sampling_efficiency_<dataset>_<metric>.png`

## Main-paper recommendation

Use only:

1. Forward SNR/rho/stiffness diagnostic table or heatmap.
2. EM generation table at 100/200/300 steps.
3. Sampling-efficiency plot from 25 to 300 steps.

Keep extra synthetic families as appendix only.
