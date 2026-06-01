TPAMI synthetic follow-up: FCPS long-step hyperparameter sweep.

Purpose:
  Check whether scalar FCPS becomes stable when Euler-Maruyama uses many more reverse steps.
  This is a reviewer-facing robustness check against the concern that 300-step EM was too coarse.

Runs:
  datasets: stationary_gaussian, block_covariance
  seeds: 0, 1, 2
  FCPS c0: 300, 700, 1200, 1800, 2300
  FCPS alpha: 2.0, 4.0, 6.0
  FCPS cmin: 0.01, 0.05, 0.1
  scheduler: FCPS only
  method: Euler-Maruyama only
  test steps: 1500, 2000, 2500
  validation steps: 1500

Total jobs:
  2 datasets * 3 seeds * 5 c0 * 3 alpha * 3 cmin = 270 GPU jobs.
  Each job evaluates all three step counts, so steps do not multiply the training jobs.

Usage:
  cd /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
  unzip tpami_fcps_longstep_hp_sweep_slurm.zip
  cd tpami_fcps_longstep_hp_sweep_slurm
  bash submit_tpami_fcps_longstep_hp_sweep.sh

Concurrency:
  The array is set to %12. Increase/decrease this in the sbatch header based on GPU quota.

Outputs:
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/aggregate/fcps_longstep_hp_all_rows.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/aggregate/fcps_longstep_hp_mean_std.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/aggregate/fcps_longstep_hp_ranked_candidates.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/aggregate/fcps_longstep_hp_stability_table.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/aggregate/fcps_longstep_hp_top10_stable_per_dataset_step.csv

How to reduce cost:
  For a quick smoke run, edit the sbatch arrays:
    SEEDS=(0)
    C0_VALUES=(700 1200 2300)
    ALPHA_VALUES=(2.0 4.0)
    CMIN_VALUES=(0.01 0.1)
  Then update the #SBATCH --array range accordingly.
