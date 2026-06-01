TPAMI synthetic follow-up: stabilized scalar FCPS c0 sweep.

Runs:
  datasets: stationary_gaussian, block_covariance
  seeds: 0, 1, 2
  FCPS c0: 50, 100, 300, 700
  scheduler: FCPS only
  method: Euler-Maruyama only
  steps: 300 only

Total jobs: 2 datasets * 3 seeds * 4 c0 = 24 GPU jobs.

Usage:
  cd /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
  unzip tpami_fcps_c0_sweep_slurm.zip
  cd tpami_fcps_c0_sweep_slurm
  bash submit_tpami_fcps_c0_sweep.sh

Outputs:
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep/aggregate/fcps_c0_sweep_all_rows.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep/aggregate/fcps_c0_sweep_mean_std.csv
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep/aggregate/fcps_c0_sweep_stability_table.csv
