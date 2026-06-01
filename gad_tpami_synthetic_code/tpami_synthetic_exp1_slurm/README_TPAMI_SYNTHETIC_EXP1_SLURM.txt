TPAMI Synthetic Exp1 Slurm package
==================================

Purpose
-------
Run the concise synthetic validation for MAS-SNR up to 300 reverse steps using GNN denoisers only.

This package launches the required main-paper synthetic experiments:

  Datasets   : stationary_gaussian, block_covariance
  Seeds      : 0, 1, 2, 3, 4
  Schedulers : FCPS, MAS-SNR
  Denoiser   : time-conditioned GNN/GCF from run_synthetic_scheduler_sampler_compare.py
  Sampler    : Euler-Maruyama only
  Steps      : 25, 50, 100, 200, 300

Paths used
----------
CODE_DIR:
  /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code

RESULT_ROOT:
  /tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1

How to install/run
------------------
1. Copy this folder to the cluster, for example into:

   /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code/slurm_exp1

2. Make sure your patched synthetic code exists at CODE_DIR.

3. Make sure your environment works:

   source ~/linuxhome/venvs/gad_real/bin/activate
   cd /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
   python run_synthetic_scheduler_sampler_compare.py --help

4. Make sure Kerberos is active:

   klist || kinit

5. Submit the whole pipeline:

   cd /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code/slurm_exp1
   bash submit_tpami_synthetic_exp1_pipeline.sh

This submits:
  preflight -> 20-job GPU array -> aggregation

Concurrency
-----------
The array is set as:

  #SBATCH --array=0-19%10

This means 20 total jobs, with up to 10 running concurrently. Change it to:

  #SBATCH --array=0-19%20

only if your GPU quota allows all 20 at the same time.

Expected output files
---------------------
Each scheduler/dataset/seed writes:

  <RESULT_ROOT>/<dataset>/seed<seed>/<scheduler>/results_summary.csv
  <RESULT_ROOT>/<dataset>/seed<seed>/<scheduler>/forward_diagnostics_summary_all.csv
  <RESULT_ROOT>/<dataset>/seed<seed>/<scheduler>/paper_generation_table_em_100_200_300.csv
  <RESULT_ROOT>/<dataset>/seed<seed>/<scheduler>/paper_forward_stiffness_table.csv

The aggregation job writes:

  <RESULT_ROOT>/aggregate/all_generation_results.csv
  <RESULT_ROOT>/aggregate/aggregate_generation_mean_std.csv
  <RESULT_ROOT>/aggregate/paper_generation_table_em_100_200_300_mean_std.csv
  <RESULT_ROOT>/aggregate/all_forward_diagnostics.csv
  <RESULT_ROOT>/aggregate/aggregate_forward_stiffness_mean_std.csv
  <RESULT_ROOT>/aggregate/sampling_efficiency_stationary_gaussian_global_swd_mean.png
  <RESULT_ROOT>/aggregate/sampling_efficiency_block_covariance_global_swd_mean.png

Main-paper usage
----------------
Use only:

1. Forward/stiffness table: aggregate_forward_stiffness_mean_std.csv
2. Generation table at 100/200/300: paper_generation_table_em_100_200_300_mean_std.csv
3. Sampling efficiency plots from 25 to 300 steps.

Do not include extra synthetic families in the main paper. Keep high-frequency or localized-burst stress tests for appendix only if needed.
