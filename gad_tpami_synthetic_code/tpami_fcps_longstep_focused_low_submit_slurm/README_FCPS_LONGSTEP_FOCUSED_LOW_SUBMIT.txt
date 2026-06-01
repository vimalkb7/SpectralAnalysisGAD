TPAMI FCPS long-step focused sweep - low submit-count package
================================================================

Why this package exists
-----------------------
The previous 270-task array can violate AssocMaxSubmitJobLimit because many clusters count array elements against the submitted-job limit.
This package reduces the submitted tasks by looping over FCPS candidates inside each Slurm task.

Recommended run
---------------
cd /home/vkumarasamybal/linuxhome/SpectralAnalysisGAD/gad_tpami_synthetic_code
unzip tpami_fcps_longstep_focused_low_submit_slurm.zip
cd tpami_fcps_longstep_focused_low_submit_slurm
bash submit_tpami_fcps_longstep_focused.sh

If you still get AssocMaxSubmitJobLimit, use:

bash submit_tpami_fcps_longstep_focused_ultra_low.sh

What it runs
------------
Datasets: stationary_gaussian, block_covariance
Seeds: 0, 1, 2
Sampler: Euler-Maruyama
Test steps: 1500, 2000, 2500
Validation steps: 1500

Focused FCPS candidates:
CID  c0    alpha  cmin
0    50    2.0    0.001
1    100   2.0    0.001
2    300   2.0    0.001
3    300   3.0    0.001
4    300   4.0    0.01
5    700   2.0    0.001
6    700   3.0    0.001
7    700   4.0    0.01
8    1200  2.0    0.001
9    1200  3.0    0.001

Reason for focused grid
-----------------------
The earlier c0-only sweep showed FCPS diverged at 300 steps even for c0=50,100,300,700.
For long-step testing, we focus on scalar schedules that have the best chance to become stable:
- alpha=2 or 3 is preferred because very high alpha increases the terminal FCPS rate.
- cmin=0.001 reduces the early-time drift floor compared with cmin=0.1.
- c0 covers under-corrupted stability controls, intermediate terminal-prior matching, and stronger terminal matching.

Output location
---------------
/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_focused

Aggregation outputs
-------------------
aggregate/fcps_longstep_focused_all_rows.csv
aggregate/fcps_longstep_focused_mean_std.csv
aggregate/fcps_longstep_focused_ranked_candidates.csv
aggregate/fcps_longstep_focused_stability_table.csv
aggregate/fcps_longstep_focused_top10_stable_per_dataset_step.csv

Useful Slurm commands
---------------------
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName%30,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>
