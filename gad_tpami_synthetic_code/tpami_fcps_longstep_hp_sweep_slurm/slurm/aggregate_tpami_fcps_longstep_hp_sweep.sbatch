#!/bin/bash
#SBATCH --job-name=tpami_fcps_hp_agg
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/logs/slurm/aggregate_%j.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep/logs/slurm/aggregate_%j.err

set -euo pipefail

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

module purge || true
source ~/linuxhome/venvs/gad_real/bin/activate

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep
AGG_DIR=${RESULT_ROOT}/aggregate
mkdir -p "${RESULT_ROOT}/logs/slurm" "${AGG_DIR}"

python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_longstep_hp_sweep')
out = root / 'aggregate'
out.mkdir(parents=True, exist_ok=True)
expected = 270

files = sorted(root.glob('*/*/*/*/seed*/results_summary.csv'))
print(f'Found results_summary.csv files: {len(files)} / expected {expected}')
if len(files) == 0:
    raise SystemExit('ERROR: no results_summary.csv files found.')

rows = []
for f in files:
    run_dir = f.parent
    cfg_path = run_dir / 'run_config.json'
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, 'r') as fh:
            cfg = json.load(fh).get('args', {})
    df = pd.read_csv(f)
    df['fcps_c0'] = float(cfg.get('fcps_c0', run_dir.parents[3].name.replace('c0_', '')))
    df['fcps_alpha'] = float(cfg.get('fcps_alpha', run_dir.parents[2].name.replace('alpha_', '')))
    df['fcps_cmin'] = float(cfg.get('fcps_cmin', run_dir.parents[1].name.replace('cmin_', '')))
    df['run_dir'] = str(run_dir)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(out / 'fcps_longstep_hp_all_rows.csv', index=False)

metrics = [
    'global_swd_mean', 'mmd_rbf_z', 'cov_rel_err', 'spec_cov_rel_err',
    'band_ratio_l1', 'smoothness_w1', 'graph_stationarity_gap_abs',
    'generated_finite_sample_fraction', 'generated_finite_element_fraction',
    'generated_nonfinite_flag', 'generated_nan_count', 'generated_inf_count',
    'forward_stiffness_proxy_max', 'forward_stiffness_proxy_log10_max',
    'forward_snr_log10_std_mean', 'forward_snr_log10_std_max',
    'forward_snr_log10_range_mean', 'forward_snr_log10_range_max',
    'forward_terminal_retained_signal_energy_mean',
    'forward_terminal_noise_prior_rel_err', 'forward_terminal_rho_prior_rel_err',
    'best_epoch'
]
metrics = [m for m in metrics if m in all_df.columns]
group_cols = ['dataset_family', 'schedule', 'method', 'steps', 'fcps_c0', 'fcps_alpha', 'fcps_cmin']
summary = all_df.groupby(group_cols)[metrics].agg(['mean', 'std', 'min', 'max']).reset_index()
summary.columns = ['_'.join([str(x) for x in col if str(x) != '']).rstrip('_') for col in summary.columns]
summary.to_csv(out / 'fcps_longstep_hp_mean_std.csv', index=False)

# Rank candidates. Divergent rows already have large metric penalties; we also explicitly prefer finite runs.
rank = summary.copy()
for col in [
    'global_swd_mean_mean', 'mmd_rbf_z_mean', 'cov_rel_err_mean', 'spec_cov_rel_err_mean',
    'band_ratio_l1_mean', 'generated_nonfinite_flag_mean', 'forward_terminal_noise_prior_rel_err_mean',
    'forward_stiffness_proxy_log10_max_mean'
]:
    if col not in rank.columns:
        rank[col] = np.nan
rank['finite_rate'] = 1.0 - rank['generated_nonfinite_flag_mean'].fillna(1.0)
rank['quality_score'] = (
    rank['global_swd_mean_mean'].fillna(1e9)
    + rank['mmd_rbf_z_mean'].fillna(1e9)
    + 0.25 * rank['cov_rel_err_mean'].fillna(1e9)
    + 0.25 * rank['spec_cov_rel_err_mean'].fillna(1e9)
    + 0.50 * rank['band_ratio_l1_mean'].fillna(1e9)
)
rank['diagnostic_score'] = (
    rank['quality_score']
    + 0.10 * rank['forward_stiffness_proxy_log10_max_mean'].fillna(100.0)
    + 0.25 * rank['forward_terminal_noise_prior_rel_err_mean'].fillna(100.0)
    + 1000.0 * rank['generated_nonfinite_flag_mean'].fillna(1.0)
)
rank = rank.sort_values(['dataset_family', 'steps', 'generated_nonfinite_flag_mean', 'diagnostic_score'])
rank.to_csv(out / 'fcps_longstep_hp_ranked_candidates.csv', index=False)

# Compact stability table: which settings ever became finite?
stability_cols = ['dataset_family', 'steps', 'fcps_c0', 'fcps_alpha', 'fcps_cmin']
stab_metrics = [m for m in ['generated_nonfinite_flag', 'generated_finite_sample_fraction', 'global_swd_mean', 'cov_rel_err', 'forward_stiffness_proxy_max', 'forward_terminal_noise_prior_rel_err'] if m in all_df.columns]
stab = all_df.groupby(stability_cols)[stab_metrics].agg(['mean', 'std', 'min', 'max']).reset_index()
stab.columns = ['_'.join([str(x) for x in col if str(x) != '']).rstrip('_') for col in stab.columns]
stab.to_csv(out / 'fcps_longstep_hp_stability_table.csv', index=False)

# Best per dataset and step, after removing any setting with non-finite rate > 0.
stable = rank[rank['generated_nonfinite_flag_mean'].fillna(1.0) == 0.0].copy()
if len(stable):
    best = stable.groupby(['dataset_family', 'steps'], as_index=False).head(10)
    best.to_csv(out / 'fcps_longstep_hp_top10_stable_per_dataset_step.csv', index=False)
else:
    (out / 'fcps_longstep_hp_top10_stable_per_dataset_step.csv').write_text('No fully stable FCPS configuration found.\n')

print('Wrote aggregate files:')
for p in sorted(out.iterdir()):
    print(' ', p, p.stat().st_size, 'bytes')
PY
