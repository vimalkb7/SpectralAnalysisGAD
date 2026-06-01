#!/bin/bash
#SBATCH --job-name=tpami_c0_agg
#SBATCH --account=testusers
#SBATCH --partition=all
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep/logs/slurm/aggregate_%j.out
#SBATCH --error=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep/logs/slurm/aggregate_%j.err

set -euo pipefail

if ! klist -s; then
  echo "ERROR: Kerberos ticket missing. Run: kinit"
  exit 1
fi

module purge || true
source ~/linuxhome/venvs/gad_real/bin/activate

RESULT_ROOT=/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep
AGG_DIR=${RESULT_ROOT}/aggregate
mkdir -p "${RESULT_ROOT}/logs/slurm" "${AGG_DIR}"

python - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd

root = Path('/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD/Synthetic_data/exp1/fcps_c0_sweep')
out = root / 'aggregate'
out.mkdir(parents=True, exist_ok=True)

files = sorted(root.glob('*/*/seed*/results_summary.csv'))
print(f'Found results_summary.csv files: {len(files)} / expected 24')
if len(files) < 24:
    print('Existing result files:')
    for f in files:
        print(f)
    raise SystemExit('ERROR: Expected 24 result files before aggregation.')

rows = []
for f in files:
    run_dir = f.parent
    cfg_path = run_dir / 'run_config.json'
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, 'r') as fh:
            cfg = json.load(fh).get('args', {})
    df = pd.read_csv(f)
    df['fcps_c0'] = float(cfg.get('fcps_c0', str(run_dir.parent.name).replace('c0_', '')))
    df['fcps_cmin'] = float(cfg.get('fcps_cmin', 0.1))
    df['fcps_alpha'] = float(cfg.get('fcps_alpha', 4.0))
    df['run_dir'] = str(run_dir)
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(out / 'fcps_c0_sweep_all_rows.csv', index=False)

metrics = [
    'global_swd_mean', 'mmd_rbf_z', 'cov_rel_err', 'spec_cov_rel_err',
    'band_ratio_l1', 'smoothness_w1', 'graph_stationarity_gap_abs',
    'generated_finite_fraction', 'generated_nonfinite_flag',
    'forward_max_stiffness_proxy', 'forward_mean_log_snr_std_active',
    'forward_max_log_snr_range_active',
]
metrics = [m for m in metrics if m in all_df.columns]

group_cols = ['dataset_family', 'schedule', 'method', 'steps', 'fcps_c0']
summary = all_df.groupby(group_cols)[metrics].agg(['mean', 'std', 'min', 'max']).reset_index()
summary.columns = ['_'.join([str(x) for x in col if str(x) != '']).rstrip('_') for col in summary.columns]
summary.to_csv(out / 'fcps_c0_sweep_mean_std.csv', index=False)

# Compact stability table for paper/rebuttal decisions.
stability_cols = ['dataset_family', 'fcps_c0']
stability_metrics = [m for m in ['generated_finite_fraction', 'generated_nonfinite_flag', 'global_swd_mean', 'cov_rel_err', 'forward_max_stiffness_proxy'] if m in all_df.columns]
stab = all_df.groupby(stability_cols)[stability_metrics].agg(['mean', 'std']).reset_index()
stab.columns = ['_'.join([str(x) for x in col if str(x) != '']).rstrip('_') for col in stab.columns]
stab.to_csv(out / 'fcps_c0_sweep_stability_table.csv', index=False)

print('Wrote:')
for p in sorted(out.iterdir()):
    print(' ', p, p.stat().st_size, 'bytes')
PY
