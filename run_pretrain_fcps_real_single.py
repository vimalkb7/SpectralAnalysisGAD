from __future__ import annotations

import argparse
import json

from pretrain_real_common import run_real_pretrain_check



def main() -> None:
    p = argparse.ArgumentParser(description='Run FCPS real-data forward diagnostics for unconditional graph-signal generation.')
    p.add_argument('--out', type=str, default='/tudelft.net/staff-bulk/ewi/insy/MMC/vimal/SpectralAnalysisGAD_Results/pretrain_checks/')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', type=str, default='metr_la', choices=['metr_la', 'pems_bay', 'air_quality_small', 'air_quality_full', 'intel_lab'])
    p.add_argument('--root', type=str, default='./data')
    p.add_argument('--intel_readings_path', type=str, default=None)
    p.add_argument('--intel_coords_path', type=str, default=None)
    p.add_argument('--intel_value_col', type=str, default='temperature', choices=['temperature', 'humidity', 'light', 'voltage'])
    p.add_argument('--train_frac', type=float, default=0.7)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--gamma', type=float, default=1e-3)
    p.add_argument('--sigma', type=float, default=1.0)
    p.add_argument('--T', type=float, default=1.0)
    p.add_argument('--num_t', type=int, default=500)
    p.add_argument('--num_eval_times', type=int, default=9)
    p.add_argument('--fcps_c0', type=float, default=2300.0)
    p.add_argument('--fcps_cmin', type=float, default=0.1)
    p.add_argument('--fcps_alpha', type=float, default=4.0)
    p.add_argument('--r_max', type=float, default=1e4)
    p.add_argument('--r_min', type=float, default=1e-3)
    p.add_argument('--num_modes_each_end', type=int, default=5)
    p.add_argument('--plot_all_modes', action='store_true')
    args = p.parse_args()

    print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
    run_real_pretrain_check(args, scheduler_name='FCPS')


if __name__ == '__main__':
    main()
