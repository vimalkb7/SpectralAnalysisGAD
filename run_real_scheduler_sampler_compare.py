from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from graphs import eigh_sym
from metrics import summarize_generation_metrics
from real_datasets import load_real_dataset
from run_synthetic_scheduler_sampler_compare import (
    ensure_dir,
    save_csv,
    sanitize_metric_dict,
    set_all_seeds,
    train_one_scheduler,
    write_json,
)
from samplers import reverse_sample
from schedules import build_schedules



def main() -> None:
    p = argparse.ArgumentParser(description='End-to-end unconditional generation on real graph-signal snapshot datasets.')
    p.add_argument('--out', type=str, default='outputs_real_compare')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'])
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
    p.add_argument('--schedulers', type=str, default='FCPS,MAS-SNR')
    p.add_argument('--methods', type=str, default='em,heun,ddpm,ddim')
    p.add_argument('--steps', type=str, default='25,50,100')
    p.add_argument('--t0', type=float, default=1e-3)
    p.add_argument('--ddim_eta', type=float, default=0.0)
    p.add_argument('--sampler_init_mode', type=str, default='algorithm2', choices=['algorithm2', 'stationary', 'rho'])
    p.add_argument('--fcps_c0', type=float, default=2300.0)
    p.add_argument('--fcps_cmin', type=float, default=0.1)
    p.add_argument('--fcps_alpha', type=float, default=4.0)
    p.add_argument('--r_max', type=float, default=1e4)
    p.add_argument('--r_min', type=float, default=None)
    p.add_argument('--match_terminal_energy', action='store_true')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--time_embed_dim', type=int, default=64)
    p.add_argument('--cheb_order', type=int, default=5)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_epochs', type=int, default=40)
    p.add_argument('--grad_clip_norm', type=float, default=1.0)
    p.add_argument('--val_every_epochs', type=int, default=5)
    p.add_argument('--val_method', type=str, default='em', choices=['em', 'heun', 'ddpm', 'ddim'])
    p.add_argument('--val_steps', type=int, default=50)
    p.add_argument('--val_num_samples', type=int, default=512)
    p.add_argument('--metric_num_projections', type=int, default=128)
    p.add_argument('--metric_num_seeds', type=int, default=5)
    p.add_argument('--best_model_metric', type=str, default='val_selection_score', choices=['val_selection_score', 'global_swd_mean', 'mmd_rbf_z', 'cov_rel_err', 'spec_cov_rel_err', 'node_w1_median', 'node_w1_q90', 'band_ratio_l1', 'smoothness_w1', 'graph_stationarity_gap_abs', 'ipr_w1', 'gini_w1'])
    p.add_argument('--val_score_swd_weight', type=float, default=1.0)
    p.add_argument('--val_score_mmd_weight', type=float, default=1.0)
    p.add_argument('--val_score_cov_weight', type=float, default=1.0)
    p.add_argument('--val_score_spec_cov_weight', type=float, default=0.5)
    p.add_argument('--val_score_band_weight', type=float, default=0.0)
    p.add_argument('--val_score_nodew1_weight', type=float, default=0.5)
    p.add_argument('--val_score_nodew1q90_weight', type=float, default=0.0)
    p.add_argument('--val_score_smoothness_weight', type=float, default=0.25)
    p.add_argument('--val_score_stationarity_weight', type=float, default=0.0)
    p.add_argument('--val_score_localization_weight', type=float, default=0.0)
    p.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'none'])
    p.add_argument('--lr_decay_factor', type=float, default=0.5)
    p.add_argument('--lr_patience_validations', type=int, default=2)
    p.add_argument('--lr_threshold', type=float, default=1e-4)
    args = p.parse_args()

    print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
    set_all_seeds(args.seed)
    ensure_dir(args.out)
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64

    bundle = load_real_dataset(
        dataset_name=args.dataset,
        root=args.root,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        gamma=args.gamma,
        stride=args.stride,
        intel_readings_path=args.intel_readings_path,
        intel_coords_path=args.intel_coords_path,
        intel_value_col=args.intel_value_col,
    )
    X_train_np = bundle.train.x.numpy().astype(np.float64)
    X_val_np = bundle.val.x.numpy().astype(np.float64)
    X_test_np = bundle.test.x.numpy().astype(np.float64)
    Lg = bundle.laplacian.astype(np.float64)
    W = bundle.adjacency.astype(np.float64)
    lam_gamma_np, V_np = eigh_sym(Lg)

    V = torch.tensor(V_np, device=device, dtype=dtype)
    lam_gamma = torch.tensor(lam_gamma_np, device=device, dtype=dtype)
    W_torch = torch.tensor(W, device=device, dtype=dtype)
    L_torch = torch.tensor(Lg, device=device, dtype=dtype)
    X_train = torch.tensor(X_train_np, device=device, dtype=dtype)
    X_val = torch.tensor(X_val_np, device=device, dtype=dtype)
    X_test = torch.tensor(X_test_np, device=device, dtype=dtype)
    s_k = ((X_train @ V) ** 2).mean(dim=0)

    sched_info = build_schedules(
        lam_gamma=lam_gamma_np,
        s_k=s_k.detach().cpu().numpy(),
        sigma=args.sigma,
        T=args.T,
        fcps_c0=args.fcps_c0,
        fcps_cmin=args.fcps_cmin,
        fcps_alpha=args.fcps_alpha,
        r_max=args.r_max,
        r_min=args.r_min,
        match_terminal_energy=args.match_terminal_energy,
    )
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    steps_list = [int(s.strip()) for s in args.steps.split(',') if s.strip()]
    scheduler_names = [s.strip() for s in args.schedulers.split(',') if s.strip()]

    write_json(os.path.join(args.out, 'run_config.json'), {
        'args': vars(args),
        'device_used': str(device),
        'dataset_name_loaded': bundle.name,
        'num_nodes': int(X_train_np.shape[1]),
        'train_samples': int(len(X_train_np)),
        'r_min_used': float(sched_info['r_min']),
    })

    results_rows: list[dict] = []
    for schedule_name in scheduler_names:
        schedule_obj = sched_info[schedule_name]
        schedule_dir = os.path.join(args.out, schedule_name.replace('/', '_'))
        model, _, _, best_summary = train_one_scheduler(
            schedule_name=schedule_name,
            schedule_obj=schedule_obj,
            X_train=X_train,
            X_val=X_val,
            X_val_np=X_val_np,
            X_test_np=X_test_np,
            V=V,
            V_np=V_np,
            lam_gamma=lam_gamma,
            W=W_torch,
            L_torch=L_torch,
            L_np=Lg,
            s_k=s_k,
            args=args,
            schedule_dir=schedule_dir,
        )
        for method in methods:
            for steps in steps_list:
                with torch.no_grad():
                    X_gen = reverse_sample(
                        model=model,
                        V=V,
                        lam_gamma=lam_gamma,
                        sigma=args.sigma,
                        schedule_name=schedule_name,
                        schedule_obj=schedule_obj,
                        s_k=s_k,
                        num_samples=min(len(X_test_np), args.val_num_samples),
                        steps=steps,
                        t0=args.t0,
                        T=args.T,
                        seed=args.seed + 80000 + steps,
                        method=method,
                        ddim_eta=args.ddim_eta,
                        init_mode=args.sampler_init_mode,
                    )
                metrics = summarize_generation_metrics(
                    X_gen.detach().cpu().numpy(),
                    X_test_np[: min(len(X_test_np), args.val_num_samples)],
                    V_np,
                    Lg,
                    sw_num_projections=args.metric_num_projections,
                    sw_seeds=range(args.metric_num_seeds),
                )
                row = {
                    'dataset': bundle.name,
                    'schedule': schedule_name,
                    'method': method,
                    'steps': steps,
                    **sanitize_metric_dict(metrics),
                }
                if best_summary is not None:
                    row['best_epoch'] = int(best_summary['epoch'])
                results_rows.append(row)
                print(f"[{bundle.name}][{schedule_name}] {method} steps={steps} | swd={row['global_swd_mean']:.6f} | mmd_z={row['mmd_rbf_z']:.6f}")
        save_csv(Path(args.out) / 'results_summary.csv', results_rows)


if __name__ == '__main__':
    main()
