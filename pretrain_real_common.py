from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graphs import eigh_sym
from metrics import cov_rel_error, sample_cov, spectral_covariance
from real_datasets import load_real_dataset


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def save_json(path: str | Path, obj) -> None:
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x) if hasattr(x, 'item') else str(x),
        )



def save_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(Path(path).parent)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def q95_relative_error(x: np.ndarray, y: np.ndarray, valid_mask: np.ndarray | None = None, min_abs_ref: float = 1e-10) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.abs(y) > float(min_abs_ref)
    if valid_mask is not None:
        mask = mask & np.asarray(valid_mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    rel = np.abs(x[mask] - y[mask]) / (np.abs(y[mask]) + 1e-12)
    return float(np.quantile(rel, 0.95))



def active_mode_mask(s_k: np.ndarray, rel_floor: float = 1e-4) -> np.ndarray:
    s_k = np.asarray(s_k, dtype=np.float64)
    thr = float(rel_floor) * float(np.max(s_k) + 1e-12)
    return s_k > max(thr, 1e-12)



def fro_offdiag_fraction(C: np.ndarray) -> float:
    C = np.asarray(C, dtype=np.float64)
    D = np.diag(np.diag(C))
    return float(np.linalg.norm(C - D, ord='fro') / (np.linalg.norm(C, ord='fro') + 1e-12))



def choose_representative_modes(lam_gamma: np.ndarray, num_each_end: int) -> list[int]:
    n = int(len(lam_gamma))
    m = max(1, min(int(num_each_end), max(1, n // 2)))
    idx = list(range(m))
    tail = list(range(max(m, n - m), n))
    rep = []
    for k in idx + tail:
        if k not in rep:
            rep.append(k)
    return rep


# -----------------------------------------------------------------------------
# Schedule formulas
# -----------------------------------------------------------------------------


def fcps_trajectory(lam_gamma: np.ndarray, T: float, c0: float, c_min: float, alpha: float, t_grid: np.ndarray):
    t_grid = np.asarray(t_grid, dtype=np.float64)
    lam_gamma = np.asarray(lam_gamma, dtype=np.float64)
    if c0 <= c_min * T:
        raise ValueError('Need c0 > c_min * T for FCPS.')
    u = np.clip(t_grid / T, 0.0, 1.0)
    cbar = c_min * t_grid + (c0 - c_min * T) * (u ** (alpha + 1.0))
    k = (c0 - c_min * T) * (alpha + 1.0) / T
    ct = c_min + k * (u ** alpha)
    a = np.exp(-np.outer(cbar, lam_gamma))
    a2 = a ** 2
    return a, a2, ct[:, None], cbar[:, None]



def mas_trajectory(lam_gamma: np.ndarray, s_k: np.ndarray, sigma: float, T: float, r_max: float, r_min: float, t_grid: np.ndarray):
    t_grid = np.asarray(t_grid, dtype=np.float64)
    x = np.clip(t_grid / T, 0.0, 1.0)
    s = 0.5 * (1.0 - np.cos(np.pi * x))
    sprime = 0.5 * (np.pi / T) * np.sin(np.pi * x)
    log_r = np.log(r_max) + (np.log(r_min) - np.log(r_max)) * s
    d_log_r_dt = (np.log(r_min) - np.log(r_max)) * sprime
    r = np.exp(log_r)[:, None]
    s_k_safe = np.maximum(np.asarray(s_k, dtype=np.float64), 1e-10)
    gk = np.asarray(lam_gamma, dtype=np.float64) * s_k_safe / (sigma ** 2)
    a2 = r / (r + gk[None, :])
    a = np.sqrt(a2)
    ct = -(0.5 / lam_gamma[None, :]) * (1.0 - a2) * d_log_r_dt[:, None]
    ct = np.clip(ct, 0.0, None)
    cbar = 0.5 / lam_gamma[None, :] * np.log1p(gk[None, :] / r)
    return a, a2, ct, cbar



def nu_from_a2(a2: np.ndarray, lam_gamma: np.ndarray, sigma: float) -> np.ndarray:
    return (sigma ** 2) * (1.0 - np.asarray(a2, dtype=np.float64)) / np.asarray(lam_gamma, dtype=np.float64)[None, :]



def rho_from_a2_nu(a2: np.ndarray, nu: np.ndarray, s_k: np.ndarray) -> np.ndarray:
    return np.asarray(a2, dtype=np.float64) * np.asarray(s_k, dtype=np.float64)[None, :] + np.asarray(nu, dtype=np.float64)



def snr_from_a2_nu(a2: np.ndarray, nu: np.ndarray, s_k: np.ndarray) -> np.ndarray:
    return (np.asarray(a2, dtype=np.float64) * np.asarray(s_k, dtype=np.float64)[None, :]) / (np.asarray(nu, dtype=np.float64) + 1e-12)



def retained_signal_energy(a2: np.ndarray, s_k: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(a2, dtype=np.float64) * np.asarray(s_k, dtype=np.float64)[None, :], axis=1)


# -----------------------------------------------------------------------------
# Empirical channel checks
# -----------------------------------------------------------------------------


def sample_forward_at_time(X0_tilde: np.ndarray, V: np.ndarray, a_t: np.ndarray, nu_t: np.ndarray, rng: np.random.Generator):
    z = rng.standard_normal(size=X0_tilde.shape)
    Xt_tilde = X0_tilde * a_t[None, :] + np.sqrt(np.clip(nu_t, 0.0, None))[None, :] * z
    Xt = Xt_tilde @ V.T
    return Xt, Xt_tilde



def estimate_mode_attenuation(X0_tilde: np.ndarray, Xt_tilde: np.ndarray) -> np.ndarray:
    num = np.mean(X0_tilde * Xt_tilde, axis=0)
    den = np.mean(X0_tilde * X0_tilde, axis=0) + 1e-12
    return num / den



def build_theory_covariances_over_time(C0_spec: np.ndarray, a: np.ndarray, nu: np.ndarray, V: np.ndarray):
    Cspec_list: list[np.ndarray] = []
    Cnode_list: list[np.ndarray] = []
    for i in range(a.shape[0]):
        A = a[i]
        Cspec = (A[:, None] * C0_spec) * A[None, :] + np.diag(nu[i])
        Cnode = V @ Cspec @ V.T
        Cspec_list.append(Cspec)
        Cnode_list.append(Cnode)
    return Cspec_list, Cnode_list


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_mode_curves(path: str | Path, t_grid: np.ndarray, Y: np.ndarray, lam_gamma: np.ndarray, title: str, ylabel: str, rep_modes: list[int], plot_all_modes: bool) -> None:
    ensure_dir(Path(path).parent)
    plt.figure(figsize=(8, 5))
    if plot_all_modes:
        for k in range(Y.shape[1]):
            plt.plot(t_grid, Y[:, k], linewidth=0.8, alpha=0.12)
    for k in rep_modes:
        plt.plot(t_grid, Y[:, k], linewidth=2.0, label=f'mode {k} (lam={lam_gamma[k]:.3g})')
    plt.xlabel('t')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if rep_modes:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()



def plot_heatmap(path: str | Path, Z: np.ndarray, t_grid: np.ndarray, title: str, cbar_label: str, log10_clip_floor: float = 1e-12) -> None:
    ensure_dir(Path(path).parent)
    Zplot = np.log10(np.maximum(np.asarray(Z, dtype=np.float64), log10_clip_floor))
    plt.figure(figsize=(9, 5))
    plt.imshow(Zplot.T, aspect='auto', origin='lower', extent=[float(t_grid[0]), float(t_grid[-1]), 0, Z.shape[1] - 1])
    plt.colorbar(label=f'log10({cbar_label})')
    plt.xlabel('t')
    plt.ylabel('mode k')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()



def plot_scalar_summaries(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    t = np.asarray([r['t'] for r in rows], dtype=np.float64)
    plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, [r['var_residual_q95_rel_err'] for r in rows], linewidth=2)
    ax1.set_title('Residual variance match')
    ax1.set_xlabel('t')
    ax1.set_ylabel('q95 relative error')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, [r['attenuation_q95_rel_err'] for r in rows], linewidth=2)
    ax2.set_title('Attenuation recovery')
    ax2.set_xlabel('t')
    ax2.set_ylabel('q95 relative error')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, [r['node_cov_rel_err_vs_theory'] for r in rows], linewidth=2)
    ax3.set_title('Node covariance vs theory')
    ax3.set_xlabel('t')
    ax3.set_ylabel('relative error')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, [r['spectral_offdiag_frac_emp'] for r in rows], linewidth=2, label='empirical')
    ax4.plot(t, [r['spectral_offdiag_frac_theory'] for r in rows], linewidth=2, linestyle='--', label='theory')
    ax4.set_title('Spectral off-diagonal fraction')
    ax4.set_xlabel('t')
    ax4.set_ylabel('fraction')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main real-data schedule check
# -----------------------------------------------------------------------------


def run_real_pretrain_check(args, scheduler_name: str) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)

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
    # X0 = bundle.train.x.numpy().astype(np.float64)
    X0_full = bundle.train.x.numpy().astype(np.float64)

    num_total_train = X0_full.shape[0]
    num_requested = int(getattr(args, 'num_train_samples_for_checks', 0) or 0)

    if num_requested > 0 and num_requested < num_total_train:
        rng_subset = np.random.default_rng(args.seed + 999)
        subset_idx = rng_subset.choice(num_total_train, size=num_requested, replace=False)
        subset_idx = np.sort(subset_idx)
        X0 = X0_full[subset_idx]
    else:
        subset_idx = np.arange(num_total_train)
        X0 = X0_full

    Lg = bundle.laplacian.astype(np.float64)
    W = bundle.adjacency.astype(np.float64)
    lam_gamma, V = eigh_sym(Lg)

    C0 = sample_cov(X0)
    C0_spec = spectral_covariance(X0, V)
    s_diag = np.diag(C0_spec).copy()
    X0_tilde = X0 @ V
    active_mask = active_mode_mask(s_diag)

    t_grid = np.linspace(0.0, args.T, args.num_t, dtype=np.float64)
    if scheduler_name == 'FCPS':
        a, a2, ct, cbar = fcps_trajectory(lam_gamma, args.T, args.fcps_c0, args.fcps_cmin, args.fcps_alpha, t_grid)
    elif scheduler_name == 'MAS-SNR':
        a, a2, ct, cbar = mas_trajectory(lam_gamma, s_diag, args.sigma, args.T, args.r_max, args.r_min, t_grid)
    else:
        raise ValueError(f'Unknown scheduler_name: {scheduler_name}')

    nu = nu_from_a2(a2, lam_gamma, args.sigma)
    rho = rho_from_a2_nu(a2, nu, s_diag)
    snr = snr_from_a2_nu(a2, nu, s_diag)
    retained = retained_signal_energy(a2, s_diag)
    theory_Cspec_list, theory_Cnode_list = build_theory_covariances_over_time(C0_spec, a, nu, V)

    eval_indices = np.linspace(0, args.num_t - 1, args.num_eval_times, dtype=int)
    eval_indices = np.unique(eval_indices)
    rng = np.random.default_rng(args.seed + 12345)
    rows: list[dict] = []
    for idx in eval_indices:
        Xt, Xt_tilde = sample_forward_at_time(X0_tilde, V, a[idx], nu[idx], rng)
        residual = Xt_tilde - a[idx][None, :] * X0_tilde
        residual_var_emp = np.var(residual, axis=0)
        est_att = estimate_mode_attenuation(X0_tilde, Xt_tilde)
        emp_node_cov = sample_cov(Xt)
        emp_spec_cov = spectral_covariance(Xt, V)
        theory_node_cov = theory_Cnode_list[idx]
        theory_spec_cov = theory_Cspec_list[idx]
        row = {
            'scheduler': scheduler_name,
            'dataset': bundle.name,
            't': float(t_grid[idx]),
            'var_residual_q95_rel_err': q95_relative_error(residual_var_emp, nu[idx], valid_mask=active_mask),
            'attenuation_q95_rel_err': q95_relative_error(est_att, a[idx], valid_mask=active_mask),
            'node_cov_rel_err_vs_theory': cov_rel_error(emp_node_cov, theory_node_cov),
            'spec_cov_rel_err_vs_theory': cov_rel_error(emp_spec_cov, theory_spec_cov),
            'spectral_offdiag_frac_emp': fro_offdiag_fraction(emp_spec_cov),
            'spectral_offdiag_frac_theory': fro_offdiag_fraction(theory_spec_cov),
            'mean_norm_per_node': float(np.linalg.norm(Xt.mean(axis=0)) / np.sqrt(Xt.shape[1])),
        }
        rows.append(row)

    rep_modes = choose_representative_modes(lam_gamma, args.num_modes_each_end)
    stem = scheduler_name.lower().replace('-', '_')
    plot_mode_curves(out_dir / f'{stem}_mode_decay.png', t_grid, a, lam_gamma, f'{scheduler_name} mode decay ({bundle.name})', 'a_k(t)', rep_modes, args.plot_all_modes)
    plot_mode_curves(out_dir / f'{stem}_covariance_growth.png', t_grid, nu, lam_gamma, f'{scheduler_name} covariance growth ({bundle.name})', 'nu_k(t)', rep_modes, args.plot_all_modes)
    plot_heatmap(out_dir / f'{stem}_snr_heatmap.png', snr, t_grid, f'{scheduler_name} SNR heatmap ({bundle.name})', 'SNR')
    plot_heatmap(out_dir / f'{stem}_rho_heatmap.png', rho, t_grid, f'{scheduler_name} rho heatmap ({bundle.name})', 'rho')
    plot_scalar_summaries(out_dir / f'{stem}_empirical_checks.png', rows)

    np.savez_compressed(
        out_dir / f'{stem}_trajectories.npz',
        t_grid=t_grid,
        lam_gamma=lam_gamma,
        s_diag=s_diag,
        a=a,
        a2=a2,
        nu=nu,
        rho=rho,
        snr=snr,
        ct=ct,
        cbar=cbar,
        retained_signal_energy=retained,
        representative_modes=np.asarray(rep_modes, dtype=np.int64),
    )
    save_csv(out_dir / f'{stem}_eval_metrics.csv', rows)

    summary = {
        'scheduler': scheduler_name,
        'dataset': bundle.name,
        'num_nodes': int(X0.shape[1]),
        'num_train_samples_total': int(num_total_train),
        'num_train_samples_used': int(X0.shape[0]),
        'subsampled_for_checks': bool(X0.shape[0] != num_total_train),
        'terminal_retained_signal_energy': float(retained[-1]),
        'terminal_max_abs_a': float(np.max(np.abs(a[-1]))),
        'terminal_mean_nu': float(np.mean(nu[-1])),
        'max_ct': float(np.max(ct)),
        'eval_metrics_mean': {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in {'scheduler', 'dataset'}},
        'params': vars(args),
    }
    save_json(out_dir / f'{stem}_summary.json', summary)
    print(json.dumps(summary, indent=2))
