from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance

from metrics import (
    cov_rel_error,
    mmd_rbf_standardized,
    nodewise_wasserstein_summary,
    repeated_sliced_wasserstein,
    sample_cov,
)
from schedules import channel_params_for_times_torch
from theory_checks import (
    guaranteed_improvement_ratio,
    mass_M,
    scalar_stiffness_lower_bound,
)


EPS = 1e-12


def _log1mexp(log_x: np.ndarray) -> np.ndarray:
    """Stable log(1-exp(log_x)) for log_x <= 0."""
    x = np.minimum(np.asarray(log_x, dtype=np.float64), 0.0)
    out = np.empty_like(x)
    cutoff = -np.log(2.0)
    mask = x < cutoff
    out[mask] = np.log1p(-np.exp(x[mask]))
    out[~mask] = np.log(-np.expm1(x[~mask]))
    return out


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compressed, self-contained plotting-data bundle."""
    ensure_dir(Path(path).parent)
    clean = {str(k): np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **clean)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(":", "_")


def channel_params_numpy(
    schedule_name: str,
    schedule_obj,
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the spectral channel parameters on CPU and return NumPy arrays."""
    lam_t = torch.as_tensor(lam_gamma, dtype=torch.float64)
    s_t = torch.as_tensor(s_k, dtype=torch.float64)
    t_t = torch.as_tensor(times, dtype=torch.float64)
    out = channel_params_for_times_torch(schedule_name, schedule_obj, lam_t, s_t, sigma, t_t)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def select_extreme_frequency_modes(n: int, num_extreme_modes: int = 4) -> list[int]:
    """Return the first and last modes in eigenvalue order, without duplicates."""
    m = max(1, int(num_extreme_modes))
    raw = list(range(min(m, n))) + list(range(max(0, n - m), n))
    out: list[int] = []
    for k in raw:
        if k not in out:
            out.append(k)
    return out


def extreme_frequency_mode_groups(n: int, num_extreme_modes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError(f"n must be positive, received {n}")
    k = min(max(1, int(num_extreme_modes)), n)
    low = np.arange(k, dtype=np.int64)
    high = np.arange(n - k, n, dtype=np.int64)
    return low, high


def empirical_spectral_power(X: np.ndarray, V: np.ndarray, pmin: float = 1e-12) -> np.ndarray:
    X_tilde = np.asarray(X, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    p = np.mean(X_tilde ** 2, axis=0)
    return np.maximum(p, pmin)


def forward_moments_from_empirical_data(
    X0: np.ndarray,
    V: np.ndarray,
    a: np.ndarray,
    nu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X0 = np.asarray(X0, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    X0_tilde = X0 @ V
    mu0_tilde = X0_tilde.mean(axis=0)
    C0_tilde = sample_cov(X0_tilde)
    A = np.diag(np.asarray(a, dtype=np.float64))
    Nt = np.diag(np.asarray(nu, dtype=np.float64))
    mu_tilde = a * mu0_tilde
    C_tilde = A @ C0_tilde @ A + Nt
    mu = V @ mu_tilde
    C = V @ C_tilde @ V.T
    return mu, C, mu_tilde, C_tilde


def sample_forward_terminal(
    X0: np.ndarray,
    V: np.ndarray,
    a: np.ndarray,
    nu: np.ndarray,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X0 = np.asarray(X0, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    num_samples = min(int(num_samples), X0.shape[0])
    idx = rng.choice(X0.shape[0], size=num_samples, replace=False) if num_samples < X0.shape[0] else np.arange(X0.shape[0])
    X0_tilde = X0[idx] @ V
    Z = rng.standard_normal(size=X0_tilde.shape)
    Xt_tilde = X0_tilde * a[None, :] + Z * np.sqrt(np.maximum(nu, 0.0))[None, :]
    return Xt_tilde @ V.T


def sample_stationary_prior(
    V: np.ndarray,
    lam_gamma: np.ndarray,
    sigma: float,
    num_samples: int,
    seed: int = 0,
    prior_var: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if prior_var is None:
        prior_var = (float(sigma) ** 2) / np.asarray(lam_gamma, dtype=np.float64)
    prior_var = np.asarray(prior_var, dtype=np.float64)
    Z = rng.standard_normal(size=(int(num_samples), len(prior_var))) * np.sqrt(np.maximum(prior_var, 0.0))[None, :]
    return Z @ np.asarray(V, dtype=np.float64).T


def stationary_covariance(
    V: np.ndarray,
    lam_gamma: np.ndarray,
    sigma: float,
    prior_var: np.ndarray | None = None,
) -> np.ndarray:
    if prior_var is None:
        prior_var = (float(sigma) ** 2) / np.asarray(lam_gamma, dtype=np.float64)
    return np.asarray(V, dtype=np.float64) @ np.diag(np.asarray(prior_var, dtype=np.float64)) @ np.asarray(V, dtype=np.float64).T


def gaussian_kl(mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray, ridge: float = 1e-8) -> float:
    mu_p = np.asarray(mu_p, dtype=np.float64)
    mu_q = np.asarray(mu_q, dtype=np.float64)
    cov_p = np.asarray(cov_p, dtype=np.float64)
    cov_q = np.asarray(cov_q, dtype=np.float64)
    n = cov_p.shape[0]
    Ip = np.eye(n)
    Cp = 0.5 * (cov_p + cov_p.T) + ridge * Ip
    Cq = 0.5 * (cov_q + cov_q.T) + ridge * Ip
    sign_p, logdet_p = np.linalg.slogdet(Cp)
    sign_q, logdet_q = np.linalg.slogdet(Cq)
    if sign_p <= 0 or sign_q <= 0:
        return float("nan")
    diff = mu_q - mu_p
    sol_Cq_Cp = np.linalg.solve(Cq, Cp)
    sol_Cq_diff = np.linalg.solve(Cq, diff)
    val = 0.5 * (np.trace(sol_Cq_Cp) + diff @ sol_Cq_diff - n + logdet_q - logdet_p)
    return float(max(val, 0.0))


def gaussian_w2(mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray, ridge: float = 1e-8) -> float:
    mu_p = np.asarray(mu_p, dtype=np.float64)
    mu_q = np.asarray(mu_q, dtype=np.float64)
    cov_p = np.asarray(cov_p, dtype=np.float64)
    cov_q = np.asarray(cov_q, dtype=np.float64)
    n = cov_p.shape[0]
    Cp = 0.5 * (cov_p + cov_p.T) + ridge * np.eye(n)
    Cq = 0.5 * (cov_q + cov_q.T) + ridge * np.eye(n)
    try:
        Cq_sqrt = sqrtm(Cq)
        middle = Cq_sqrt @ Cp @ Cq_sqrt
        middle_sqrt = sqrtm(middle)
        tr_term = np.trace(Cp + Cq - 2.0 * middle_sqrt)
        val_sq = float(np.sum((mu_p - mu_q) ** 2) + np.real(tr_term))
        return float(np.sqrt(max(val_sq, 0.0)))
    except Exception:
        return float("nan")


def forward_terminal_metrics(
    X0: np.ndarray,
    V: np.ndarray,
    lam_gamma: np.ndarray,
    sigma: float,
    schedule_name: str,
    schedule_obj,
    s_k: np.ndarray,
    num_samples: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    pars_T = channel_params_numpy(schedule_name, schedule_obj, lam_gamma, s_k, sigma, np.asarray([schedule_obj.T], dtype=np.float64))
    a_T = pars_T["a"][0]
    nu_T = pars_T["nu"][0]
    prior_var_tilde = pars_T["prior_var"][0]

    mu_T, C_T, mu_T_tilde, C_T_tilde = forward_moments_from_empirical_data(X0, V, a_T, nu_T)
    C_stat = stationary_covariance(V, lam_gamma, sigma, prior_var=prior_var_tilde)
    mu_stat = np.zeros_like(mu_T)

    n_samp = min(int(num_samples), X0.shape[0])
    XT = sample_forward_terminal(X0, V, a_T, nu_T, num_samples=n_samp, seed=seed)
    XP = sample_stationary_prior(V, lam_gamma, sigma, num_samples=n_samp, seed=seed + 17, prior_var=prior_var_tilde)

    stat_var_tilde = np.asarray(prior_var_tilde, dtype=np.float64)
    terminal_var_tilde = np.diag(C_T_tilde)
    terminal_mean_abs = float(np.mean(np.abs(mu_T)))
    terminal_spectral_mean_abs = float(np.mean(np.abs(mu_T_tilde)))

    out: dict[str, float] = {}
    out["forward_to_stationary_kl_gaussian"] = gaussian_kl(mu_T, C_T, mu_stat, C_stat)
    out["stationary_to_forward_kl_gaussian"] = gaussian_kl(mu_stat, C_stat, mu_T, C_T)
    out["symmetric_kl_gaussian"] = 0.5 * (out["forward_to_stationary_kl_gaussian"] + out["stationary_to_forward_kl_gaussian"])
    out["gaussian_w2_to_stationary"] = gaussian_w2(mu_T, C_T, mu_stat, C_stat)
    out["cov_rel_err_to_stationary"] = cov_rel_error(C_T, C_stat)
    out["spectral_var_rel_err_to_stationary"] = float(np.linalg.norm(terminal_var_tilde - stat_var_tilde) / (np.linalg.norm(stat_var_tilde) + EPS))
    out["terminal_mean_abs"] = terminal_mean_abs
    out["terminal_spectral_mean_abs"] = terminal_spectral_mean_abs
    out["flattened_w1_to_stationary_samples"] = float(wasserstein_distance(XT.ravel(), XP.ravel()))
    out["mmd_rbf_z_to_stationary_samples"] = float(mmd_rbf_standardized(XT, XP))
    out.update({f"terminal_{k}": v for k, v in repeated_sliced_wasserstein(XT, XP, num_projections=128, seeds=(0, 1, 2)).items()})
    out.update({f"terminal_{k}": v for k, v in nodewise_wasserstein_summary(XT, XP).items()})
    out["terminal_a2_signal_energy_mean"] = float(np.mean((a_T ** 2) * np.asarray(s_k, dtype=np.float64)))
    out["terminal_noise_var_mean"] = float(np.mean(nu_T))
    out["stationary_var_mean"] = float(np.mean(stat_var_tilde))
    return out


def compute_forward_plot_arrays(
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    schedule_name: str,
    schedule_obj,
    num_times: int = 256,
    t_min: float = 1e-4,
    stiffness_h_proxy: float = 0.0,
    num_extreme_modes: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    times = np.linspace(max(float(t_min), 1e-8), float(schedule_obj.T), int(num_times), dtype=np.float64)
    pars = channel_params_numpy(schedule_name, schedule_obj, lam_gamma, s_k, sigma, times)

    
    forward_curve_times = np.linspace(0.0, float(schedule_obj.T), int(num_times), dtype=np.float64)
    forward_curve_pars = channel_params_numpy(
        schedule_name,
        schedule_obj,
        lam_gamma,
        s_k,
        sigma,
        forward_curve_times,
    )
    lam = np.asarray(lam_gamma, dtype=np.float64)
    power = np.asarray(s_k, dtype=np.float64)
    prior_power = np.asarray(pars["prior_var"][0], dtype=np.float64)

    log_a2 = np.asarray(pars["log_a2"], dtype=np.float64)
    log_noise_fraction = _log1mexp(log_a2)
    log_snr_nat = (
        log_a2
        + np.log(np.maximum(power, np.finfo(np.float64).tiny))[None, :]
        - np.log(np.maximum(prior_power, np.finfo(np.float64).tiny))[None, :]
        - log_noise_fraction
    )
    log_snr = log_snr_nat / np.log(10.0)
    snr = np.exp(np.clip(log_snr_nat, -745.0, 709.0))
    log_snr_std = np.std(log_snr, axis=1)
    log_snr_range = np.max(log_snr, axis=1) - np.min(log_snr, axis=1)

    h_proxy = float(stiffness_h_proxy)
    noise_fraction = -np.expm1(log_a2)
    stiffness_proxy = pars["drift_rate"] * (
        1.0 + pars["a2"] - 2.0 * pars["a"] * h_proxy
    ) / np.maximum(noise_fraction, np.finfo(np.float64).tiny)

    h_star = pars["a"] * power[None, :] / np.maximum(pars["rho"], EPS)
    stiffness_optimal = pars["drift_rate"] * (
        1.0 + pars["a2"] - 2.0 * pars["a"] * h_star
    ) / np.maximum(noise_fraction, np.finfo(np.float64).tiny)

    low_mode_indices, high_mode_indices = extreme_frequency_mode_groups(
        len(lam), num_extreme_modes=num_extreme_modes
    )
    selected_modes = np.asarray(
        select_extreme_frequency_modes(len(lam), num_extreme_modes=num_extreme_modes),
        dtype=np.int64,
    )
    mode_band = np.full(len(lam), "middle", dtype="U8")
    mode_band[low_mode_indices] = "low"
    mode_band[high_mode_indices] = "high"
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "normalized_time": times / float(schedule_obj.T),
        "forward_curve_times": forward_curve_times,
        "forward_curve_normalized_time": forward_curve_times / float(schedule_obj.T),
        "mode_index": np.arange(len(lam), dtype=np.int64),
        "lam_gamma": lam,
        "spectral_power": power,
        "prior_power": prior_power,
        "selected_modes": selected_modes,
        "low_mode_indices": low_mode_indices,
        "high_mode_indices": high_mode_indices,
        "mode_band": mode_band,
        "forward_curve_a": forward_curve_pars["a"],
        "forward_curve_a2": forward_curve_pars["a2"],
        "forward_curve_nu": forward_curve_pars["nu"],
        "forward_curve_rho": forward_curve_pars["rho"],
        "a": pars["a"],
        "a2": pars["a2"],
        "nu": pars["nu"],
        "rho": pars["rho"],
        "ct": pars["ct"],
        "cbar": pars["cbar"],
        "drift_rate": pars["drift_rate"],
        "diffusion_sq": pars["diffusion_sq"],
        "prior_var": pars["prior_var"],
        "snr": snr,
        "log_a2": log_a2,
        "log_noise_fraction": log_noise_fraction,
        "log10_snr": log_snr,
        "log10_snr_plot_clipped": np.clip(log_snr, -30.0, 30.0),
        "log10_snr_std": log_snr_std,
        "log10_snr_range": log_snr_range,
        "h_star": h_star,
        "stiffness_proxy": stiffness_proxy,
        "stiffness_proxy_max_abs": np.max(np.abs(stiffness_proxy), axis=1),
        "stiffness_proxy_mean_abs": np.mean(np.abs(stiffness_proxy), axis=1),
        "stiffness_optimal": stiffness_optimal,
        "stiffness_optimal_max_abs": np.max(np.abs(stiffness_optimal), axis=1),
        "stiffness_optimal_mean_abs": np.mean(np.abs(stiffness_optimal), axis=1),
    }
    metrics = {
        "mean_log_snr_std": float(np.mean(log_snr_std)),
        "max_log_snr_std": float(np.max(log_snr_std)),
        "mean_log_snr_range": float(np.mean(log_snr_range)),
        "max_log_snr_range": float(np.max(log_snr_range)),
        "max_stiffness_proxy": float(np.max(np.abs(stiffness_proxy))),
        "max_stiffness_optimal": float(np.max(np.abs(stiffness_optimal))),
        "stiffness_h_proxy": h_proxy,
    }
    return arrays, metrics


def _write_forward_timeseries_csv(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    rows = []
    for i, t in enumerate(arrays["times"]):
        rows.append(
            {
                "time": float(t),
                "normalized_time": float(arrays["normalized_time"][i]),
                "log10_snr_std": float(arrays["log10_snr_std"][i]),
                "log10_snr_range": float(arrays["log10_snr_range"][i]),
                "stiffness_proxy_max_abs": float(arrays["stiffness_proxy_max_abs"][i]),
                "stiffness_proxy_mean_abs": float(arrays["stiffness_proxy_mean_abs"][i]),
                "stiffness_optimal_max_abs": float(arrays["stiffness_optimal_max_abs"][i]),
                "stiffness_optimal_mean_abs": float(arrays["stiffness_optimal_mean_abs"][i]),
            }
        )
    write_csv(path, rows)


def _write_selected_mode_csv(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    rows = []
    low_rank = {int(k): i for i, k in enumerate(arrays["low_mode_indices"])}
    high_rank = {int(k): i for i, k in enumerate(arrays["high_mode_indices"])}
    for i, t in enumerate(arrays["times"]):
        for k in arrays["selected_modes"]:
            kk = int(k)
            if kk in low_rank:
                frequency_band = "low"
                band_rank = low_rank[kk]
            else:
                frequency_band = "high"
                band_rank = high_rank[kk]
            rows.append(
                {
                    "time": float(t),
                    "normalized_time": float(arrays["normalized_time"][i]),
                    "frequency_band": frequency_band,
                    "band_rank": int(band_rank),
                    "mode": kk,
                    "lam_gamma": float(arrays["lam_gamma"][kk]),
                    "spectral_power": float(arrays["spectral_power"][kk]),
                    "prior_power": float(arrays["prior_power"][kk]),
                    "a": float(arrays["a"][i, kk]),
                    "a2": float(arrays["a2"][i, kk]),
                    "nu": float(arrays["nu"][i, kk]),
                    "rho": float(arrays["rho"][i, kk]),
                    "snr": float(arrays["snr"][i, kk]),
                    "log10_snr": float(arrays["log10_snr"][i, kk]),
                    "stiffness_optimal": float(arrays["stiffness_optimal"][i, kk]),
                }
            )
    write_csv(path, rows)


def _write_extreme_mode_mean_decay_csv(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write Figure-1-ready a_{t,k} curves for the first/last K modes."""
    rows: list[dict] = []
    groups = (
        ("low", arrays["low_mode_indices"]),
        ("high", arrays["high_mode_indices"]),
    )
    for band, indices in groups:
        for band_rank, k in enumerate(indices):
            kk = int(k)
            for i, t in enumerate(arrays["forward_curve_times"]):
                rows.append(
                    {
                        "time": float(t),
                        "normalized_time": float(arrays["forward_curve_normalized_time"][i]),
                        "frequency_band": band,
                        "band_rank": int(band_rank),
                        "mode": kk,
                        "lam_gamma": float(arrays["lam_gamma"][kk]),
                        "spectral_power": float(arrays["spectral_power"][kk]),
                        "prior_power": float(arrays["prior_power"][kk]),
                        "mean_decay_a_tk": float(arrays["forward_curve_a"][i, kk]),
                        "mean_decay_a2_tk": float(arrays["forward_curve_a2"][i, kk]),
                        "conditional_noise_variance_nu_tk": float(arrays["forward_curve_nu"][i, kk]),
                        "effective_variance_rho_tk": float(arrays["forward_curve_rho"][i, kk]),
                    }
                )
    write_csv(path, rows)


def _write_modewise_log10_snr_csv(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    rows: list[dict] = []
    n = len(arrays["mode_index"])
    for i, t in enumerate(arrays["times"]):
        for k in range(n):
            rows.append(
                {
                    "time": float(t),
                    "normalized_time": float(arrays["normalized_time"][i]),
                    "mode": int(k),
                    "frequency_band": str(arrays["mode_band"][k]),
                    "lam_gamma": float(arrays["lam_gamma"][k]),
                    "spectral_power": float(arrays["spectral_power"][k]),
                    "prior_power": float(arrays["prior_power"][k]),
                    "snr_tk": float(arrays["snr"][i, k]),
                    "log10_snr_tk": float(arrays["log10_snr"][i, k]),
                }
            )
    write_csv(path, rows)


def run_forward_diagnostics(
    X_train_np: np.ndarray,
    V_np: np.ndarray,
    lam_gamma_np: np.ndarray,
    sigma: float,
    schedules: dict[str, object],
    scheduler_names: Iterable[str],
    dataset_name: str,
    out_dir: str | Path,
    num_times: int = 256,
    num_extreme_modes: int = 4,
    terminal_num_samples: int = 1024,
    seed: int = 0,
    pmin: float = 1e-10,
    t_min: float = 1e-4,
    stiffness_h_proxy: float = 0.0,
) -> list[dict[str, float | str]]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    s_k = empirical_spectral_power(X_train_np, V_np, pmin=pmin)
    rows: list[dict[str, float | str]] = []
    manifest_entries: list[dict[str, object]] = []

    for schedule_name in scheduler_names:
        if schedule_name not in schedules:
            raise KeyError(f"Schedule {schedule_name!r} is not available. Available: {sorted(schedules.keys())}")
        schedule_obj = schedules[schedule_name]
        schedule_out = out_dir / sanitize_name(schedule_name)
        ensure_dir(schedule_out)

        arrays, snr_metrics = compute_forward_plot_arrays(
            lam_gamma=lam_gamma_np,
            s_k=s_k,
            sigma=sigma,
            schedule_name=schedule_name,
            schedule_obj=schedule_obj,
            num_times=num_times,
            t_min=t_min,
            stiffness_h_proxy=stiffness_h_proxy,
            num_extreme_modes=num_extreme_modes,
        )
        npz_path = schedule_out / "forward_plot_data.npz"
        timeseries_path = schedule_out / "figure_a_timeseries.csv"
        selected_modes_path = schedule_out / "selected_mode_forward_curves.csv"
        mean_decay_path = schedule_out / "extreme_mode_mean_decay_curves.csv"
        modewise_snr_path = schedule_out / "modewise_log10_snr.csv"
        write_npz(npz_path, arrays)
        _write_forward_timeseries_csv(timeseries_path, arrays)
        _write_selected_mode_csv(selected_modes_path, arrays)
        _write_extreme_mode_mean_decay_csv(mean_decay_path, arrays)
        _write_modewise_log10_snr_csv(modewise_snr_path, arrays)

        terminal_metrics = forward_terminal_metrics(
            X0=X_train_np,
            V=V_np,
            lam_gamma=lam_gamma_np,
            sigma=sigma,
            schedule_name=schedule_name,
            schedule_obj=schedule_obj,
            s_k=s_k,
            num_samples=terminal_num_samples,
            seed=seed,
        )
        row: dict[str, float | str] = {
            "dataset": dataset_name,
            "schedule": schedule_name,
            "log_snr_role": "implementation_sanity_check_for_MASS; descriptive_path_diagnostic_for_scalar_baselines",
            "stiffness_role": "primary_falsifiable_schedule_only_reverse_rate_diagnostic",
            "forward_plot_npz": str(npz_path),
            "figure_a_timeseries_csv": str(timeseries_path),
            "selected_mode_curves_csv": str(selected_modes_path),
            "extreme_mode_mean_decay_csv": str(mean_decay_path),
            "modewise_log10_snr_csv": str(modewise_snr_path),
            **snr_metrics,
            **terminal_metrics,
        }
        rows.append(row)
        write_json(schedule_out / "forward_diagnostics_metrics.json", row)
        print_forward_diagnostic_summary(row)
        manifest_entries.append(
            {
                "schedule": schedule_name,
                "npz": str(npz_path.relative_to(out_dir)),
                "figure_a_timeseries_csv": str(timeseries_path.relative_to(out_dir)),
                "selected_mode_curves_csv": str(selected_modes_path.relative_to(out_dir)),
                "extreme_mode_mean_decay_csv": str(mean_decay_path.relative_to(out_dir)),
                "modewise_log10_snr_csv": str(modewise_snr_path.relative_to(out_dir)),
                "metrics_json": str((schedule_out / "forward_diagnostics_metrics.json").relative_to(out_dir)),
            }
        )

    summary_csv = out_dir / f"forward_diagnostics_summary_{sanitize_name(dataset_name)}.csv"
    summary_json = out_dir / f"forward_diagnostics_summary_{sanitize_name(dataset_name)}.json"
    manifest_path = out_dir / f"forward_plot_manifest_{sanitize_name(dataset_name)}.json"
    write_csv(summary_csv, rows)
    write_json(summary_json, {"rows": rows})

    theory_rows: list[dict[str, object]] = []
    row_by_schedule = {str(r["schedule"]): r for r in rows}
    if "MAS-SNR" in row_by_schedule and "MAS-SNR" in schedules:
        measured_mass = float(row_by_schedule["MAS-SNR"]["max_stiffness_proxy"])
        mass_schedule_only_upper = float(mass_M(schedules["MAS-SNR"]))  # B=0 because h_t,k=0
        for scalar_name in ("FCPS", "Uniform", "Cosine"):
            if scalar_name not in row_by_schedule or scalar_name not in schedules:
                continue
            scalar_theory = scalar_stiffness_lower_bound(
                scalar_name, schedules[scalar_name], lam_gamma_np, s_k, sigma
            )
            measured_scalar = float(row_by_schedule[scalar_name]["max_stiffness_proxy"])
            measured_ratio = measured_scalar / measured_mass if measured_mass > 0 else float("nan")
            predicted_min_ratio = guaranteed_improvement_ratio(
                float(scalar_theory["scalar_stiffness_lower_bound"]), mass_schedule_only_upper
            )
            theory_rows.append(
                {
                    "dataset": dataset_name,
                    "scalar_schedule": scalar_name,
                    "measured_L_scalar_h0": measured_scalar,
                    "measured_L_MASS_h0": measured_mass,
                    "measured_ratio_Lscalar_over_LMASS": measured_ratio,
                    **scalar_theory,
                    "mass_schedule_only_upper_bound_M": mass_schedule_only_upper,
                    "theory_ratio_lower_bound_scalarLower_over_massUpper": predicted_min_ratio,
                    "comparison_interpretation": "measured ratio versus conservative theorem-implied lower bound",
                }
            )
        theory_csv = out_dir / f"stiffness_theory_comparison_{sanitize_name(dataset_name)}.csv"
        theory_json = out_dir / f"stiffness_theory_comparison_{sanitize_name(dataset_name)}.json"
        theory_npz = out_dir / f"stiffness_theory_comparison_{sanitize_name(dataset_name)}.npz"
        write_csv(theory_csv, theory_rows)
        write_json(theory_json, {"rows": theory_rows})
        if theory_rows:
            write_npz(
                theory_npz,
                {
                    "scalar_schedule": np.asarray([r["scalar_schedule"] for r in theory_rows], dtype="U32"),
                    "measured_L_scalar_h0": np.asarray([r["measured_L_scalar_h0"] for r in theory_rows], dtype=np.float64),
                    "measured_L_MASS_h0": np.asarray([r["measured_L_MASS_h0"] for r in theory_rows], dtype=np.float64),
                    "measured_ratio": np.asarray([r["measured_ratio_Lscalar_over_LMASS"] for r in theory_rows], dtype=np.float64),
                    "scalar_lower_bound": np.asarray([r["scalar_stiffness_lower_bound"] for r in theory_rows], dtype=np.float64),
                    "mass_upper_bound": np.asarray([r["mass_schedule_only_upper_bound_M"] for r in theory_rows], dtype=np.float64),
                    "theory_ratio_lower_bound": np.asarray([r["theory_ratio_lower_bound_scalarLower_over_massUpper"] for r in theory_rows], dtype=np.float64),
                },
            )
    write_json(
        manifest_path,
        {
            "schema_version": 3,
            "dataset": dataset_name,
            "description": "Plot-ready forward arrays; no figures were generated during the experiment.",
            "diagnostic_roles": {
                "log_snr": "MASS equalization is a construction/implementation sanity check; scalar ranges describe path imbalance.",
                "schedule_only_stiffness_h0": "primary falsifiable quantity used for cross-scheduler theory comparison",
            },
            "entries": manifest_entries,
            "theory_comparison_rows": theory_rows,
        },
    )
    return rows


def print_forward_diagnostic_summary(row: dict[str, float | str]) -> None:
    schedule = row.get("schedule", "?")
    dataset = row.get("dataset", "?")
    print(f"\n[Forward diagnostics] dataset={dataset} schedule={schedule}")
    keys = [
        "mean_log_snr_std",
        "max_log_snr_range",
        "max_stiffness_proxy",
        "max_stiffness_optimal",
        "forward_to_stationary_kl_gaussian",
        "stationary_to_forward_kl_gaussian",
        "symmetric_kl_gaussian",
        "gaussian_w2_to_stationary",
        "cov_rel_err_to_stationary",
        "spectral_var_rel_err_to_stationary",
        "flattened_w1_to_stationary_samples",
        "mmd_rbf_z_to_stationary_samples",
        "terminal_swd_mean",
    ]
    for key in keys:
        if key in row:
            val = row[key]
            if isinstance(val, (float, int, np.floating, np.integer)):
                print(f"  {key:42s}: {float(val):.6g}")
            else:
                print(f"  {key:42s}: {val}")
    print(f"  plot-data NPZ: {row.get('forward_plot_npz')}")
    print(f"  Figure A CSV : {row.get('figure_a_timeseries_csv')}")
    print(f"  Mean-decay CSV: {row.get('extreme_mode_mean_decay_csv')}")
    print(f"  Log-SNR CSV   : {row.get('modewise_log10_snr_csv')}")


def spectral_power(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    Xtil = np.asarray(X, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    return np.mean(Xtil ** 2, axis=0)


def bandlimited_spectral_metrics(
    X_gen: np.ndarray,
    X_ref: np.ndarray,
    V: np.ndarray,
    low_frac: float = 0.25,
    high_frac: float = 0.25,
) -> dict[str, float]:
    X_gen = np.asarray(X_gen, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = V.shape[0]
    low_k = max(1, int(round(float(low_frac) * n)))
    high_k = max(1, int(round(float(high_frac) * n)))
    low = slice(0, low_k)
    high = slice(n - high_k, n)

    G = X_gen @ V
    R = X_ref @ V
    Pg = np.mean(G ** 2, axis=0)
    Pr = np.mean(R ** 2, axis=0)

    def rel_l1(sl: slice) -> float:
        return float(np.sum(np.abs(Pg[sl] - Pr[sl])) / (np.sum(np.abs(Pr[sl])) + EPS))

    def rel_l2(sl: slice) -> float:
        return float(np.linalg.norm(Pg[sl] - Pr[sl]) / (np.linalg.norm(Pr[sl]) + EPS))

    def ratio_gap(sl: slice) -> float:
        eg = float(np.sum(Pg[sl]) / (np.sum(Pg) + EPS))
        er = float(np.sum(Pr[sl]) / (np.sum(Pr) + EPS))
        return abs(eg - er)

    low_energy_g = np.sum(G[:, low] ** 2, axis=1)
    low_energy_r = np.sum(R[:, low] ** 2, axis=1)
    high_energy_g = np.sum(G[:, high] ** 2, axis=1)
    high_energy_r = np.sum(R[:, high] ** 2, axis=1)

    return {
        "low_band_k": float(low_k),
        "high_band_k": float(high_k),
        "low_band_power_rel_l1": rel_l1(low),
        "high_band_power_rel_l1": rel_l1(high),
        "low_band_power_rel_l2": rel_l2(low),
        "high_band_power_rel_l2": rel_l2(high),
        "low_band_energy_ratio_gap": ratio_gap(low),
        "high_band_energy_ratio_gap": ratio_gap(high),
        "low_band_energy_w1": float(wasserstein_distance(low_energy_g, low_energy_r)),
        "high_band_energy_w1": float(wasserstein_distance(high_energy_g, high_energy_r)),
        "full_spectrum_power_rel_l1": float(np.sum(np.abs(Pg - Pr)) / (np.sum(np.abs(Pr)) + EPS)),
    }


def export_generation_plot_data(
    X_gen: np.ndarray,
    X_ref: np.ndarray,
    V: np.ndarray,
    lam_gamma: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    low_frac: float = 0.25,
    high_frac: float = 0.25,
    include_samples: bool = True,
) -> dict[str, str]:
    ensure_dir(out_dir)
    V = np.asarray(V, dtype=np.float64)
    lam = np.asarray(lam_gamma, dtype=np.float64)
    X_gen = np.asarray(X_gen, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    n = V.shape[0]
    low_k = max(1, int(round(float(low_frac) * n)))
    high_k = max(1, int(round(float(high_frac) * n)))
    Pg = spectral_power(X_gen, V)
    Pr = spectral_power(X_ref, V)
    power_ratio = np.maximum(Pg, EPS) / np.maximum(Pr, EPS)
    log_ratio = np.log10(np.maximum(Pg, EPS) / np.maximum(Pr, EPS))
    power_difference = Pg - Pr
    absolute_power_error = np.abs(power_difference)
    relative_power_error = absolute_power_error / np.maximum(np.abs(Pr), EPS)
    band = np.full(n, "middle", dtype="U8")
    band[:low_k] = "low"
    band[n - high_k :] = "high"

    arrays: dict[str, np.ndarray] = {
        "mode_index": np.arange(n, dtype=np.int64),
        "lam_gamma": lam,
        "reference_power": Pr,
        "generated_power": Pg,
        "power_ratio": power_ratio,
        "log10_power_ratio": log_ratio,
        "power_difference": power_difference,
        "absolute_power_error": absolute_power_error,
        "relative_power_error": relative_power_error,
        "low_indices": np.arange(low_k, dtype=np.int64),
        "high_indices": np.arange(n - high_k, n, dtype=np.int64),
        "full_indices": np.arange(n, dtype=np.int64),
        "full_spectrum_indices": np.arange(n, dtype=np.int64),
        "normalized_frequency_rank": np.arange(n, dtype=np.float64) / max(1, n - 1),
        "band": band,
    }
    if include_samples:
        arrays["X_generated"] = X_gen
        arrays["X_reference"] = X_ref

    safe_prefix = sanitize_name(prefix)
    npz_path = Path(out_dir) / f"{safe_prefix}_generation_plot_data.npz"
    csv_path = Path(out_dir) / f"{safe_prefix}_spectrum.csv"
    full_spectrum_csv_path = Path(out_dir) / f"{safe_prefix}_full_spectrum_power_matching.csv"
    write_npz(npz_path, arrays)
    spectrum_rows = [
            {
                "mode": int(k),
                "normalized_frequency_rank": float(arrays["normalized_frequency_rank"][k]),
                "lam_gamma": float(lam[k]),
                "reference_power": float(Pr[k]),
                "generated_power": float(Pg[k]),
                "power_ratio": float(power_ratio[k]),
                "log10_power_ratio": float(log_ratio[k]),
                "power_difference": float(power_difference[k]),
                "absolute_power_error": float(absolute_power_error[k]),
                "relative_power_error": float(relative_power_error[k]),
                "band": str(band[k]),
            }
            for k in range(n)
        ]
    write_csv(csv_path, spectrum_rows)
    write_csv(full_spectrum_csv_path, spectrum_rows)
    return {
        "generation_plot_npz": str(npz_path),
        "generation_spectrum_csv": str(csv_path),
        "full_spectrum_plot_npz": str(npz_path),
        "full_spectrum_power_matching_csv": str(full_spectrum_csv_path),
    }


def generation_diagnostics(
    X_gen: np.ndarray,
    X_ref: np.ndarray,
    V: np.ndarray,
    lam_gamma: np.ndarray,
    out_dir: str | Path,
    prefix: str,
    low_frac: float = 0.25,
    high_frac: float = 0.25,
    include_samples: bool = True,
) -> dict[str, float | str]:
    metrics = bandlimited_spectral_metrics(
        X_gen=X_gen,
        X_ref=X_ref,
        V=V,
        low_frac=low_frac,
        high_frac=high_frac,
    )
    paths = export_generation_plot_data(
        X_gen=X_gen,
        X_ref=X_ref,
        V=V,
        lam_gamma=lam_gamma,
        out_dir=out_dir,
        prefix=prefix,
        low_frac=low_frac,
        high_frac=high_frac,
        include_samples=include_samples,
    )
    out: dict[str, float | str] = {**metrics, **paths}
    write_json(Path(out_dir) / f"{sanitize_name(prefix)}_generation_spectral_diagnostics.json", out)
    return out

