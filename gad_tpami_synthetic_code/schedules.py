from __future__ import annotations

import numpy as np
import torch


class FCPSSchedule:
    def __init__(self, T: float = 1.0, c0: float = 2300.0, c_min: float = 0.1, alpha: float = 4.0):
        self.T = float(T)
        self.c0 = float(c0)
        self.c_min = float(c_min)
        self.alpha = float(alpha)
        gap = self.c0 - self.c_min * self.T
        if gap <= 0:
            raise ValueError("Need c0 > c_min * T for FCPS.")
        self.k = gap * (self.alpha + 1.0) / self.T


class MASCorridor:
    def __init__(self, T: float = 1.0, r_max: float = 1e4, r_min: float = 1e-3):
        if not (r_max > r_min > 0):
            raise ValueError("Require r_max > r_min > 0.")
        self.T = float(T)
        self.r_max = float(r_max)
        self.r_min = float(r_min)


def scalar_a2_terminal(lam_gamma: np.ndarray, schedule: FCPSSchedule, T: float) -> np.ndarray:
    u = np.clip(T / schedule.T, 0.0, 1.0)
    cbar = schedule.c_min * T + (schedule.c0 - schedule.c_min * schedule.T) * (u ** (schedule.alpha + 1.0))
    return np.exp(-2.0 * np.asarray(lam_gamma, dtype=np.float64) * cbar)


def retained_signal_energy(a2: np.ndarray, s_k: np.ndarray) -> float:
    return float(np.mean(np.asarray(a2, dtype=np.float64) * np.asarray(s_k, dtype=np.float64)))


def solve_rmin_for_target_terminal_energy(
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    target_energy: float,
    rmin_lo: float = 1e-8,
    rmin_hi: float = 1e6,
    max_iter: int = 100,
) -> float:
    lam_gamma = np.asarray(lam_gamma, dtype=np.float64)
    s_k = np.asarray(s_k, dtype=np.float64)
    gk = lam_gamma * s_k / (sigma ** 2)

    def f(rmin: float) -> float:
        a2_T = rmin / (rmin + gk)
        return float(np.mean(a2_T * s_k))

    while f(rmin_hi) < target_energy:
        rmin_hi *= 10.0

    lo, hi = rmin_lo, rmin_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if f(mid) < target_energy:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def build_schedules(
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    T: float,
    fcps_c0: float,
    fcps_cmin: float,
    fcps_alpha: float,
    r_max: float,
    r_min: float | None,
    match_terminal_energy: bool,
) -> dict[str, object]:
    fcps = FCPSSchedule(T=T, c0=fcps_c0, c_min=fcps_cmin, alpha=fcps_alpha)
    if r_min is None:
        a2_fcps_T = scalar_a2_terminal(lam_gamma, fcps, T)
        target_energy = retained_signal_energy(a2_fcps_T, s_k)
        if match_terminal_energy:
            r_min = solve_rmin_for_target_terminal_energy(lam_gamma, s_k, sigma, target_energy)
        else:
            r_min = 1e-3
    mas = MASCorridor(T=T, r_max=r_max, r_min=r_min)
    return {"FCPS": fcps, "MAS-SNR": mas, "r_min": float(r_min)}


def channel_params_for_times_torch(
    schedule_name: str,
    schedule_obj,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t = t.to(dtype=lam_gamma.dtype)
    T = float(schedule_obj.T)

    if schedule_name == "FCPS":
        u = torch.clamp(t / T, 0.0, 1.0)
        c_min = float(schedule_obj.c_min)
        c0 = float(schedule_obj.c0)
        alpha = float(schedule_obj.alpha)
        cbar = c_min * t + (c0 - c_min * T) * (u ** (alpha + 1.0))
        k = float(schedule_obj.k)
        ct_scalar = c_min + k * (u ** alpha)
        a = torch.exp(-cbar[:, None] * lam_gamma[None, :])
        a2 = a ** 2
        nu = (sigma ** 2) * (1.0 - a2) / lam_gamma[None, :]
        ct = ct_scalar[:, None].expand(-1, lam_gamma.numel())
        cbar_full = cbar[:, None].expand(-1, lam_gamma.numel())
    elif schedule_name == "MAS-SNR":
        log_r_max = float(np.log(schedule_obj.r_max))
        log_r_min = float(np.log(schedule_obj.r_min))
        x = torch.clamp(t / T, 0.0, 1.0)
        s = 0.5 * (1.0 - torch.cos(torch.pi * x))
        sprime = 0.5 * (torch.pi / T) * torch.sin(torch.pi * x)
        log_r = log_r_max + (log_r_min - log_r_max) * s
        d_log_r_dt = (log_r_min - log_r_max) * sprime
        r = torch.exp(log_r)[:, None]
        s_k_safe = torch.clamp(s_k, min=1e-10)
        gk = lam_gamma * s_k_safe / (sigma ** 2)
        a2 = r / (r + gk[None, :])
        a = torch.sqrt(a2)
        nu = s_k_safe[None, :] / (r + gk[None, :])
        cbar_full = 0.5 / lam_gamma[None, :] * torch.log1p(gk[None, :] / r)
        ct = -(0.5 / lam_gamma[None, :]) * (1.0 - a2) * d_log_r_dt[:, None]
        ct = torch.clamp(ct, min=0.0)
    else:
        raise ValueError(f"Unknown schedule_name: {schedule_name}")

    rho = a2 * s_k[None, :] + nu
    return {"a": a, "a2": a2, "nu": nu, "rho": rho, "ct": ct, "cbar": cbar_full}
