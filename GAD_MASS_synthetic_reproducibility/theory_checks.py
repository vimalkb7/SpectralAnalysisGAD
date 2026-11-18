from __future__ import annotations

import math

import numpy as np
import torch

from schedules import MASCorridor, scalar_clock_for_times_torch, channel_params_for_times_torch


def mass_r_at_time(schedule: MASCorridor, t: float) -> float:
    x = min(max(float(t) / float(schedule.T), 0.0), 1.0)
    u = 0.5 * (1.0 - math.cos(math.pi * x))
    return float(math.exp((1-u)*math.log(schedule.r_max) + u*math.log(schedule.r_min)))


def mass_M(schedule: MASCorridor) -> float:
    return float(0.5 * math.pi / float(schedule.T) * math.log(schedule.r_max / schedule.r_min))


def mass_B_optimal(schedule: MASCorridor, t0: float) -> float:
    r = mass_r_at_time(schedule, t0)
    return float(r / (1.0 + r))


def mass_stiffness_upper_bound(schedule: MASCorridor, t0: float) -> float:
    return float(mass_M(schedule) * (1.0 + mass_B_optimal(schedule, t0)))


def scalar_terminal_rate(schedule_name: str, schedule_obj) -> float:
    t = torch.tensor([float(schedule_obj.T)], dtype=torch.float64)
    _, ct = scalar_clock_for_times_torch(schedule_name, schedule_obj, t)
    return float(ct.item())


def scalar_stiffness_lower_bound(
    schedule_name: str,
    schedule_obj,
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
) -> dict[str, float | bool]:
    lam = np.asarray(lam_gamma, dtype=np.float64)
    p = np.asarray(s_k, dtype=np.float64)
    t = torch.tensor([float(schedule_obj.T)], dtype=torch.float64)
    pars = channel_params_for_times_torch(
        schedule_name,
        schedule_obj,
        torch.tensor(lam, dtype=torch.float64),
        torch.tensor(p, dtype=torch.float64),
        float(sigma),
        t,
    )
    snr_high = float((pars["a2"][0, -1] * p[-1] / torch.clamp(pars["nu"][0, -1], min=1e-30)).item())
    c_T = scalar_terminal_rate(schedule_name, schedule_obj)
    lower = 0.5 * c_T * float(lam[-1])
    return {
        "scalar_c_T": c_T,
        "terminal_high_mode_snr": snr_high,
        "scalar_theorem_condition_snr_le_1_over_3": bool(snr_high <= 1.0 / 3.0),
        "scalar_stiffness_lower_bound": float(lower),
    }


def guaranteed_improvement_ratio(scalar_lower: float, mass_upper: float) -> float:
    if mass_upper <= 0:
        return float("nan")
    return float(scalar_lower / mass_upper)
