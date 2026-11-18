from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

_MAX_EXPONENT = 700.0  # keeps expm1 finite in float64


def mode_snr(cbar: float, lambdas: np.ndarray, g: np.ndarray) -> np.ndarray:
    expo = np.minimum(2.0 * lambdas * float(cbar), _MAX_EXPONENT)
    return g / np.expm1(expo)


def _solve_decreasing(fn: Callable[[float], float], target: float,
                      lo: float = 1e-16, hi: float = 1e8, iters: int = 200) -> float:
    if not (fn(hi) <= target <= fn(lo)):
        raise ValueError(f"target {target:g} is outside the bracket [{fn(hi):g}, {fn(lo):g}]")
    log_lo, log_hi = math.log(lo), math.log(hi)
    for _ in range(iters):
        mid = 0.5 * (log_lo + log_hi)
        if fn(math.exp(mid)) > target:
            log_lo = mid
        else:
            log_hi = mid
    return math.exp(log_hi)


def fcps_cbar(t, c_min: float, cbar_T: float, alpha: float, T: float):
    return c_min * t + (cbar_T - c_min * T) * (t / T) ** (alpha + 1.0)


def fcps_rate(t, c_min: float, cbar_T: float, alpha: float, T: float):
    return c_min + (cbar_T - c_min * T) * (alpha + 1.0) * (t / T) ** alpha / T


def calibrate_fcps(
    lambdas: np.ndarray,
    s: np.ndarray,
    sigma: float,
    r_max: float,
    r_min: float,
    t_eps: float,
    T_diff: float,
    alpha: float,
    calibration: str = "snr_matched",
    cbar_T: Optional[float] = None,
    c_min: Optional[float] = None,
) -> Dict[str, float]:
    lam = np.asarray(lambdas, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    g = lam * s / float(sigma) ** 2
    alpha = float(alpha)
    t0, T = float(t_eps), float(T_diff)
    if alpha < 0:
        raise ValueError("--fcps_alpha must be nonnegative")

    max_snr = lambda c: float(mode_snr(c, lam, g).max())
    min_snr = lambda c: float(mode_snr(c, lam, g).min())

    # Values the MASS endpoints imply, reported in both modes for reference.
    cbar_T_matched = _solve_decreasing(max_snr, float(r_min))
    cbar_t0_matched = _solve_decreasing(min_snr, float(r_max))
    alpha_min = math.log(cbar_T_matched / cbar_t0_matched) / math.log(T / t0) - 1.0

    if calibration == "snr_matched":
        cbar_T = cbar_T_matched
        rho = (t0 / T) ** (alpha + 1.0)
        denom = t0 - T * rho  # = t0 * (1 - (t0/T)^alpha), zero at alpha = 0
        numer = cbar_t0_matched - cbar_T * rho
        if denom <= 0.0 or numer < 0.0:
            raise ValueError(
                f"FCPS with alpha={alpha:g} cannot match the MASS start SNR r_max={r_max:g}: "
                f"its polynomial term alone corrupts the first grid point beyond that level. "
                f"Use --fcps_alpha >= {alpha_min:.3f} on this graph, or --fcps_calibration manual."
            )
        c_min = numer / denom
    elif calibration == "manual":
        if cbar_T is None or c_min is None:
            raise ValueError("--fcps_calibration manual requires --fcps_cbar_T and --fcps_cmin")
        cbar_T, c_min = float(cbar_T), float(c_min)
    else:
        raise ValueError(f"Unknown FCPS calibration: {calibration}")

    if c_min < 0.0:
        raise ValueError("FCPS requires c_min >= 0")
    if cbar_T < c_min * T:
        raise ValueError("FCPS requires cbar_T >= c_min * T so that the polynomial term is nondecreasing")

    cbar_t0 = float(fcps_cbar(t0, c_min, cbar_T, alpha, T))
    return {
        "fcps_alpha": alpha,
        "fcps_c_min": float(c_min),
        "fcps_cbar_T": float(cbar_T),
        "fcps_cbar_t0": cbar_t0,
        "fcps_alpha_min_for_start_match": float(alpha_min),
        "fcps_start_min_snr": min_snr(cbar_t0),
        "fcps_start_max_snr": max_snr(cbar_t0),
        "fcps_terminal_min_snr": min_snr(cbar_T),
        "fcps_terminal_max_snr": max_snr(cbar_T),
        "fcps_matched_r_max": float(r_max),
        "fcps_matched_r_min": float(r_min),
    }
