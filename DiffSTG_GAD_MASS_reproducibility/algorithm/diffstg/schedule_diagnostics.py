from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .fcps import FCPS
from .mas_snr import MASSNR


def _np(x) -> np.ndarray:
    return x.detach().cpu().double().numpy()


def continuous_schedule(sched: MASSNR, u: np.ndarray):
    u = np.asarray(u, dtype=np.float64)
    lam = _np(sched.lambda_gamma)
    if isinstance(sched, FCPS):
        t = sched.fcps_t0 + (1.0 - sched.fcps_t0) * u
        cbar = np.repeat(sched.cbar_of_time(t)[:, None], lam.size, axis=1)
        rate = np.repeat(((1.0 - sched.fcps_t0) * sched.rate_of_time(t))[:, None], lam.size, axis=1)
        return cbar, rate
    g = _np(sched.gk)
    log_rmax, log_rmin = math.log(sched.r_max), math.log(sched.r_min)
    r = np.exp((1.0 - u) * log_rmax + u * log_rmin)[:, None]
    cbar = np.log1p(g[None, :] / r) / (2.0 * lam[None, :])
    dlogr_du = log_rmin - log_rmax
    rate = -(1.0 / (2.0 * lam[None, :])) * (g[None, :] / (r + g[None, :])) * dlogr_du
    return cbar, rate


def _discrete_log_snr(sched: MASSNR) -> np.ndarray:
    lam = _np(sched.lambda_gamma)
    p = _np(sched.sk)
    u = np.arange(sched.num_steps, dtype=np.float64) / float(sched.num_steps - 1)
    cbar, _ = continuous_schedule(sched, u)
    two_lc = 2.0 * lam[None, :] * cbar
    return (np.log(p) - np.log((sched.sigma ** 2) / lam))[None, :] - two_lc - np.log(-np.expm1(-two_lc))


def schedule_report(sched: MASSNR, eps: float = 0.1, n_grid: int = 20001) -> Dict[str, object]:
    lam = _np(sched.lambda_gamma)
    k_lo, k_hi = int(np.argmin(lam)), int(np.argmax(lam))

    log_snr = _discrete_log_snr(sched)
    spread10 = (log_snr.max(axis=1) - log_snr.min(axis=1)) / math.log(10.0)
    lo, hi = math.log(eps), math.log(1.0 / eps)
    informative = ((log_snr >= lo) & (log_snr <= hi)).mean(axis=0)            # (V,)
    any_informative = ((log_snr >= lo) & (log_snr <= hi)).any(axis=1).mean()

    # Dense near u = 0, where a scalar schedule's rate factor peaks.
    u = np.unique(np.concatenate([[0.0], np.geomspace(1e-9, 1e-2, 2001), np.linspace(0.0, 1.0, n_grid)]))
    cbar, rate = continuous_schedule(sched, u)
    two_lc = 2.0 * lam[None, :] * cbar
    a2 = np.exp(-two_lc)
    one_minus = -np.expm1(-two_lc)
    zeta0 = rate * lam[None, :] * (1.0 + a2) / one_minus
    iu, ik = np.unravel_index(int(np.argmax(zeta0)), zeta0.shape)
    zeta_T = zeta0[-1]

    levels = np.linspace(0, sched.num_steps - 1, 5).round().astype(int)
    return {
        "scheduler": "fcps" if isinstance(sched, FCPS) else "mas_snr",
        "num_levels": int(sched.num_steps),
        "cond_L_gamma": float(lam.max() / lam.min()),
        "log10_snr_spread_across_modes": {
            "mean_over_levels": float(spread10.mean()),
            "max_over_levels": float(spread10.max()),
            "at_level": {int(i): float(spread10[i]) for i in levels},
        },
        "informative_eps": float(eps),
        "informative_fraction_of_levels": {
            "min_over_modes": float(informative.min()),
            "median_over_modes": float(np.median(informative)),
            "max_over_modes": float(informative.max()),
            "lowest_graph_frequency": float(informative[k_lo]),
            "highest_graph_frequency": float(informative[k_hi]),
            "levels_with_any_informative_mode": float(any_informative),
        },
        "schedule_only_stiffness": {
            "L_S0": float(zeta0[iu, ik]),
            "argmax_u": float(u[iu]),
            "argmax_lambda_gamma": float(lam[ik]),
            "max_over_modes_at_terminal": float(zeta_T.max()),
            "note": "sup_{u,k} c lambda (1+a^2)/(1-a^2) at h=0, normalized horizon u in [0,1]",
        },
    }
