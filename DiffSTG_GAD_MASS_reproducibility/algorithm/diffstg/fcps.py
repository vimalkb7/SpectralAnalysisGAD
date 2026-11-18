from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .mas_snr import MASSNR


class FCPS(MASSNR):
    """Scalar GAD channel with the floor-constrained polynomial schedule."""

    ENDPOINT_RULES = ("match_mass", "delta")
    GRIDS = ("uniform", "logsnr")

    def __init__(
        self,
        adj: np.ndarray,
        train_data: np.ndarray,
        num_steps: int,
        device: torch.device,
        fcps_alpha: float = 2.0,
        fcps_c_min: float = 0.5,
        fcps_endpoints: str = "match_mass",
        fcps_delta: float = 1e-3,
        fcps_cbar_T: Optional[float] = None,
        fcps_cbar_t0: Optional[float] = None,
        **mass_kwargs,
    ) -> None:
        # Builds V, lambda_gamma, sk, gk exactly as MASS does.  r_max/r_min are
        # kept only as the reference used by endpoints='match_mass'.
        super().__init__(adj=adj, train_data=train_data, num_steps=num_steps, device=device, **mass_kwargs)

        if fcps_alpha < 0:
            raise ValueError(f"fcps_alpha must be >= 0 (the GAD paper uses alpha > 1), got {fcps_alpha}")
        if fcps_c_min < 0:
            raise ValueError(f"fcps_c_min must be >= 0, got {fcps_c_min}")
        if fcps_endpoints not in self.ENDPOINT_RULES:
            raise ValueError(f"fcps_endpoints must be one of {self.ENDPOINT_RULES}, got {fcps_endpoints}")
        if fcps_endpoints == "delta" and not (0.0 < fcps_delta < 1.0):
            raise ValueError(f"fcps_delta must lie in (0,1), got {fcps_delta}")

        self.fcps_alpha = float(fcps_alpha)
        self.fcps_c_min = float(fcps_c_min)
        self.fcps_endpoints = str(fcps_endpoints)
        self.fcps_delta = float(fcps_delta)

        lam = self.lambda_gamma.detach().cpu().double().numpy()
        g = self.gk.detach().cpu().double().numpy()
        sk = self.sk.detach().cpu().double().numpy()
        lam_min, lam_max = float(lam.min()), float(lam.max())

        if self.fcps_endpoints == "match_mass":
            cbar_t0 = math.log1p(float(g.max()) / self.r_max) / (2.0 * lam_max)
            cbar_T = math.log1p(float(g.min()) / self.r_min) / (2.0 * lam_min)
        else:
            cbar_t0 = -math.log1p(-self.fcps_delta) / (2.0 * lam_max)
            cbar_T = math.log(1.0 / self.fcps_delta) / (2.0 * lam_min)
        if fcps_cbar_t0 is not None:
            cbar_t0 = float(fcps_cbar_t0)
        if fcps_cbar_T is not None:
            cbar_T = float(fcps_cbar_T)

        if not (cbar_T > cbar_t0 > 0.0):
            raise ValueError(f"FCPS requires cbar_T > cbar_t0 > 0, got cbar_T={cbar_T}, cbar_t0={cbar_t0}")
        if cbar_T < self.fcps_c_min:
            raise ValueError(
                f"FCPS requires cbar_T >= c_min*T (T=1); got cbar_T={cbar_T:.4g} < c_min={self.fcps_c_min:.4g}. "
                "Lower --fcps_c_min."
            )
        self.fcps_cbar_t0 = float(cbar_t0)
        self.fcps_cbar_T = float(cbar_T)
        self.fcps_k = (self.fcps_cbar_T - self.fcps_c_min) * (self.fcps_alpha + 1.0)
        self.fcps_t0 = self._solve_t0(self.fcps_cbar_t0)

        # Discrete channel tables, computed in float64 (expm1 avoids the
        # cancellation in 1 - a^2 at the least-corrupted levels).
        t_idx = self.index_to_time(np.arange(self.num_steps, dtype=np.float64))
        cbar = self.cbar_of_time(t_idx)                               # (N,)
        two_lc = 2.0 * cbar[:, None] * lam[None, :]                   # (N,V)
        a = np.exp(-0.5 * two_lc)
        one_minus_a2 = -np.expm1(-two_lc)
        nu = np.maximum((self.sigma ** 2) / lam[None, :] * one_minus_a2, self.eps)
        log_snr = np.log(sk)[None, :] - np.log((self.sigma ** 2) / lam)[None, :] - two_lc - np.log(one_minus_a2)

        self.register_buffer("fcps_cbar", torch.tensor(cbar, dtype=torch.float32, device=device))
        self.register_buffer("fcps_a", torch.tensor(a, dtype=torch.float32, device=device))
        self.register_buffer("fcps_nu", torch.tensor(nu, dtype=torch.float32, device=device))
        self._log_snr_table = log_snr                                 # float64, natural log, (N,V)
        self._mean_log_snr = log_snr.mean(axis=1)                     # strictly decreasing in index

    # ------------------------------------------------------------------ schedule
    def cbar_of_time(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.fcps_c_min * t + (self.fcps_cbar_T - self.fcps_c_min) * t ** (self.fcps_alpha + 1.0)

    def rate_of_time(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        return self.fcps_c_min + self.fcps_k * t ** self.fcps_alpha

    def index_to_time(self, i: np.ndarray) -> np.ndarray:
        u = np.asarray(i, dtype=np.float64) / float(self.num_steps - 1)
        return self.fcps_t0 + (1.0 - self.fcps_t0) * u

    def _solve_t0(self, target: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(self.cbar_of_time(mid)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------- MASSNR interface
    def r_of_t(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "FCPS has no common SNR trajectory r_t (the SNR differs per mode). "
            "Use --mas_time_loss_weighting none with --diffusion_scheduler fcps."
        )

    def coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (a, nu, snr), each (B, V).  Callers in DiffSTG use only a and nu."""
        t = t.long().to(self.fcps_a.device)
        a = self.fcps_a[t]
        nu = self.fcps_nu[t]
        snr = (a ** 2) * self.sk[None, :] / nu
        return a, nu, snr

    def sample_sequence(self, sample_steps: int, grid: str = "uniform"):
        if grid not in self.GRIDS:
            raise ValueError(f"Unknown FCPS sampling grid={grid}; choose from {self.GRIDS}")
        if grid == "uniform":
            return super().sample_sequence(sample_steps, "uniform")
        n = self.num_steps
        if sample_steps < 2 or sample_steps > n:
            raise ValueError(f"sample_steps must be in [2,{n}], got {sample_steps}")
        m = self._mean_log_snr
        targets = np.linspace(m[0], m[-1], sample_steps)
        idx = [int(np.argmin(np.abs(m - tg))) for tg in targets]
        idx[0], idx[-1] = 0, n - 1
        for j in range(1, sample_steps):
            idx[j] = max(idx[j], idx[j - 1] + 1)
        for j in range(sample_steps - 2, -1, -1):
            idx[j] = min(idx[j], idx[j + 1] - 1)
        if idx[0] != 0 or idx[-1] != n - 1 or len(set(idx)) != sample_steps:
            raise RuntimeError(f"FCPS logsnr grid construction failed: {idx}")
        return idx

    def diagnostics(self) -> Dict[str, object]:
        with torch.no_grad():
            lam = self.lambda_gamma.detach().cpu().double().numpy()
            a0 = self.fcps_a[0].detach().cpu().double().numpy()
            aT = self.fcps_a[-1].detach().cpu().double().numpy()
            nuT = self.fcps_nu[-1].detach().cpu().double().numpy()
            frac = nuT / ((self.sigma ** 2) / lam)
            gmin, gmax = float(self.gk.min()), float(self.gk.max())
            spread = (self._log_snr_table.max(axis=1) - self._log_snr_table.min(axis=1)) / math.log(10.0)
            q = np.linspace(0, self.num_steps - 1, 5).round().astype(int)
            return {
                "scheduler": "fcps",
                "power_mode": self.power_mode,
                "power_meta": self.power_meta,
                "gamma": self.gamma,
                "sigma": self.sigma,
                "s_min_value": self.s_min_value,
                "sk_raw_min": float(self.sk_raw.min()),
                "sk_raw_median": float(self.sk_raw.median()),
                "sk_raw_max": float(self.sk_raw.max()),
                "g_min": gmin,
                "g_max": gmax,
                "lambda_gamma_min": float(lam.min()),
                "lambda_gamma_max": float(lam.max()),
                "cond_L_gamma": float(lam.max() / lam.min()),
                "fcps_alpha": self.fcps_alpha,
                "fcps_c_min": self.fcps_c_min,
                "fcps_k": self.fcps_k,
                "fcps_endpoints": self.fcps_endpoints,
                "fcps_delta": self.fcps_delta if self.fcps_endpoints == "delta" else None,
                "fcps_t0": self.fcps_t0,
                "fcps_cbar_t0": self.fcps_cbar_t0,
                "fcps_cbar_T": self.fcps_cbar_T,
                "mass_reference_r_max": self.r_max,
                "mass_reference_r_min": self.r_min,
                "min_initial_a2": float((a0 ** 2).min()),
                "max_terminal_a2": float((aT ** 2).max()),
                "min_terminal_prior_variance_fraction": float(frac.min()),
                "max_terminal_prior_variance_fraction": float(frac.max()),
                "log10_snr_spread_across_modes_at_index": {int(i): float(spread[i]) for i in q},
            }
