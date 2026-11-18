from __future__ import annotations

import enum
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.fcps_schedule import calibrate_fcps, fcps_cbar


class ModelMeanType(enum.Enum):
    START_X = enum.auto()


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=tuple(range(1, x.ndim)))


class GADMASDiffusion(nn.Module):

    def __init__(
        self,
        graph_asset_path: str,
        steps: int,
        device: torch.device,
        scheduler: str = "mas_snr",
        objective: str = "x0_mse",
        sigma: float = 1.0,
        r_max: float = 1e4,
        r_min: float = 1e-4,
        t_eps: float = 1e-3,
        T_diff: float = 1.0,
        scalar_cbar_min: float = 1e-5,
        scalar_cbar_max: float = 6.0,
        fcps_alpha: float = 3.0,
        fcps_calibration: str = "snr_matched",
        fcps_cbar_T: Optional[float] = None,
        fcps_cmin: Optional[float] = None,
        reweight: bool = False,
        dsm_variance_weighted: bool = True,
        history_num_per_term: int = 10,
    ) -> None:
        super().__init__()
        if scheduler not in {"mas_snr", "scalar_gad", "fcps"}:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        if scheduler == "fcps" and reweight:
            raise ValueError("--mas_reweight uses the MASS SNR trajectory and is undefined for fcps")
        if objective not in {"x0_mse", "spectral_dsm"}:
            raise ValueError(f"Unknown objective: {objective}")
        if steps < 2:
            raise ValueError("GAD/MAS-SNR requires at least two diffusion steps")
        if not (0 < t_eps < T_diff):
            raise ValueError("Require 0 < t_eps < T_diff")
        if not (r_max > r_min > 0):
            raise ValueError("Require r_max > r_min > 0")

        self.mean_type = ModelMeanType.START_X
        self.steps = int(steps)
        self.device_ref = device
        self.scheduler = scheduler
        self.objective = objective
        self.sigma = float(sigma)
        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.t_eps = float(t_eps)
        self.T_diff = float(T_diff)
        self.reweight = bool(reweight)
        self.dsm_variance_weighted = bool(dsm_variance_weighted)
        self.history_num_per_term = int(history_num_per_term)

        assets = np.load(graph_asset_path, allow_pickle=False)
        V = torch.as_tensor(assets["eigvecs"], dtype=torch.float32)
        lam = torch.as_tensor(assets["eigvals_shifted"], dtype=torch.float32)
        s_real = torch.as_tensor(assets["s_real"], dtype=torch.float32)
        s_pseudo = torch.as_tensor(assets["s_pseudo"], dtype=torch.float32)
        if V.ndim != 2 or V.shape[0] != V.shape[1]:
            raise ValueError("Expected a square eigenvector matrix in graph assets")
        n_item = V.shape[0]
        if lam.shape[0] != n_item or s_real.shape[0] != n_item:
            raise ValueError("Graph asset arrays do not share the item dimension")

        self.n_item = int(n_item)
        self.register_buffer("V", V)
        self.register_buffer("VT", V.transpose(0, 1).contiguous())
        self.register_buffer("lambdas", lam.clamp_min(1e-12))
        self.register_buffer("s_real", s_real.clamp_min(1e-12))
        self.register_buffer("s_pseudo", s_pseudo.clamp_min(1e-12))
        self.register_buffer("time_grid", torch.linspace(self.t_eps, self.T_diff, self.steps))

        self.fcps_params: Dict[str, float] = {}
        if self.scheduler == "fcps":
            # One scalar schedule serves both branches. It is calibrated on the
            # real-user powers, the branch every launcher diffuses (--no-tf).
            self.fcps_params = calibrate_fcps(
                lambdas=self.lambdas.detach().cpu().double().numpy(),
                s=self.s_real.detach().cpu().double().numpy(),
                sigma=self.sigma, r_max=self.r_max, r_min=self.r_min,
                t_eps=self.t_eps, T_diff=self.T_diff, alpha=fcps_alpha,
                calibration=fcps_calibration, cbar_T=fcps_cbar_T, c_min=fcps_cmin,
            )

        self._prepare_schedule_tables(scalar_cbar_min, scalar_cbar_max)
        self.register_buffer("Lt_history", torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float32))
        self.register_buffer("Lt_count", torch.zeros(self.steps, dtype=torch.long))

        try:
            metadata = str(assets["diagnostics_json"].item())
            self.graph_diagnostics = json.loads(metadata)
        except Exception:
            self.graph_diagnostics = {}

    def _prepare_schedule_tables(self, scalar_cbar_min: float, scalar_cbar_max: float) -> None:
        u = (self.time_grid - self.t_eps) / (self.T_diff - self.t_eps)
        log_r = (1.0 - u) * math.log(self.r_max) + u * math.log(self.r_min)
        r = torch.exp(log_r)
        self.register_buffer("r_table", r)

        def branch_tables(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.scheduler == "mas_snr":
                g = self.lambdas * s / (self.sigma ** 2)
                a = torch.sqrt(r[:, None] / (r[:, None] + g[None, :]))
                nu = s[None, :] / (r[:, None] + g[None, :])
                cbar = 0.5 * torch.log1p(g[None, :] / r[:, None]) / self.lambdas[None, :]
            elif self.scheduler == "fcps":
                # Float64 with expm1: near t0 the attenuation is 1 - O(1e-7), and
                # 1 - a^2 would lose most of its digits in float32.
                p = self.fcps_params
                cbar_scalar = fcps_cbar(self.time_grid.double(), p["fcps_c_min"], p["fcps_cbar_T"],
                                        p["fcps_alpha"], self.T_diff)
                lam64 = self.lambdas.double()
                expo = cbar_scalar[:, None] * lam64[None, :]
                a = torch.exp(-expo)
                nu = (self.sigma ** 2) * (-torch.expm1(-2.0 * expo)) / lam64[None, :]
                cbar = cbar_scalar[:, None].expand(-1, self.n_item)
            else:
                cbar_scalar = scalar_cbar_min + u * (scalar_cbar_max - scalar_cbar_min)
                cbar = cbar_scalar[:, None].expand(-1, self.n_item)
                a = torch.exp(-cbar * self.lambdas[None, :])
                nu = (self.sigma ** 2) * (1.0 - a.square()) / self.lambdas[None, :]
            return a.float(), nu.clamp_min(1e-12).float(), cbar.float()

        a_real, nu_real, cbar_real = branch_tables(self.s_real)
        a_pseudo, nu_pseudo, cbar_pseudo = branch_tables(self.s_pseudo)
        self.register_buffer("a_real", a_real)
        self.register_buffer("nu_real", nu_real)
        self.register_buffer("cbar_real", cbar_real)
        self.register_buffer("a_pseudo", a_pseudo)
        self.register_buffer("nu_pseudo", nu_pseudo)
        self.register_buffer("cbar_pseudo", cbar_pseudo)

    def _tables(self, branch: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if branch == "real":
            return self.a_real, self.nu_real, self.cbar_real
        if branch == "pseudo":
            return self.a_pseudo, self.nu_pseudo, self.cbar_pseudo
        raise ValueError(f"Unknown branch: {branch}")

    def graph_fourier(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.V

    def inverse_graph_fourier(self, x_tilde: torch.Tensor) -> torch.Tensor:
        return x_tilde @ self.VT

    def coefficients(self, t: torch.Tensor, branch: str = "real") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, nu, cbar = self._tables(branch)
        return a[t], nu[t], cbar[t]

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, branch: str = "real", noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x0_spec = self.graph_fourier(x_start)
        if noise is None:
            noise = torch.randn_like(x0_spec)
        a, nu, _ = self.coefficients(t, branch)
        xt_spec = a * x0_spec + torch.sqrt(nu) * noise
        return self.inverse_graph_fourier(xt_spec), xt_spec

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        method: str = "uniform",
        uniform_prob: float = 1e-3,
        low_time_prob: float = 0.70,
        low_time_max_step: int = 3,
    ):
        low_max = int(max(0, min(low_time_max_step, self.steps - 1)))
        if method == "low_time_only":
            t = torch.randint(0, low_max + 1, (batch_size,), device=device).long()
            return t, torch.ones(batch_size, device=device)
        if method == "low_time_mix":
            low = torch.randint(0, low_max + 1, (batch_size,), device=device).long()
            uni = torch.randint(0, self.steps, (batch_size,), device=device).long()
            choose_low = torch.rand(batch_size, device=device) < float(low_time_prob)
            t = torch.where(choose_low, low, uni)
            return t, torch.ones(batch_size, device=device)
        if method == "importance" and torch.all(self.Lt_count == self.history_num_per_term):
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history.square(), dim=-1)).clamp_min(1e-12)
            probs = Lt_sqrt / Lt_sqrt.sum()
            probs = (1.0 - uniform_prob) * probs + uniform_prob / len(probs)
            t = torch.multinomial(probs, num_samples=batch_size, replacement=True).to(device)
            pt = probs.to(device).gather(0, t) * len(probs)
            return t.long(), pt.float()
        t = torch.randint(0, self.steps, (batch_size,), device=device).long()
        return t, torch.ones(batch_size, device=device)

    def get_output(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        branch: str = "real",
        t_sampling: str = "uniform",
        low_time_prob: float = 0.70,
        low_time_max_step: int = 3,
    ) -> Dict[str, torch.Tensor]:
        ts, pt = self.sample_timesteps(
            x_start.size(0),
            x_start.device,
            method=t_sampling,
            low_time_prob=low_time_prob,
            low_time_max_step=low_time_max_step,
        )
        x_t, x_t_spec = self.q_sample(x_start, ts, branch=branch)
        pred_x0 = model(x_t, ts).float()
        return {
            "target_x0": x_start.float(),
            "pred_x0": pred_x0,
            "ts": ts,
            "pt": pt,
            "x_t": x_t,
            "x_t_spec": x_t_spec,
            "branch": branch,
        }

    def _corridor_weights(self, ts: torch.Tensor) -> torch.Tensor:
        # Normalized positive analogue of the SNR-difference reweighting used in DiffRec.
        prev = torch.clamp(ts - 1, min=0)
        raw = (self.r_table[prev] - self.r_table[ts]).abs()
        raw = torch.where(ts == 0, torch.ones_like(raw), raw)
        return raw / raw.mean().clamp_min(1e-12)

    def get_losses(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, context: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ts = context["ts"]
        pt = context["pt"]
        branch = context["branch"]
        if self.objective == "x0_mse":
            per_sample = mean_flat((target_x0.float() - pred_x0.float()).square())
        else:
            a, nu, _ = self.coefficients(ts, branch)
            target_spec = self.graph_fourier(target_x0)
            pred_spec = self.graph_fourier(pred_x0)
            x_t_spec = context["x_t_spec"]
            target_score = (a * target_spec - x_t_spec) / nu
            pred_score = (a * pred_spec - x_t_spec) / nu
            sqerr = (pred_score - target_score).square()
            if self.dsm_variance_weighted:
                sqerr = nu * sqerr
            per_sample = mean_flat(sqerr)
        if self.reweight:
            per_sample = self._corridor_weights(ts) * per_sample

        weighted = per_sample / pt
        with torch.no_grad():
            for ti, li in zip(ts.detach().cpu(), per_sample.detach().cpu()):
                idx = int(ti)
                count = int(self.Lt_count[idx])
                if count < self.history_num_per_term:
                    self.Lt_history[idx, count] = li
                    self.Lt_count[idx] += 1
                else:
                    self.Lt_history[idx, :-1] = self.Lt_history[idx, 1:].clone()
                    self.Lt_history[idx, -1] = li
        return {"loss": weighted, "per_sample": per_sample}

    def p_sample(
        self,
        model: nn.Module,
        x_condition: torch.Tensor,
        sampling_steps: int = 0,
        sampling_noise: bool = False,
        branch: str = "real",
        inference_mode: str = "terminal_prior",
        partial_start_step: int = 0,
    ) -> torch.Tensor:
        B = x_condition.shape[0]
        if inference_mode in {"clean_direct", "paper_compatible"}:
            t0 = torch.zeros(B, dtype=torch.long, device=x_condition.device)
            return model(x_condition.float(), t0).float()
        if inference_mode in {"terminal_prior", "public_code_compatible"}:
            start_idx = self.steps - 1
        elif inference_mode == "partial_corruption":
            start_idx = int(max(0, min(partial_start_step, self.steps - 1)))
        else:
            raise ValueError(f"Unknown inference_mode: {inference_mode}")
        if start_idx <= 0:
            t0 = torch.zeros(B, dtype=torch.long, device=x_condition.device)
            return model(x_condition.float(), t0).float()
        t_start = torch.full((B,), start_idx, dtype=torch.long, device=x_condition.device)
        x_t, x_spec = self.q_sample(x_condition, t_start, branch=branch)
        a_tab, nu_tab, cbar_tab = self._tables(branch)

        if sampling_steps <= 0 or sampling_steps >= start_idx:
            reverse_grid = list(range(start_idx, -1, -1))
        else:
            grid = np.linspace(start_idx, 0, num=sampling_steps + 1)
            reverse_grid = [int(round(v)) for v in grid]
            # Remove duplicate rounded indices while retaining descending order.
            reverse_grid = list(dict.fromkeys(reverse_grid))
            if reverse_grid[-1] != 0:
                reverse_grid.append(0)

        for i, j in zip(reverse_grid[:-1], reverse_grid[1:]):
            t = torch.full((B,), i, dtype=torch.long, device=x_condition.device)
            pred_x0 = model(x_t, t).float()
            pred_spec = self.graph_fourier(pred_x0)
            a = a_tab[i][None, :]
            nu = nu_tab[i][None, :]
            score_spec = (a * pred_spec - x_spec) / nu
            delta_t = float(self.time_grid[i] - self.time_grid[j])
            c_i = (cbar_tab[i] - cbar_tab[j]) / max(delta_t, 1e-12)
            drift = self.lambdas[None, :] * c_i[None, :] * x_spec + 2.0 * c_i[None, :] * (self.sigma ** 2) * score_spec
            x_spec = x_spec + drift * delta_t
            if sampling_noise and j > 0:
                z = torch.randn_like(x_spec)
                x_spec = x_spec + torch.sqrt((2.0 * c_i * (self.sigma ** 2) * delta_t).clamp_min(0.0))[None, :] * z
            x_t = self.inverse_graph_fourier(x_spec)

        t0 = torch.zeros(B, dtype=torch.long, device=x_condition.device)
        return model(x_t, t0).float()

    def diagnostics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for label, idx in (("start", 0), ("mid", self.steps // 2), ("end", self.steps - 1)):
            a = self.a_real[idx]
            nu = self.nu_real[idx]
            # The floor only matters for fcps, whose high modes can underflow to
            # zero SNR at the terminal; MASS values are far above it.
            snr = (a.square() * self.s_real / nu).clamp_min(torch.finfo(torch.float32).tiny)
            rho = a.square() * self.s_real + nu
            out[f"{label}_snr_log10_spread"] = float(torch.log10(snr.max()) - torch.log10(snr.min()))
            out[f"{label}_rho_log10_spread"] = float(torch.log10(rho.max()) - torch.log10(rho.min()))
        out.update(self.fcps_params)
        return out
