from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class MASSNR(torch.nn.Module):
    """Graph-spectral MASS corruption for tensors shaped (B,F,V,T)."""

    def __init__(
        self,
        adj: np.ndarray,
        train_data: np.ndarray,
        num_steps: int,
        device: torch.device,
        gamma: float = 1e-2,
        sigma: float = 1.0,
        r_max: float = 1e2,
        r_min: float = 1e-3,
        s_min: Optional[float] = None,
        s_min_quantile: float = 0.05,
        power_mode: str = "centered",
        conditional_history: int = 12,
        conditional_horizon: int = 12,
        conditional_ridge: float = 1e-3,
        conditional_stride: int = 1,
        graph_symmetrization: str = "max",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_steps < 2:
            raise ValueError("MASS requires num_steps >= 2")
        if not (r_max > r_min > 0):
            raise ValueError(f"Require r_max > r_min > 0, got {r_max}, {r_min}")
        if sigma <= 0 or gamma <= 0:
            raise ValueError("sigma and gamma must be positive")
        if power_mode not in {"second_moment", "centered", "conditional_residual"}:
            raise ValueError(f"Unknown power_mode={power_mode}")

        self.num_steps = int(num_steps)
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.eps = float(eps)
        self.power_mode = str(power_mode)
        self.conditional_history = int(conditional_history)
        self.conditional_horizon = int(conditional_horizon)
        self.conditional_ridge = float(conditional_ridge)
        self.conditional_stride = int(max(1, conditional_stride))

        adj = np.asarray(adj, dtype=np.float64)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square, got shape {adj.shape}")
        self.num_nodes = int(adj.shape[0])

        adj_sym = self._symmetrize(adj, graph_symmetrization)
        lap = self._normalized_laplacian(adj_sym)
        evals, evecs = np.linalg.eigh(lap)
        evals = np.maximum(evals, 0.0)
        lambda_gamma = evals + self.gamma

        x = self._as_time_node(train_data)
        sk_raw, power_meta = self._estimate_spectral_power(x, evecs)
        if s_min is None:
            positive = sk_raw[sk_raw > 0]
            s_min_value = float(np.quantile(positive, s_min_quantile)) if positive.size else 1e-4
        else:
            s_min_value = float(s_min)
        s_min_value = max(s_min_value, self.eps)
        sk = np.maximum(sk_raw, s_min_value)
        gk = lambda_gamma * sk / (self.sigma ** 2)

        self.register_buffer("V", torch.tensor(evecs, dtype=torch.float32, device=device))
        self.register_buffer("lambda_graph", torch.tensor(evals, dtype=torch.float32, device=device))
        self.register_buffer("lambda_gamma", torch.tensor(lambda_gamma, dtype=torch.float32, device=device))
        self.register_buffer("sk_raw", torch.tensor(sk_raw, dtype=torch.float32, device=device))
        self.register_buffer("sk", torch.tensor(sk, dtype=torch.float32, device=device))
        self.register_buffer("gk", torch.tensor(gk, dtype=torch.float32, device=device))
        self.s_min_value = float(s_min_value)
        self.power_meta = power_meta

    @staticmethod
    def _symmetrize(adj: np.ndarray, mode: str) -> np.ndarray:
        if mode == "max":
            out = np.maximum(adj, adj.T)
        elif mode == "mean":
            out = 0.5 * (adj + adj.T)
        else:
            raise ValueError(f"Unknown graph_symmetrization={mode}")
        out = np.asarray(out, dtype=np.float64)
        np.fill_diagonal(out, 0.0)
        return out

    @staticmethod
    def _normalized_laplacian(adj: np.ndarray) -> np.ndarray:
        degree = adj.sum(axis=1)
        d_inv_sqrt = np.zeros_like(degree, dtype=np.float64)
        nz = degree > 0
        d_inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
        norm_adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
        lap = np.eye(adj.shape[0], dtype=np.float64) - norm_adj
        return 0.5 * (lap + lap.T)

    def _as_time_node(self, train_data: np.ndarray) -> np.ndarray:
        arr = np.asarray(train_data, dtype=np.float64)
        v = self.num_nodes
        if arr.ndim == 3:
            if arr.shape[1] != v:
                raise ValueError(f"train_data node dimension must be {v}, got {arr.shape}")
            # PEMS08 has one feature.  If more are present, each feature-time
            # observation is treated as another graph signal.
            return np.transpose(arr, (0, 2, 1)).reshape(-1, v)
        if arr.ndim == 2:
            if arr.shape[1] != v:
                raise ValueError(f"train_data node dimension must be {v}, got {arr.shape}")
            return arr
        raise ValueError(f"Unsupported train_data shape {arr.shape}")

    def _estimate_spectral_power(self, x: np.ndarray, evecs: np.ndarray):
        if self.power_mode == "second_moment":
            z = x @ evecs
            sk = np.mean(z ** 2, axis=0)
            return sk, {"estimator": "uncentered second moment", "num_signals": int(x.shape[0])}

        node_mean = np.mean(x, axis=0, keepdims=True)
        xc = x - node_mean
        zc = xc @ evecs
        centered_power = np.mean(zc ** 2, axis=0)

        if self.power_mode == "centered":
            return centered_power, {
                "estimator": "centered modal variance",
                "num_signals": int(x.shape[0]),
                "node_mean_rms": float(np.sqrt(np.mean(node_mean ** 2))),
            }

        H, P = self.conditional_history, self.conditional_horizon
        if x.shape[0] <= H + P:
            raise ValueError("Not enough training samples for conditional residual power")
        sk = np.empty(self.num_nodes, dtype=np.float64)
        n_windows = None
        eye = np.eye(H + 1, dtype=np.float64)
        eye[-1, -1] = 0.0  # do not penalize intercept
        for k in range(self.num_nodes):
            series = zc[:, k]
            windows = np.lib.stride_tricks.sliding_window_view(series, H + P)[:: self.conditional_stride]
            X = windows[:, :H]
            Y = windows[:, H:]
            if n_windows is None:
                n_windows = int(X.shape[0])
            X1 = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
            xtx = X1.T @ X1
            scale = max(float(np.trace(xtx) / max(H, 1)), 1.0)
            ridge = self.conditional_ridge * scale
            W = np.linalg.solve(xtx + ridge * eye, X1.T @ Y)
            residual = Y - X1 @ W
            sk[k] = np.mean(residual ** 2)

        ratio = np.divide(sk, centered_power, out=np.zeros_like(sk), where=centered_power > 0)
        return sk, {
            "estimator": "conditional residual modal variance",
            "history": H,
            "horizon": P,
            "ridge": self.conditional_ridge,
            "stride": self.conditional_stride,
            "num_windows": int(n_windows or 0),
            "median_residual_to_centered_power": float(np.median(ratio)),
            "mean_residual_to_centered_power": float(np.mean(ratio)),
        }

    def to_spec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfvt,vk->bfkt", x, self.V)

    def from_spec(self, x_spec: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfkt,vk->bfvt", x_spec, self.V)

    def r_of_t(self, t: torch.Tensor) -> torch.Tensor:
        u = t.float() / float(self.num_steps - 1)
        log_r = (1.0 - u) * math.log(self.r_max) + u * math.log(self.r_min)
        return torch.exp(log_r)

    def coeffs(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.r_of_t(t).to(self.gk.device)
        denom = r[:, None] + self.gk[None, :]
        a2 = torch.clamp(r[:, None] / denom, min=self.eps)
        a = torch.sqrt(a2)
        nu = torch.clamp(self.sk[None, :] / denom, min=self.eps)
        return a, nu, r

    def coeffs_for_next_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = t.shape[0]
        a = torch.ones((b, self.num_nodes), dtype=torch.float32, device=self.gk.device)
        nu = torch.zeros((b, self.num_nodes), dtype=torch.float32, device=self.gk.device)
        valid = t >= 0
        if torch.any(valid):
            av, nv, _ = self.coeffs(t[valid].long())
            a[valid], nu[valid] = av, nv
        return a, nu

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps_spec: Optional[torch.Tensor] = None):
        x0_spec = self.to_spec(x0)
        if eps_spec is None:
            eps_spec = torch.randn_like(x0_spec)
        a, nu, _ = self.coeffs(t)
        xt_spec = a[:, None, :, None] * x0_spec + torch.sqrt(nu)[:, None, :, None] * eps_spec
        return self.from_spec(xt_spec), eps_spec

    def terminal_sample(self, shape: Tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
        b, f, v, tt = shape
        if v != self.num_nodes:
            raise ValueError(f"Expected V={self.num_nodes}, got {shape}")
        t = torch.full((b,), self.num_steps - 1, dtype=torch.long, device=device)
        _, nu, _ = self.coeffs(t)
        z = torch.randn((b, f, v, tt), device=device)
        return self.from_spec(torch.sqrt(nu)[:, None, :, None] * z)

    def sample_sequence(self, sample_steps: int, grid: str = "logsnr"):
        if sample_steps < 2 or sample_steps > self.num_steps:
            raise ValueError(f"sample_steps must be in [2,{self.num_steps}], got {sample_steps}")
        if grid not in {"logsnr", "uniform"}:
            raise ValueError(f"Unknown MASS sampling grid={grid}")
        idx = np.rint(np.linspace(0, self.num_steps - 1, sample_steps)).astype(int)
        idx[0], idx[-1] = 0, self.num_steps - 1
        if len(np.unique(idx)) != sample_steps:
            raise RuntimeError("MASS sequence rounding produced duplicate indices")
        return idx.tolist()

    def profile_features(self, t: torch.Tensor) -> torch.Tensor:
        """Compact corruption profile for optional denoiser conditioning."""
        a, nu, _ = self.coeffs(t)
        u = t.float() / float(self.num_steps - 1)
        sqrt_nu = torch.sqrt(nu)
        return torch.stack([
            2.0 * u - 1.0,
            a.mean(dim=1),
            a.std(dim=1, unbiased=False),
            sqrt_nu.mean(dim=1),
            sqrt_nu.std(dim=1, unbiased=False),
        ], dim=1)

    def diagnostics(self) -> Dict[str, object]:
        with torch.no_grad():
            t0 = torch.zeros(1, dtype=torch.long, device=self.gk.device)
            tT = torch.full((1,), self.num_steps - 1, dtype=torch.long, device=self.gk.device)
            a0, _, _ = self.coeffs(t0)
            _, nuT, _ = self.coeffs(tT)
            q = (self.sigma ** 2) / self.lambda_gamma
            frac = nuT[0] / q
            gmin, gmax = float(self.gk.min()), float(self.gk.max())
            return {
                "power_mode": self.power_mode,
                "power_meta": self.power_meta,
                "gamma": self.gamma,
                "sigma": self.sigma,
                "r_max": self.r_max,
                "r_min": self.r_min,
                "s_min_value": self.s_min_value,
                "sk_raw_min": float(self.sk_raw.min()),
                "sk_raw_median": float(self.sk_raw.median()),
                "sk_raw_max": float(self.sk_raw.max()),
                "g_min": gmin,
                "g_max": gmax,
                "g_condition": gmax / max(gmin, self.eps),
                "rmax_over_gmax": self.r_max / max(gmax, self.eps),
                "rmin_over_gmin": self.r_min / max(gmin, self.eps),
                "min_initial_a": float(a0.min()),
                "min_initial_a2": float((a0 ** 2).min()),
                "min_terminal_prior_variance_fraction": float(frac.min()),
                "max_terminal_prior_variance_fraction": float(frac.max()),
            }
