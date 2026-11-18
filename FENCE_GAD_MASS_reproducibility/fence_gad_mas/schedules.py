from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch


SCHEDULE_KINDS = ("mass", "fcps")
FCPS_HORIZON_MODES = ("match_mass_power", "delta", "explicit")
FCPS_GRIDS = ("time", "cbar")


@dataclass
class TimeGrid:

    T: float = 1.0
    t_eps: float = 1e-3

    def __post_init__(self) -> None:
        if not (self.T > self.t_eps >= 0):
            raise ValueError("Require T > t_eps >= 0")

    def time_from_u(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(u, 0.0, 1.0)
        return self.t_eps + u * (self.T - self.t_eps)

    def u_from_time(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp((t - self.t_eps) / (self.T - self.t_eps), 0.0, 1.0)

    def time_from_index(self, idx: torch.Tensor, num_steps: int) -> torch.Tensor:
        if num_steps <= 1:
            raise ValueError("num_steps must be > 1")
        u = idx.to(torch.float32) / float(num_steps - 1)
        return self.time_from_u(u)

    def sample_times(
        self,
        nfe: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        grid_power: float = 1.0,
    ) -> torch.Tensor:
        if int(nfe) < 2:
            raise ValueError("nfe must be >= 2")
        if float(grid_power) <= 0:
            raise ValueError("grid_power must be > 0")
        q = torch.linspace(0.0, 1.0, int(nfe), device=device, dtype=dtype)
        if abs(float(grid_power) - 1.0) > 1e-12:
            # q^p puts more points near q=0 (clean endpoint) for p>1.
            q = q.pow(float(grid_power))
        return self.time_from_u(q)

    def condition(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - 2.0 * self.u_from_time(t)


@dataclass
class MASCorridor(TimeGrid):
    r_max: float = 1e4
    r_min: float = 1e-4
    kind = "mass"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.r_max > self.r_min > 0):
            raise ValueError("Require r_max > r_min > 0")

    @property
    def log_slope(self) -> float:
        return (np.log(self.r_min) - np.log(self.r_max)) / (self.T - self.t_eps)

    def r_torch(self, t: torch.Tensor) -> torch.Tensor:
        u = self.u_from_time(t)
        log_r = np.log(self.r_max) + u * (np.log(self.r_min) - np.log(self.r_max))
        return torch.exp(log_r)

    def normalized_logsnr(self, t: torch.Tensor) -> torch.Tensor:
        log_r = torch.log(torch.clamp(self.r_torch(t), min=1e-30))
        lo = float(np.log(self.r_min))
        hi = float(np.log(self.r_max))
        z01 = (log_r - lo) / max(hi - lo, 1e-12)
        return 2.0 * z01 - 1.0

    def condition(self, t: torch.Tensor) -> torch.Tensor:
        return self.normalized_logsnr(t)

    def channel_params(self, lam_gamma, s_k, sigma, t):
        return mas_channel_params(self, lam_gamma, s_k, sigma, t)

    def cbar(self, lam_gamma, s_k, sigma, t):
        return mas_cbar(self, lam_gamma, s_k, sigma, t)

    def rate(self, lam_gamma, s_k, sigma, t):
        lam = lam_gamma.to(device=t.device, dtype=t.dtype)
        s = torch.clamp(s_k.to(device=t.device, dtype=t.dtype), min=1e-10)
        r = self.r_torch(t).to(dtype=t.dtype)
        g = lam * s / (float(sigma) ** 2)
        frac = g[None, :] / torch.clamp(r[:, None] + g[None, :], min=1e-12)
        return -frac * float(self.log_slope) / (2.0 * torch.clamp(lam[None, :], min=1e-12))

    def step_coefficients_time(self, lam_gamma, s_k, sigma, t_cur, t_next, *, device, dtype=torch.float32):
        return mas_step_coefficients_time(
            self, lam_gamma, s_k, sigma, t_cur, t_next, device=device, dtype=dtype
        )

    def describe(self) -> dict:
        return {
            "kind": "mass", "T": float(self.T), "t_eps": float(self.t_eps),
            "r_max": float(self.r_max), "r_min": float(self.r_min),
        }


def mas_channel_params(
    corridor: MASCorridor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lam = lam_gamma.to(device=t.device, dtype=t.dtype)
    s = torch.clamp(s_k.to(device=t.device, dtype=t.dtype), min=1e-10)
    r = corridor.r_torch(t).to(dtype=t.dtype)
    g = lam * s / (float(sigma) ** 2)
    a2 = r[:, None] / torch.clamp(r[:, None] + g[None, :], min=1e-12)
    a = torch.sqrt(torch.clamp(a2, min=1e-12))
    nu = s[None, :] / torch.clamp(r[:, None] + g[None, :], min=1e-12)
    return {"a": a, "nu": torch.clamp(nu, min=1e-12), "r": r}


def mas_cbar(
    corridor: MASCorridor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    t: torch.Tensor,
) -> torch.Tensor:
    lam = lam_gamma.to(device=t.device, dtype=t.dtype)
    s = torch.clamp(s_k.to(device=t.device, dtype=t.dtype), min=1e-10)
    r = corridor.r_torch(t).to(dtype=t.dtype)
    g = lam * s / (float(sigma) ** 2)
    return torch.log1p(g[None, :] / torch.clamp(r[:, None], min=1e-12)) / (
        2.0 * torch.clamp(lam[None, :], min=1e-12)
    )


def mas_step_coefficients_time(
    corridor: MASCorridor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    t_cur: torch.Tensor | float,
    t_next: torch.Tensor | float,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    t_cur_t = torch.as_tensor(t_cur, device=device, dtype=dtype).reshape(1)
    t_next_t = torch.as_tensor(t_next, device=device, dtype=dtype).reshape(1)
    if float(t_cur_t.item()) <= float(t_next_t.item()):
        raise ValueError(f"Reverse transition requires t_cur > t_next, got {t_cur_t.item()} <= {t_next_t.item()}")
    times = torch.cat([t_cur_t, t_next_t], dim=0)
    dt = torch.clamp(t_cur_t - t_next_t, min=1e-12)
    cbar = mas_cbar(corridor, lam_gamma, s_k, sigma, times)
    c_mode = torch.clamp((cbar[0] - cbar[1]) / dt, min=1e-12)
    eta2 = torch.clamp(2.0 * c_mode * (float(sigma) ** 2) * dt, min=1e-12)
    pars_cur = mas_channel_params(corridor, lam_gamma, s_k, sigma, t_cur_t)
    pars_next = mas_channel_params(corridor, lam_gamma, s_k, sigma, t_next_t)
    return {
        "t_cur": t_cur_t,
        "t_next": t_next_t,
        "dt": dt,
        "c_mode": c_mode,
        "eta2": eta2,
        "pars_cur": pars_cur,
        "pars_next": pars_next,
    }


def mas_step_coefficients(
    corridor: MASCorridor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    idx_cur: int,
    idx_next: int,
    num_steps: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    idx = torch.tensor([idx_cur, idx_next], device=device, dtype=torch.float32)
    times = corridor.time_from_index(idx, num_steps).to(device=device, dtype=dtype)
    return mas_step_coefficients_time(
        corridor,
        lam_gamma,
        s_k,
        sigma,
        times[0],
        times[1],
        device=device,
        dtype=dtype,
    )



@dataclass
class FCPSSchedule(TimeGrid):
    cbar_T: float = 7.0
    c_min: float = 0.1
    alpha: float = 4.0
    grid: str = "time"
    kind = "fcps"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (self.alpha >= 0.0):
            raise ValueError("FCPS requires alpha >= 0")
        if not (self.c_min >= 0.0):
            raise ValueError("FCPS requires c_min >= 0")
        if not (self.cbar_T > self.c_min * self.T):
            raise ValueError(
                f"FCPS requires cbar_T > c_min*T for a nondecreasing positive rate; "
                f"got cbar_T={self.cbar_T}, c_min*T={self.c_min * self.T}"
            )
        if self.grid not in FCPS_GRIDS:
            raise ValueError(f"FCPS grid must be one of {FCPS_GRIDS}, got {self.grid!r}")

    # --- scalar clock ---
    def cbar_scalar(self, t: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(t / self.T, 0.0, 1.0)
        return self.c_min * t + (self.cbar_T - self.c_min * self.T) * u.pow(self.alpha + 1.0)

    def rate_scalar(self, t: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(t / self.T, 0.0, 1.0)
        k = (self.cbar_T - self.c_min * self.T) * (self.alpha + 1.0) / self.T
        return self.c_min + k * u.pow(self.alpha)

    def _cbar_float(self, t: float) -> float:
        u = min(max(t / self.T, 0.0), 1.0)
        return self.c_min * t + (self.cbar_T - self.c_min * self.T) * u ** (self.alpha + 1.0)

    def _time_from_cbar(self, targets: np.ndarray) -> np.ndarray:
        """Invert the strictly increasing cbar(t) on [t_eps, T] by bisection."""
        lo = np.full_like(targets, float(self.t_eps), dtype=np.float64)
        hi = np.full_like(targets, float(self.T), dtype=np.float64)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            c_mid = self.c_min * mid + (self.cbar_T - self.c_min * self.T) * (mid / self.T) ** (self.alpha + 1.0)
            below = c_mid < targets
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def sample_times(
        self,
        nfe: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        grid_power: float = 1.0,
    ) -> torch.Tensor:
        if self.grid == "time":
            return super().sample_times(nfe, device=device, dtype=dtype, grid_power=grid_power)
        if int(nfe) < 2:
            raise ValueError("nfe must be >= 2")
        if float(grid_power) <= 0:
            raise ValueError("grid_power must be > 0")
        q = np.linspace(0.0, 1.0, int(nfe), dtype=np.float64)
        if abs(float(grid_power) - 1.0) > 1e-12:
            q = q ** float(grid_power)
        c0 = self._cbar_float(float(self.t_eps))
        c1 = self._cbar_float(float(self.T))
        times = self._time_from_cbar(c0 + q * (c1 - c0))
        times[0] = float(self.t_eps)
        times[-1] = float(self.T)
        out = torch.as_tensor(times, device=device, dtype=dtype)
        if not bool(torch.all(out[1:] > out[:-1])):
            raise ValueError(f"FCPS cbar grid is not strictly increasing at nfe={nfe} in {dtype}")
        return out

    # --- common schedule interface ---
    def channel_params(self, lam_gamma, s_k, sigma, t):
        """Return a, nu (and diagnostics) for batch times. Shapes [B,N]."""
        lam = lam_gamma.to(device=t.device, dtype=t.dtype)
        s = torch.clamp(s_k.to(device=t.device, dtype=t.dtype), min=1e-10)
        cbar = self.cbar_scalar(t)  # [B]
        expo = cbar[:, None] * lam[None, :]
        a = torch.exp(-expo)
        one_minus_a2 = -torch.expm1(-2.0 * expo)
        nu = (float(sigma) ** 2) * one_minus_a2 / torch.clamp(lam[None, :], min=1e-12)
        nu = torch.clamp(nu, min=1e-12)
        snr = a.pow(2) * s[None, :] / nu
        return {"a": a, "nu": nu, "cbar": cbar, "snr": snr}

    def cbar(self, lam_gamma, s_k, sigma, t):
        n = int(lam_gamma.shape[0])
        return self.cbar_scalar(t)[:, None].expand(-1, n)

    def rate(self, lam_gamma, s_k, sigma, t):
        n = int(lam_gamma.shape[0])
        return self.rate_scalar(t)[:, None].expand(-1, n)

    def step_coefficients_time(self, lam_gamma, s_k, sigma, t_cur, t_next, *, device, dtype=torch.float32):
        """Same local clock construction as the MASS sampler, with a scalar cbar."""
        t_cur_t = torch.as_tensor(t_cur, device=device, dtype=dtype).reshape(1)
        t_next_t = torch.as_tensor(t_next, device=device, dtype=dtype).reshape(1)
        if float(t_cur_t.item()) <= float(t_next_t.item()):
            raise ValueError(f"Reverse transition requires t_cur > t_next, got {t_cur_t.item()} <= {t_next_t.item()}")
        times = torch.cat([t_cur_t, t_next_t], dim=0)
        dt = torch.clamp(t_cur_t - t_next_t, min=1e-12)
        cbar = self.cbar(lam_gamma, s_k, sigma, times)
        c_mode = torch.clamp((cbar[0] - cbar[1]) / dt, min=1e-12)
        eta2 = torch.clamp(2.0 * c_mode * (float(sigma) ** 2) * dt, min=1e-12)
        pars_cur = self.channel_params(lam_gamma, s_k, sigma, t_cur_t)
        pars_next = self.channel_params(lam_gamma, s_k, sigma, t_next_t)
        return {
            "t_cur": t_cur_t,
            "t_next": t_next_t,
            "dt": dt,
            "c_mode": c_mode,
            "eta2": eta2,
            "pars_cur": pars_cur,
            "pars_next": pars_next,
        }

    def describe(self) -> dict:
        return {
            "kind": "fcps", "T": float(self.T), "t_eps": float(self.t_eps),
            "cbar_T": float(self.cbar_T), "c_min": float(self.c_min),
            "alpha": float(self.alpha), "grid": str(self.grid),
        }


def _as_float64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def resolve_fcps_cbar_T(
    mode: str,
    lam_gamma,
    s_k,
    sigma: float,
    *,
    r_min: float | None = None,
    delta: float | None = None,
    cbar_T: float | None = None,
) -> tuple[float, dict]:
    mode = str(mode).lower()
    if mode not in FCPS_HORIZON_MODES:
        raise ValueError(f"fcps_horizon must be one of {FCPS_HORIZON_MODES}, got {mode!r}")
    lam = _as_float64(lam_gamma)
    p = np.maximum(_as_float64(s_k), 1e-10)
    lam_min = float(lam.min())
    g = lam * p / float(sigma) ** 2

    def retained_scalar(c: float) -> float:
        return float(np.sum(p * np.exp(-2.0 * lam * c)) / np.sum(p))

    info: dict = {"horizon_mode": mode}
    mass_retained = None
    if r_min is not None:
        mass_a2 = float(r_min) / (float(r_min) + g)
        mass_retained = float(np.sum(p * mass_a2) / np.sum(p))
        info["mass_terminal_power_retained"] = mass_retained
        info["mass_terminal_max_mode_retained"] = float(mass_a2.max())

    if mode == "explicit":
        if cbar_T is None or not (float(cbar_T) > 0):
            raise ValueError("fcps_horizon=explicit requires fcps_cbar_T > 0")
        value = float(cbar_T)
    elif mode == "delta":
        d = 1e-3 if delta is None else float(delta)
        if not (0.0 < d < 1.0):
            raise ValueError("fcps_delta must lie in (0,1)")
        value = math.log(1.0 / d) / (2.0 * lam_min)
        info["delta"] = d
    else:
        if mass_retained is None:
            raise ValueError("fcps_horizon=match_mass_power requires r_min")
        target = min(max(mass_retained, 1e-300), 1.0 - 1e-12)
        lo, hi = 0.0, math.log(1.0 / target) / (2.0 * lam_min)
        # retained_scalar(hi) <= exp(-2 lam_min hi) = target, so [lo, hi] brackets the root.
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if retained_scalar(mid) > target:
                lo = mid
            else:
                hi = mid
        value = 0.5 * (lo + hi)

    info["cbar_T"] = float(value)
    info["fcps_terminal_power_retained"] = retained_scalar(value)
    info["fcps_terminal_max_mode_retained"] = float(math.exp(-2.0 * lam_min * value))
    info["cond_L_gamma"] = float(lam.max() / lam_min)
    return float(value), info


def build_schedule(gad, *, lam_gamma=None, s_k=None) -> tuple[TimeGrid, dict]:
    kind = str(gad.get("schedule", "mass")).strip().lower()
    T = float(gad.get("T_diff", 1.0))
    t_eps = float(gad.get("t_eps", 1e-3))
    r_max = float(gad.get("r_max", 1e4))
    r_min = float(gad.get("r_min", 1e-4))
    if kind == "mass":
        return MASCorridor(T=T, t_eps=t_eps, r_max=r_max, r_min=r_min), {}
    if kind != "fcps":
        raise ValueError(f"Unknown gad.schedule={kind!r}; expected one of {SCHEDULE_KINDS}")
    if lam_gamma is None or s_k is None:
        raise ValueError("FCPS needs lam_gamma and s_k to resolve its horizon")
    explicit = str(gad.get("fcps_cbar_T", "")).strip()
    cbar_T, info = resolve_fcps_cbar_T(
        str(gad.get("fcps_horizon", "match_mass_power")),
        lam_gamma,
        s_k,
        float(gad.get("sigma", 1.0)),
        r_min=r_min,
        delta=float(gad.get("fcps_delta", 1e-3)),
        cbar_T=float(explicit) if explicit else None,
    )
    sched = FCPSSchedule(
        T=T,
        t_eps=t_eps,
        cbar_T=cbar_T,
        c_min=float(gad.get("fcps_c_min", 0.1)),
        alpha=float(gad.get("fcps_alpha", 4.0)),
        grid=str(gad.get("fcps_grid", "time")).strip().lower(),
    )
    return sched, info


def schedule_stiffness_report(
    schedule,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    nfes=(),
    *,
    fine_points: int = 20001,
) -> dict:
    dev = torch.device("cpu")
    lam = lam_gamma.detach().to(dev, torch.float64)
    s = s_k.detach().to(dev, torch.float64)
    t = torch.linspace(float(schedule.t_eps), float(schedule.T), int(fine_points), dtype=torch.float64)
    pars = schedule.channel_params(lam, s, sigma, t)
    a2 = pars["a"].pow(2)
    one_minus = torch.clamp(1.0 - a2, min=1e-300)
    zeta0 = schedule.rate(lam, s, sigma, t) * lam[None, :] * (1.0 + a2) / one_minus
    flat = int(torch.argmax(zeta0).item())
    ti, ki = divmod(flat, zeta0.shape[1])
    out = {
        "schedule": schedule.describe(),
        "L_S0": float(zeta0.max().item()),
        "L_S0_at_t": float(t[ti].item()),
        "L_S0_at_mode": int(ki),
        "per_nfe": {},
    }
    for nfe in nfes:
        grid = schedule.sample_times(int(nfe), device=dev, dtype=torch.float64)
        c_cur = schedule.cbar(lam, s, sigma, grid[1:])
        c_next = schedule.cbar(lam, s, sigma, grid[:-1])
        a2_cur = schedule.channel_params(lam, s, sigma, grid[1:])["a"].pow(2)
        z = (c_cur - c_next) * lam[None, :] * (1.0 + a2_cur) / torch.clamp(1.0 - a2_cur, min=1e-300)
        flat = int(torch.argmax(z).item())
        si, ki = divmod(flat, z.shape[1])
        out["per_nfe"][int(nfe)] = {
            "max_zeta0_dt": float(z.max().item()),
            "at_t_cur": float(grid[1 + si].item()),
            "at_mode": int(ki),
            "unstable_steps": int((z.max(dim=1).values >= 2.0).sum().item()),
            "transitions": int(z.shape[0]),
        }
    return out
