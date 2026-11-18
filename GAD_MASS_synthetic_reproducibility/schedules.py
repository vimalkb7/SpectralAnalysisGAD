from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FCPSSchedule:
    T: float = 1.0
    c0: float = 2300.0
    c_min: float = 0.1
    alpha: float = 4.0

    def __post_init__(self) -> None:
        gap = float(self.c0) - float(self.c_min) * float(self.T)
        if gap <= 0:
            raise ValueError("Need c0 > c_min * T for FCPS.")
        if self.alpha < 0:
            raise ValueError("FCPS alpha must be non-negative.")

    @property
    def k(self) -> float:
        gap = float(self.c0) - float(self.c_min) * float(self.T)
        return gap * (float(self.alpha) + 1.0) / float(self.T)


@dataclass(frozen=True)
class UniformTimeSchedule:

    T: float = 1.0
    cbar_T: float = 2300.0

    def __post_init__(self) -> None:
        if self.T <= 0 or self.cbar_T <= 0:
            raise ValueError("Uniform schedule requires T > 0 and cbar_T > 0.")


@dataclass(frozen=True)
class CosineSchedule:
    T: float = 1.0
    cbar_T: float = 2300.0
    s: float = 0.008
    terminal_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.T <= 0 or self.cbar_T <= 0:
            raise ValueError("Cosine schedule requires T > 0 and cbar_T > 0.")
        if self.s < 0:
            raise ValueError("Cosine offset s must be non-negative.")
        if not (0 < self.terminal_epsilon < 1):
            raise ValueError("terminal_epsilon must lie in (0, 1).")


@dataclass(frozen=True)
class IsotropicVPUniformSchedule:
    T: float = 1.0
    s: float = 0.008
    terminal_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("Isotropic VP Uniform requires T > 0.")
        if self.s < 0:
            raise ValueError("Cosine offset s must be non-negative.")
        if not (0 < self.terminal_epsilon < 1):
            raise ValueError("terminal_epsilon must lie in (0, 1).")


@dataclass(frozen=True)
class IsotropicVPCosineSchedule:
    T: float = 1.0
    s: float = 0.008
    terminal_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError("Isotropic VP requires T > 0.")
        if self.s < 0:
            raise ValueError("Cosine offset s must be non-negative.")
        if not (0 < self.terminal_epsilon < 1):
            raise ValueError("terminal_epsilon must lie in (0, 1).")


@dataclass(frozen=True)
class MASCorridor:
    T: float = 1.0
    r_max: float = 1e4
    r_min: float = 1e-3

    def __post_init__(self) -> None:
        if not (self.r_max > self.r_min > 0):
            raise ValueError("Require r_max > r_min > 0.")


def schedule_metadata(schedule_name: str, schedule_obj: object) -> dict[str, object]:
    if is_dataclass(schedule_obj):
        params = asdict(schedule_obj)
    else:
        params = dict(vars(schedule_obj))
    meta = {"name": schedule_name, "class": type(schedule_obj).__name__, "params": params}
    if schedule_name == "Cosine":
        meta["interpretation"] = "endpoint-matched cosine time warp of scalar GAD"
    elif schedule_name == "Isotropic-VP-Uniform":
        meta["interpretation"] = "standard graph-agnostic isotropic VP channel with constant beta (Uniform)"
    elif schedule_name in {"Isotropic-VP-Cosine", "Isotropic-VP"}:
        meta["interpretation"] = "standard graph-agnostic isotropic VP channel with cosine alpha-bar"
        if schedule_name == "Isotropic-VP":
            meta["deprecated_alias_for"] = "Isotropic-VP-Cosine"
    elif schedule_name == "FCPS":
        meta["alpha_zero_is_uniform"] = True
    return meta


def _fcps_cbar_ct_torch(schedule: FCPSSchedule, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = float(schedule.T)
    u = torch.clamp(t / T, 0.0, 1.0)
    cbar = float(schedule.c_min) * t + (
        float(schedule.c0) - float(schedule.c_min) * T
    ) * (u ** (float(schedule.alpha) + 1.0))
    ct = float(schedule.c_min) + float(schedule.k) * (u ** float(schedule.alpha))
    return cbar, ct


def _uniform_cbar_ct_torch(schedule: UniformTimeSchedule, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    T = float(schedule.T)
    u = torch.clamp(t / T, 0.0, 1.0)
    cbar = float(schedule.cbar_T) * u
    ct = torch.full_like(t, float(schedule.cbar_T) / T)
    return cbar, ct


def _cosine_geometry(schedule_s: float, terminal_epsilon: float, T: float, t: torch.Tensor):
    s = float(schedule_s)
    eps_end = float(terminal_epsilon)
    u = torch.clamp(t / T, 0.0, 1.0)
    u_eff = (1.0 - eps_end) * u
    theta0 = (s / (1.0 + s)) * (torch.pi / 2.0)
    theta = ((u_eff + s) / (1.0 + s)) * (torch.pi / 2.0)
    theta_end = (((1.0 - eps_end) + s) / (1.0 + s)) * (torch.pi / 2.0)

    tiny = torch.finfo(t.dtype).tiny
    cos0 = torch.clamp(torch.cos(torch.as_tensor(theta0, dtype=t.dtype, device=t.device)), min=tiny)
    cos_t = torch.clamp(torch.cos(theta), min=tiny)
    cos_end = torch.clamp(torch.cos(torch.as_tensor(theta_end, dtype=t.dtype, device=t.device)), min=tiny)
    log_cos0 = torch.log(cos0)
    log_cos = torch.log(cos_t)
    log_cos_end = torch.log(cos_end)

    q = -2.0 * (log_cos - log_cos0)
    q_end = -2.0 * (log_cos_end - log_cos0)
    w = q / q_end
    dq_du = (1.0 - eps_end) * torch.pi / (1.0 + s) * torch.tan(theta)
    dw_dt = dq_du / (q_end * T)
    return w, dw_dt, q, dq_du / T


def _cosine_cbar_ct_torch(schedule: CosineSchedule, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, dw_dt, _, _ = _cosine_geometry(schedule.s, schedule.terminal_epsilon, float(schedule.T), t)
    cbar = float(schedule.cbar_T) * w
    ct = float(schedule.cbar_T) * dw_dt
    return cbar, torch.clamp(ct, min=0.0)


def _isotropic_vp_cosine_params_torch(schedule: IsotropicVPCosineSchedule, t: torch.Tensor):
    _, _, q, dq_dt = _cosine_geometry(schedule.s, schedule.terminal_epsilon, float(schedule.T), t)
    alpha_bar = torch.exp(-q)
    beta = torch.clamp(dq_dt, min=0.0)
    return alpha_bar, beta, q


def _isotropic_vp_uniform_params_torch(schedule: IsotropicVPUniformSchedule, t: torch.Tensor):
    t_end = torch.full((1,), float(schedule.T), dtype=t.dtype, device=t.device)
    _, _, q_end_arr, _ = _cosine_geometry(
        schedule.s, schedule.terminal_epsilon, float(schedule.T), t_end
    )
    q_end = q_end_arr[0]
    u = torch.clamp(t / float(schedule.T), 0.0, 1.0)
    q = q_end * u
    beta_scalar = q_end / float(schedule.T)
    beta = torch.ones_like(t) * beta_scalar
    alpha_bar = torch.exp(-q)
    return alpha_bar, beta, q


def scalar_clock_for_times_torch(schedule_name: str, schedule_obj: object, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if schedule_name == "FCPS":
        return _fcps_cbar_ct_torch(schedule_obj, t)
    if schedule_name == "Uniform":
        return _uniform_cbar_ct_torch(schedule_obj, t)
    if schedule_name == "Cosine":
        return _cosine_cbar_ct_torch(schedule_obj, t)
    raise ValueError(f"{schedule_name!r} is not a scalar GAD schedule.")


def scalar_a2_terminal(lam_gamma: np.ndarray, schedule: FCPSSchedule, T: float) -> np.ndarray:
    t = torch.as_tensor([float(T)], dtype=torch.float64)
    cbar, _ = _fcps_cbar_ct_torch(schedule, t)
    return np.exp(-2.0 * np.asarray(lam_gamma, dtype=np.float64) * float(cbar.item()))


def retained_signal_energy(a2: np.ndarray, s_k: np.ndarray) -> float:
    return float(np.mean(np.asarray(a2, dtype=np.float64) * np.asarray(s_k, dtype=np.float64)))


def solve_rmin_for_target_terminal_energy(
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    target_energy: float,
    rmin_lo: float = 1e-12,
    rmin_hi: float = 1e6,
    max_iter: int = 120,
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


def scalar_cbar_from_terminal_a2(gamma: float, terminal_a2: float = 1e-3) -> float:
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if not (0 < terminal_a2 < 1):
        raise ValueError("terminal_a2 must lie in (0,1).")
    return float(-np.log(terminal_a2) / (2.0 * gamma))


def mass_endpoints_from_tolerances(
    lam_gamma: np.ndarray,
    s_k: np.ndarray,
    sigma: float,
    start_a2_error: float = 1e-3,
    terminal_a2: float = 1e-3,
    pmin: float = 1e-12,
) -> tuple[float, float]:
    if not (0 < start_a2_error < 1 and 0 < terminal_a2 < 1):
        raise ValueError("Endpoint tolerances must lie in (0,1).")
    p = np.maximum(np.asarray(s_k, dtype=np.float64), pmin)
    g = np.asarray(lam_gamma, dtype=np.float64) * p / (float(sigma) ** 2)
    g_min = float(np.min(g))
    g_max = float(np.max(g))
    r_max = ((1.0 - start_a2_error) / start_a2_error) * g_max
    r_min = (terminal_a2 / (1.0 - terminal_a2)) * g_min
    if not (r_max > r_min > 0):
        raise RuntimeError("Calibrated MASS corridor is invalid; inspect p_k/gamma/sigma.")
    return float(r_max), float(r_min)


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
    uniform_cbar_T: float | None = None,
    cosine_cbar_T: float | None = None,
    cosine_s: float = 0.008,
    cosine_terminal_epsilon: float = 1e-3,
    include_isotropic_vp: bool = True,
    iso_cosine_s: float | None = None,
    iso_cosine_terminal_epsilon: float | None = None,
) -> dict[str, object]:
    fcps = FCPSSchedule(T=T, c0=fcps_c0, c_min=fcps_cmin, alpha=fcps_alpha)
    scalar_terminal = float(fcps_c0)
    uniform = UniformTimeSchedule(T=T, cbar_T=scalar_terminal if uniform_cbar_T is None else float(uniform_cbar_T))
    cosine = CosineSchedule(
        T=T,
        cbar_T=scalar_terminal if cosine_cbar_T is None else float(cosine_cbar_T),
        s=cosine_s,
        terminal_epsilon=cosine_terminal_epsilon,
    )
    if r_min is None:
        a2_fcps_T = scalar_a2_terminal(lam_gamma, fcps, T)
        target_energy = retained_signal_energy(a2_fcps_T, s_k)
        r_min = solve_rmin_for_target_terminal_energy(lam_gamma, s_k, sigma, target_energy) if match_terminal_energy else 1e-3
    mas = MASCorridor(T=T, r_max=r_max, r_min=r_min)
    out: dict[str, object] = {
        "FCPS": fcps,
        "Uniform": uniform,
        "Cosine": cosine,
        "MAS-SNR": mas,
        "r_min": float(r_min),
    }
    if include_isotropic_vp:
        iso_s = float(cosine_s if iso_cosine_s is None else iso_cosine_s)
        iso_eps = float(cosine_terminal_epsilon if iso_cosine_terminal_epsilon is None else iso_cosine_terminal_epsilon)
        iso_uniform = IsotropicVPUniformSchedule(T=T, s=iso_s, terminal_epsilon=iso_eps)
        iso_cosine = IsotropicVPCosineSchedule(T=T, s=iso_s, terminal_epsilon=iso_eps)
        out["Isotropic-VP-Uniform"] = iso_uniform
        out["Isotropic-VP-Cosine"] = iso_cosine
        # Backwards-compatible alias for old manifests/checkpoints.
        out["Isotropic-VP"] = iso_cosine
    return out


def channel_params_for_times_torch(
    schedule_name: str,
    schedule_obj: object,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    t = t.to(dtype=lam_gamma.dtype)
    T = float(schedule_obj.T)
    n = lam_gamma.numel()

    if schedule_name in {"FCPS", "Uniform", "Cosine"}:
        cbar, ct_scalar = scalar_clock_for_times_torch(schedule_name, schedule_obj, t)
        log_a2 = -2.0 * cbar[:, None] * lam_gamma[None, :]
        a2 = torch.exp(log_a2)
        a = torch.exp(0.5 * log_a2)
        nu = (sigma ** 2) * (1.0 - a2) / lam_gamma[None, :]
        ct = ct_scalar[:, None].expand(-1, n)
        cbar_full = cbar[:, None].expand(-1, n)
        drift_rate = ct * lam_gamma[None, :]
        diffusion_sq = 2.0 * ct * (sigma ** 2)
        prior_var = ((sigma ** 2) / lam_gamma)[None, :].expand(t.shape[0], -1)
    elif schedule_name == "MAS-SNR":
        log_r_max = float(np.log(schedule_obj.r_max))
        log_r_min = float(np.log(schedule_obj.r_min))
        x = torch.clamp(t / T, 0.0, 1.0)
        u_t = 0.5 * (1.0 - torch.cos(torch.pi * x))
        du_dt = 0.5 * (torch.pi / T) * torch.sin(torch.pi * x)
        log_r = log_r_max + (log_r_min - log_r_max) * u_t
        d_log_r_dt = (log_r_min - log_r_max) * du_dt
        r = torch.exp(log_r)[:, None]
        s_k_safe = torch.clamp(s_k, min=1e-10)
        gk = lam_gamma * s_k_safe / (sigma ** 2)
        # Stable analytic log attenuation is exported for diagnostics even
        # when a2 itself underflows at very late times.
        log_a2 = log_r[:, None] - torch.logaddexp(log_r[:, None], torch.log(gk)[None, :])
        a2 = torch.exp(log_a2)
        a = torch.exp(0.5 * log_a2)
        nu = s_k_safe[None, :] / (r + gk[None, :])
        cbar_full = 0.5 / lam_gamma[None, :] * torch.log1p(gk[None, :] / r)
        ct = -(0.5 / lam_gamma[None, :]) * (1.0 - a2) * d_log_r_dt[:, None]
        ct = torch.clamp(ct, min=0.0)
        drift_rate = ct * lam_gamma[None, :]
        diffusion_sq = 2.0 * ct * (sigma ** 2)
        prior_var = ((sigma ** 2) / lam_gamma)[None, :].expand(t.shape[0], -1)
    elif schedule_name in {"Isotropic-VP-Uniform", "Isotropic-VP-Cosine", "Isotropic-VP"}:
        if schedule_name == "Isotropic-VP-Uniform":
            alpha_bar, beta, q = _isotropic_vp_uniform_params_torch(schedule_obj, t)
        else:
            alpha_bar, beta, q = _isotropic_vp_cosine_params_torch(schedule_obj, t)
        log_a2 = (-q)[:, None].expand(-1, n)
        a_scalar = torch.exp(-0.5 * q)
        a = a_scalar[:, None].expand(-1, n)
        a2 = alpha_bar[:, None].expand(-1, n)
        nu_scalar = torch.clamp(1.0 - alpha_bar, min=0.0)
        nu = nu_scalar[:, None].expand(-1, n)
        # ct/cbar are exported for API compatibility; for isotropic VP they mean
        # beta/2 and q/2 so that drift_rate=ct with unit effective eigenvalue.
        ct = (0.5 * beta)[:, None].expand(-1, n)
        cbar_full = (0.5 * q)[:, None].expand(-1, n)
        drift_rate = ct
        diffusion_sq = beta[:, None].expand(-1, n)
        prior_var = torch.ones_like(nu)
    else:
        raise ValueError(
            f"Unknown schedule_name: {schedule_name}. Expected FCPS, Uniform, Cosine, MAS-SNR, "
            "Isotropic-VP-Uniform, Isotropic-VP-Cosine, or the legacy Isotropic-VP alias."
        )

    rho = a2 * s_k[None, :] + nu
    return {
        "a": a,
        "a2": a2,
        "log_a2": log_a2,
        "nu": nu,
        "rho": rho,
        "ct": ct,
        "cbar": cbar_full,
        "drift_rate": drift_rate,
        "diffusion_sq": diffusion_sq,
        "prior_var": prior_var,
    }
