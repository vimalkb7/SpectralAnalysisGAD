from __future__ import annotations

import torch

from schedules import channel_params_for_times_torch


def method_nfe(method: str, steps: int) -> int:
    m = method.lower()
    if m in {"em", "ddpm", "ddim", "dpmpp2m", "dpm-solver", "dpm_solver", "dpmsolver"}:
        return int(steps)
    if m == "heun":
        return int(2 * steps)
    raise ValueError(f"Unknown method: {method}")


def _predict_x0_and_eps_tilde(model, xt, V, t, a, nu):
    xt_tilde = xt @ V
    x0_hat = model(xt, V, t)
    x0_hat_tilde = x0_hat @ V
    sqrt_nu = torch.sqrt(torch.clamp(nu, min=1e-12))
    eps_hat = (xt_tilde - a * x0_hat_tilde) / sqrt_nu
    return xt_tilde, x0_hat_tilde, eps_hat


def _pairwise_channel_posterior(a_t, nu_t, a_prev, nu_prev):
    eps = 1e-12
    alpha = a_t / (a_prev + eps)
    nu_delta = torch.clamp(nu_t - (alpha ** 2) * nu_prev, min=0.0)
    post_var = torch.clamp(nu_prev * nu_delta / (nu_t + eps), min=0.0)
    coef_x0 = a_prev * nu_delta / (nu_t + eps)
    coef_xt = alpha * nu_prev / (nu_t + eps)
    return {"post_var": post_var, "coef_x0": coef_x0, "coef_xt": coef_xt}


def _initial_variance(init_mode: str, pars_T: dict[str, torch.Tensor]) -> torch.Tensor:
    mode = init_mode.lower()
    if mode in {"rho", "corrupted_marginal"}:
        return torch.clamp(pars_T["rho"][0], min=1e-12)
    if mode in {"algorithm2", "alg2", "nu"}:
        return torch.clamp(pars_T["nu"][0], min=1e-12)
    if mode in {"stationary", "theory"}:
        return torch.clamp(pars_T["prior_var"][0], min=1e-12)
    raise ValueError(f"Unknown init_mode: {init_mode}")


def _signed_safe_denominator(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    e = torch.as_tensor(eps, dtype=x.dtype, device=x.device)
    return torch.where(torch.abs(x) < e, torch.where(x < 0, -e, e), x)


def _normalized_vp_coefficients(pars: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_a2 = pars["log_a2"]
    alpha = torch.exp(0.5 * log_a2)
    noise_fraction = torch.clamp(-torch.expm1(log_a2), min=torch.finfo(log_a2.dtype).tiny)
    sigma_bar = torch.sqrt(noise_fraction)
    lambda_half_log_snr = 0.5 * log_a2 - 0.5 * torch.log(noise_fraction)
    return alpha, sigma_bar, lambda_half_log_snr


@torch.no_grad()
def reverse_sample(
    model,
    V: torch.Tensor,
    lam_gamma: torch.Tensor,
    sigma: float,
    schedule_name: str,
    schedule_obj,
    s_k: torch.Tensor,
    num_samples: int,
    steps: int,
    t0: float = 1e-3,
    T: float = 1.0,
    seed: int = 0,
    method: str = "em",
    ddim_eta: float = 0.0,
    init_mode: str = "algorithm2",
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = V.device
    dtype = V.dtype
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    tT = torch.full((1,), T, device=device, dtype=dtype)
    pars_T = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, tT)
    init_var = _initial_variance(init_mode, pars_T)
    xt_tilde = torch.randn((num_samples, lam_gamma.numel()), generator=g, device=device, dtype=dtype) * torch.sqrt(init_var)[None, :]
    xt = xt_tilde @ V.T

    t_grid = torch.linspace(T, t0, steps + 1, device=device, dtype=dtype)
    method_key = method.lower()
    if method_key in {"dpm-solver", "dpm_solver", "dpmsolver"}:
        method_key = "dpmpp2m"
    dpm_prev_model: torch.Tensor | None = None
    dpm_prev_lambda: torch.Tensor | None = None

    for i in range(steps):
        t = t_grid[i]
        t_prev = t_grid[i + 1]
        dt = t - t_prev
        t_arr = torch.full((num_samples,), t, device=device, dtype=dtype)
        t_prev_arr = torch.full((num_samples,), t_prev, device=device, dtype=dtype)

        pars = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, t_arr)
        pars_prev = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, t_prev_arr)
        xt_tilde = xt @ V

        if method_key in {"em", "heun"}:
            score_tilde = model.score_from_prediction(xt=xt, V=V, t=t_arr, a=pars["a"], nu=pars["nu"])
            drift_tilde = pars["drift_rate"] * xt_tilde + pars["diffusion_sq"] * score_tilde
            noise = torch.sqrt(torch.clamp(pars["diffusion_sq"] * dt, min=0.0)) * torch.randn(
                xt_tilde.shape, generator=g, device=device, dtype=dtype
            )

            if method_key == "em":
                xnext_tilde = xt_tilde + drift_tilde * dt + noise
            else:
                xpred_tilde = xt_tilde + drift_tilde * dt + noise
                xpred = xpred_tilde @ V.T
                score_pred = model.score_from_prediction(
                    xt=xpred, V=V, t=t_prev_arr, a=pars_prev["a"], nu=pars_prev["nu"]
                )
                drift_pred = pars_prev["drift_rate"] * xpred_tilde + pars_prev["diffusion_sq"] * score_pred
                xnext_tilde = xt_tilde + 0.5 * (drift_tilde + drift_pred) * dt + noise

        elif method_key in {"ddpm", "ddim"}:
            xt_tilde, x0_hat_tilde, eps_hat = _predict_x0_and_eps_tilde(
                model, xt, V, t_arr, pars["a"], pars["nu"]
            )
            post = _pairwise_channel_posterior(pars["a"], pars["nu"], pars_prev["a"], pars_prev["nu"])
            if method_key == "ddpm":
                post_mean = post["coef_x0"] * x0_hat_tilde + post["coef_xt"] * xt_tilde
                post_std = torch.sqrt(torch.clamp(post["post_var"], min=0.0))
                noise = torch.randn(xt_tilde.shape, generator=g, device=device, dtype=dtype)
                xnext_tilde = post_mean + post_std * noise
            else:
                sigma_ddim = torch.sqrt(torch.clamp(post["post_var"], min=0.0)) * float(ddim_eta)
                c_ddim = torch.sqrt(torch.clamp(pars_prev["nu"] - sigma_ddim ** 2, min=0.0))
                noise = torch.randn(xt_tilde.shape, generator=g, device=device, dtype=dtype)
                xnext_tilde = pars_prev["a"] * x0_hat_tilde + c_ddim * eps_hat + sigma_ddim * noise

        elif method_key == "dpmpp2m":
            xt_tilde = xt @ V
            x0_hat = model(xt, V, t_arr)
            x0_hat_tilde = x0_hat @ V
            q_sqrt = torch.sqrt(torch.clamp(pars["prior_var"], min=1e-20))
            q_prev_sqrt = torch.sqrt(torch.clamp(pars_prev["prior_var"], min=1e-20))
            y = xt_tilde / q_sqrt
            model_s = x0_hat_tilde / q_sqrt

            alpha_s, sigma_s, lambda_s = _normalized_vp_coefficients(pars)
            alpha_tgt, sigma_tgt, lambda_tgt = _normalized_vp_coefficients(pars_prev)
            h = lambda_tgt - lambda_s
            phi_1 = torch.expm1(-h)

            if dpm_prev_model is None or dpm_prev_lambda is None:
                y_next = (sigma_tgt / torch.clamp(sigma_s, min=1e-20)) * y - alpha_tgt * phi_1 * model_s
            else:
                h0 = lambda_s - dpm_prev_lambda
                r0 = h0 / _signed_safe_denominator(h)
                D1_0 = (model_s - dpm_prev_model) / _signed_safe_denominator(r0)
                y_next = (
                    (sigma_tgt / torch.clamp(sigma_s, min=1e-20)) * y
                    - alpha_tgt * phi_1 * model_s
                    - 0.5 * alpha_tgt * phi_1 * D1_0
                )

            dpm_prev_model = model_s
            dpm_prev_lambda = lambda_s
            xnext_tilde = y_next * q_prev_sqrt
        else:
            raise ValueError(f"Unknown method: {method}")

        if not torch.isfinite(xnext_tilde).all():
            raise FloatingPointError(
                f"Non-finite reverse state: schedule={schedule_name}, method={method}, step={i+1}/{steps}, t={float(t):.6g}."
            )
        xt = xnext_tilde @ V.T

    return xt
