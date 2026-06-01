from __future__ import annotations

import torch

from schedules import channel_params_for_times_torch



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
    return {
        "post_var": post_var,
        "coef_x0": coef_x0,
        "coef_xt": coef_xt,
    }



def _initial_variance(init_mode: str, pars_T: dict[str, torch.Tensor], lam_gamma: torch.Tensor, sigma: float) -> torch.Tensor:
    mode = init_mode.lower()
    if mode in {"rho", "corrupted_marginal"}:
        return torch.clamp(pars_T["rho"][0], min=1e-12)
    if mode in {"algorithm2", "alg2", "nu"}:
        return torch.clamp(pars_T["nu"][0], min=1e-12)
    if mode in {"stationary", "theory"}:
        return torch.clamp((sigma ** 2) / lam_gamma, min=1e-12)
    raise ValueError(f"Unknown init_mode: {init_mode}")


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
    device = V.device
    dtype = V.dtype
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))

    tT = torch.full((1,), T, device=device, dtype=dtype)
    pars_T = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, tT)
    init_var = _initial_variance(init_mode, pars_T, lam_gamma, sigma)
    xt_tilde = torch.randn((num_samples, lam_gamma.numel()), generator=g, device=device, dtype=dtype) * torch.sqrt(init_var)[None, :]
    xt = xt_tilde @ V.T

    t_grid = torch.linspace(T, t0, steps + 1, device=device, dtype=dtype)

    for i in range(steps):
        t = t_grid[i]
        t_prev = t_grid[i + 1]
        dt = t - t_prev
        t_arr = torch.full((num_samples,), t, device=device, dtype=dtype)
        t_prev_arr = torch.full((num_samples,), t_prev, device=device, dtype=dtype)

        pars = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, t_arr)
        pars_prev = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, t_prev_arr)
        ci = torch.clamp((pars["cbar"] - pars_prev["cbar"]) / max(float(dt.item()) if hasattr(dt, 'item') else float(dt), 1e-12), min=0.0)
        xt_tilde = xt @ V

        if method in {"em", "heun"}:
            score_tilde = model.score_from_prediction(xt=xt, V=V, t=t_arr, a=pars["a"], nu=pars["nu"])
            drift_tilde = ci * lam_gamma.unsqueeze(0) * xt_tilde + 2.0 * ci * (sigma ** 2) * score_tilde
            noise = torch.sqrt(torch.clamp(2.0 * ci * (sigma ** 2) * dt, min=0.0)) * torch.randn(xt_tilde.shape, generator=g, device=device, dtype=dtype)

            if method == "em":
                xnext_tilde = xt_tilde + drift_tilde * dt + noise
            else:
                xpred_tilde = xt_tilde + drift_tilde * dt + noise
                xpred = xpred_tilde @ V.T
                score_pred = model.score_from_prediction(xt=xpred, V=V, t=t_prev_arr, a=pars_prev["a"], nu=pars_prev["nu"])
                drift_pred = pars_prev["ct"] * lam_gamma.unsqueeze(0) * xpred_tilde + 2.0 * pars_prev["ct"] * (sigma ** 2) * score_pred
                xnext_tilde = xt_tilde + 0.5 * (drift_tilde + drift_pred) * dt + noise

        elif method in {"ddpm", "ddim"}:
            xt_tilde, x0_hat_tilde, eps_hat = _predict_x0_and_eps_tilde(model, xt, V, t_arr, pars["a"], pars["nu"])
            post = _pairwise_channel_posterior(pars["a"], pars["nu"], pars_prev["a"], pars_prev["nu"])
            if method == "ddpm":
                post_mean = post["coef_x0"] * x0_hat_tilde + post["coef_xt"] * xt_tilde
                post_std = torch.sqrt(torch.clamp(post["post_var"], min=0.0))
                noise = torch.randn(xt_tilde.shape, generator=g, device=device, dtype=dtype)
                xnext_tilde = post_mean + post_std * noise
            else:
                sigma_ddim = torch.sqrt(torch.clamp(post["post_var"], min=0.0)) * float(ddim_eta)
                c_ddim = torch.sqrt(torch.clamp(pars_prev["nu"] - sigma_ddim ** 2, min=0.0))
                noise = torch.randn(xt_tilde.shape, generator=g, device=device, dtype=dtype)
                xnext_tilde = pars_prev["a"] * x0_hat_tilde + c_ddim * eps_hat + sigma_ddim * noise
        else:
            raise ValueError(f"Unknown method: {method}")

        xt = xnext_tilde @ V.T

    return xt
