from __future__ import annotations

import numpy as np
import torch

from schedules import FCPSSchedule, UniformTimeSchedule, build_schedules, channel_params_for_times_torch


def main() -> None:
    lam = np.linspace(0.01, 2.0, 32)
    p = np.linspace(1.0, 0.2, 32)
    sigma = 1.0
    schedules = build_schedules(
        lam_gamma=lam,
        s_k=p,
        sigma=sigma,
        T=1.0,
        fcps_c0=30.0,
        fcps_cmin=0.1,
        fcps_alpha=4.0,
        r_max=1e4,
        r_min=1e-3,
        match_terminal_energy=False,
    )
    t = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    lam_t = torch.tensor(lam, dtype=torch.float64)
    p_t = torch.tensor(p, dtype=torch.float64)

    for name in ["FCPS", "Uniform", "Cosine", "MAS-SNR", "Isotropic-VP-Uniform", "Isotropic-VP-Cosine"]:
        out = channel_params_for_times_torch(name, schedules[name], lam_t, p_t, sigma, t)
        for key in ["a", "nu", "ct", "a2", "log_a2", "rho", "drift_rate", "diffusion_sq", "prior_var"]:
            assert torch.all(torch.isfinite(out[key])), (name, key)
        assert torch.all(out["ct"] >= -1e-12)
        assert torch.all(out["drift_rate"] >= -1e-12)
        assert torch.all(out["diffusion_sq"] >= -1e-12)
        assert torch.all(out["a2"] >= -1e-12)
        assert torch.all(out["a2"] <= 1.0 + 1e-12)
        assert torch.all(out["nu"] >= -1e-12)
        assert torch.all(out["cbar"][1:] >= out["cbar"][:-1] - 1e-10)

    for name in ["FCPS", "Uniform", "Cosine"]:
        out = channel_params_for_times_torch(
            name, schedules[name], lam_t, p_t, sigma, torch.tensor([1.0], dtype=torch.float64)
        )
        np.testing.assert_allclose(out["cbar"].cpu().numpy(), 30.0, rtol=0, atol=1e-9)

    # Critical paper invariant: FCPS alpha=0 is not a distinct baseline; it is Uniform exactly.
    fcps0 = FCPSSchedule(T=1.0, c0=30.0, c_min=0.1, alpha=0.0)
    uniform = UniformTimeSchedule(T=1.0, cbar_T=30.0)
    f0 = channel_params_for_times_torch("FCPS", fcps0, lam_t, p_t, sigma, t)
    uu = channel_params_for_times_torch("Uniform", uniform, lam_t, p_t, sigma, t)
    torch.testing.assert_close(f0["a"], uu["a"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(f0["nu"], uu["nu"], rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(f0["drift_rate"], uu["drift_rate"], rtol=1e-13, atol=1e-13)

    mass = channel_params_for_times_torch("MAS-SNR", schedules["MAS-SNR"], lam_t, p_t, sigma, t[1:])
    snr = mass["a2"] * p_t[None, :] / mass["nu"]
    spread = torch.max(torch.log(snr), dim=1).values - torch.min(torch.log(snr), dim=1).values
    assert float(torch.max(torch.abs(spread))) < 1e-10

    gk = lam_t * p_t / sigma**2
    reconstructed_cbar = 0.5 / lam_t[None, :] * torch.log1p(gk[None, :] / snr)
    torch.testing.assert_close(mass["cbar"], reconstructed_cbar, rtol=1e-10, atol=1e-10)

    # Both graph-agnostic VP baselines must be identical across graph modes,
    # and endpoint matched to each other while following distinct paths.
    iso_u = channel_params_for_times_torch("Isotropic-VP-Uniform", schedules["Isotropic-VP-Uniform"], lam_t, p_t, sigma, t)
    iso_c = channel_params_for_times_torch("Isotropic-VP-Cosine", schedules["Isotropic-VP-Cosine"], lam_t, p_t, sigma, t)
    for iso in (iso_u, iso_c):
        torch.testing.assert_close(iso["a"], iso["a"][:, :1].expand_as(iso["a"]), rtol=0, atol=0)
        torch.testing.assert_close(iso["nu"], iso["nu"][:, :1].expand_as(iso["nu"]), rtol=0, atol=0)
    torch.testing.assert_close(iso_u["a2"][-1], iso_c["a2"][-1], rtol=1e-12, atol=1e-12)
    assert not torch.allclose(iso_u["a2"][64], iso_c["a2"][64])

    # The isotropic controls may carry their own cosine parameters (Table IV
    # ran them with s=0.02, eps=0.01 while GAD Cosine used 0.008, 1e-3). The
    # override must leave GAD Cosine untouched and default to inheritance.
    common = dict(lam_gamma=lam, s_k=p, sigma=sigma, T=1.0, fcps_c0=30.0, fcps_cmin=0.1, fcps_alpha=4.0,
                  r_max=1e4, r_min=1e-3, match_terminal_energy=False, cosine_s=0.008, cosine_terminal_epsilon=1e-3)
    inherited = build_schedules(**common)
    overridden = build_schedules(**common, iso_cosine_s=0.02, iso_cosine_terminal_epsilon=0.01)
    assert inherited["Isotropic-VP-Cosine"].s == 0.008 and inherited["Isotropic-VP-Cosine"].terminal_epsilon == 1e-3
    assert overridden["Isotropic-VP-Cosine"].s == 0.02 and overridden["Isotropic-VP-Cosine"].terminal_epsilon == 0.01
    assert overridden["Isotropic-VP-Uniform"].s == 0.02 and overridden["Isotropic-VP-Uniform"].terminal_epsilon == 0.01
    assert overridden["Cosine"].s == 0.008 and overridden["Cosine"].terminal_epsilon == 1e-3
    a2_inh = channel_params_for_times_torch("Isotropic-VP-Cosine", inherited["Isotropic-VP-Cosine"], lam_t, p_t, sigma, t)["a2"]
    a2_ovr = channel_params_for_times_torch("Isotropic-VP-Cosine", overridden["Isotropic-VP-Cosine"], lam_t, p_t, sigma, t)["a2"]
    assert not torch.allclose(a2_inh[64], a2_ovr[64])
    print("All schedule invariants passed.")


if __name__ == "__main__":
    main()
