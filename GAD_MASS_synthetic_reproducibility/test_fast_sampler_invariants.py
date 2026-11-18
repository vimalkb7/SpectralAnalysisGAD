"""Numerical checks for DDIM and the spectral DPM-Solver++ 2M implementation."""
from __future__ import annotations

import numpy as np
import torch

from samplers import method_nfe, reverse_sample
from schedules import build_schedules


class ZeroX0(torch.nn.Module):
    def forward(self, xt, V, t):
        return torch.zeros_like(xt)


def main() -> None:
    n = 12
    V = torch.eye(n, dtype=torch.float64)
    lam = torch.linspace(0.1, 2.0, n, dtype=torch.float64)
    p = torch.linspace(1.0, 0.2, n, dtype=torch.float64)
    schedules = build_schedules(
        lam_gamma=lam.numpy(), s_k=p.numpy(), sigma=1.0, T=1.0,
        fcps_c0=3.0, fcps_cmin=0.1, fcps_alpha=4.0,
        r_max=100.0, r_min=1e-2, match_terminal_energy=False,
    )
    model = ZeroX0()
    names = [
        "Uniform", "FCPS", "Cosine", "MAS-SNR",
        "Isotropic-VP-Uniform", "Isotropic-VP-Cosine",
    ]
    for name in names:
        x = reverse_sample(
            model=model, V=V, lam_gamma=lam, sigma=1.0,
            schedule_name=name, schedule_obj=schedules[name], s_k=p,
            num_samples=32, steps=10, t0=1e-2, T=1.0, seed=7,
            method="dpmpp2m", init_mode="algorithm2",
        )
        assert x.shape == (32, n)
        assert torch.isfinite(x).all(), name


    assert method_nfe("dpmpp2m", 25) == 25
    assert method_nfe("dpm-solver", 25) == 25

    class HalfPredictor(torch.nn.Module):
        def forward(self, xt, V, t):
            return 0.5 * xt

    model2 = HalfPredictor()
    x_ddim = reverse_sample(
        model2, V, lam, 1.0, "Uniform", schedules["Uniform"], p,
        16, 1, t0=0.2, T=1.0, seed=19, method="ddim", ddim_eta=0.0,
    )
    x_dpm = reverse_sample(
        model2, V, lam, 1.0, "Uniform", schedules["Uniform"], p,
        16, 1, t0=0.2, T=1.0, seed=19, method="dpmpp2m",
    )
    torch.testing.assert_close(x_ddim, x_dpm, rtol=2e-10, atol=2e-10)
    print("All fast-sampler invariants passed.")


if __name__ == "__main__":
    main()
