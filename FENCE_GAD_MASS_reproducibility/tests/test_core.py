from __future__ import annotations

import configparser

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fence_gad_mas.metrics import evaluate_sampling_metrics
from fence_gad_mas.model import GADMASFENCEBase
from fence_gad_mas.schedules import MASCorridor, mas_channel_params
from fence_gad_mas.spectral_power import estimate_spectral_power_from_loader


def test_mass_equal_snr_identity():
    corridor = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=1e-4)
    lam = torch.tensor([0.01, 0.2, 1.0, 1.9], dtype=torch.float64)
    power = torch.tensor([4.0, 0.7, 2.0, 0.1], dtype=torch.float64)
    t = torch.tensor([1e-3, 0.2, 0.7, 1.0], dtype=torch.float64)
    pars = mas_channel_params(corridor, lam, power, 1.0, t)
    snr = pars["a"].pow(2) * power[None, :] / pars["nu"]
    assert torch.allclose(snr, pars["r"][:, None].expand_as(snr), rtol=1e-10, atol=1e-10)


def test_fbg_posterior_has_official_plus_offset_sign():
    obj = object.__new__(GADMASFENCEBase)
    torch.nn.Module.__init__(obj)
    obj.temp = 2.0
    obj.offset = 0.1
    obj.logp_delta_clip = 0.0
    obj.minimal_log_posterior = -100.0
    prev = torch.tensor([0.0])
    diff = torch.tensor([3.0])
    got = obj._fbg_posterior_step(prev, diff)
    expected = torch.tensor([-2.9])  # 0 - 0.5*2*3 + 0.1
    assert torch.allclose(got, expected)


class _TinyDataset(Dataset):
    def __init__(self, x):
        self.x = torch.as_tensor(x, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return {"observed_data": self.x[i]}


class _IdentityGraph:
    def to_spectral(self, x): return x


def test_centered_power_removes_constant_mean():
    # Loader shape is [B,T,N]. Node 0 has a large constant mean and small variation.
    x = np.array([
        [[10., 1.], [12., 3.]],
        [[ 8., 5.], [10., 7.]],
    ], dtype=np.float32)
    loader = DataLoader(_TinyDataset(x), batch_size=2, shuffle=False)
    graph = _IdentityGraph()
    second = estimate_spectral_power_from_loader(graph, loader, "cpu", mode="second_moment", min_power=0.0)
    centered = estimate_spectral_power_from_loader(graph, loader, "cpu", mode="centered", min_power=0.0)
    assert np.all(second >= centered - 1e-7)
    assert second[0] > centered[0] * 10.0


class _DummyModel(torch.nn.Module):
    def __init__(self, samples, target, mask):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.samples = samples
        self.target = target
        self.mask = mask
    def evaluate(self, batch, nsample):
        B, _, N, T = self.samples.shape
        return self.samples[:, :nsample], self.target, self.mask, torch.ones_like(self.mask), torch.arange(T)[None, :].repeat(B, 1)


def test_fence_metric_uses_sample_median():
    # Median prediction = 2, mean prediction = 34.33; target=2 => median has zero error.
    samples = torch.tensor([[[[1.0]], [[2.0]], [[100.0]]]])  # [1,3,1,1]
    target = torch.tensor([[[2.0]]])
    mask = torch.tensor([[[1.0]]])
    model = _DummyModel(samples, target, mask)
    loader = [object()]
    med = evaluate_sampling_metrics(model, loader, 1.0, 10.0, nsample=3, aggregation="median")
    mean = evaluate_sampling_metrics(model, loader, 1.0, 10.0, nsample=3, aggregation="mean")
    assert med["MAE"] == 0.0
    assert mean["MAE"] > 30.0


def test_continuous_embedding_is_not_tied_to_nfe():
    from fence_gad_mas.diff_models import ContinuousDiffusionEmbedding
    emb = ContinuousDiffusionEmbedding(embedding_dim=16)
    z = torch.tensor([-1.0, 0.0, 1.0])
    out1 = emb(z)
    out2 = emb(z)
    assert out1.shape == (3, 16)
    assert torch.allclose(out1, out2)


def test_mass_sample_times_nfe_endpoints_and_count():
    corridor = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=1e-4)
    for nfe in (6, 8, 12, 50):
        t = corridor.sample_times(nfe, device="cpu")
        assert len(t) == nfe
        assert torch.isclose(t[0], torch.tensor(corridor.t_eps), atol=1e-7)
        assert torch.isclose(t[-1], torch.tensor(corridor.T), atol=1e-7)
        assert torch.all(t[1:] > t[:-1])


def test_conditional_residual_power_is_finite():
    from fence_gad_mas.spectral_power import estimate_conditional_residual_power_from_loader
    class _ResidualDataset(Dataset):
        def __init__(self):
            self.x = torch.tensor([
                [[1., 2.], [2., 3.], [3., 4.], [4., 5.]],
                [[2., 3.], [3., 4.], [4., 5.], [5., 6.]],
            ])  # [B,T,N]
            self.m = torch.tensor([
                [[1., 1.], [1., 0.], [0., 0.], [1., 1.]],
                [[1., 1.], [0., 1.], [0., 0.], [1., 1.]],
            ])
        def __len__(self): return len(self.x)
        def __getitem__(self, i):
            return {"observed_data": self.x[i], "gt_mask": self.m[i]}
    class _Graph:
        def __init__(self):
            self.A = torch.tensor([[0., 1.], [1., 0.]])
        def to_spectral(self, x): return x
    power = estimate_conditional_residual_power_from_loader(
        _Graph(), DataLoader(_ResidualDataset(), batch_size=2), "cpu", min_power=1e-8
    )
    assert power.shape == (2,)
    assert np.all(np.isfinite(power))
    assert np.all(power > 0)


# ----------------------------------------------------------------------
# FCPS scalar-schedule baseline
# ----------------------------------------------------------------------
from fence_gad_mas.schedules import (
    FCPSSchedule,
    build_schedule,
    resolve_fcps_cbar_T,
    schedule_stiffness_report,
)

_LAM = torch.tensor([0.03, 0.2, 0.9, 1.6, 2.03], dtype=torch.float64)
_POW = torch.tensor([30.0, 5.0, 1.0, 0.3, 0.05], dtype=torch.float64)


def test_fcps_channel_is_scalar_gad_heat_channel():
    sch = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=40.0, c_min=0.1, alpha=4.0)
    t = torch.tensor([1e-3, 0.25, 0.6, 1.0], dtype=torch.float64)
    pars = sch.channel_params(_LAM, _POW, 1.3, t)
    cbar = sch.cbar_scalar(t)
    assert torch.isclose(sch.cbar_scalar(torch.tensor([0.0], dtype=torch.float64)), torch.tensor([0.0], dtype=torch.float64)).all()
    assert torch.isclose(cbar[-1], torch.tensor(40.0, dtype=torch.float64))
    assert torch.allclose(pars["a"], torch.exp(-cbar[:, None] * _LAM[None, :]))
    nu_ref = 1.3 ** 2 / _LAM[None, :] * (1.0 - torch.exp(-2.0 * cbar[:, None] * _LAM[None, :]))
    assert torch.allclose(pars["nu"], torch.clamp(nu_ref, min=1e-12), rtol=1e-10)


def test_fcps_rate_is_derivative_of_cbar():
    sch = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=25.0, c_min=0.1, alpha=4.0)
    t = torch.linspace(0.05, 0.95, 7, dtype=torch.float64)
    h = 1e-6
    fd = (sch.cbar_scalar(t + h) - sch.cbar_scalar(t - h)) / (2 * h)
    assert torch.allclose(sch.rate_scalar(t), fd, rtol=1e-6)
    assert torch.all(sch.rate_scalar(t)[1:] >= sch.rate_scalar(t)[:-1])


def test_fcps_snr_separates_modes_mass_does_not():
    fcps = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=40.0)
    mass = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=1e-4)
    t = torch.tensor([0.5, 0.9], dtype=torch.float64)
    s_f = fcps.channel_params(_LAM, _POW, 1.0, t)["snr"]
    p_m = mass.channel_params(_LAM, _POW, 1.0, t)
    s_m = p_m["a"].pow(2) * _POW[None, :] / p_m["nu"]
    assert float((s_f.max(1).values / s_f.min(1).values).min()) > 1e3
    assert torch.allclose(s_m / s_m[:, :1], torch.ones_like(s_m), rtol=1e-9)


def test_fcps_reverse_clock_is_shared_across_modes():
    sch = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=40.0)
    c = sch.step_coefficients_time(_LAM, _POW, 1.0, 0.7, 0.6, device="cpu", dtype=torch.float64)
    dc = float(sch._cbar_float(0.7) - sch._cbar_float(0.6))
    assert torch.allclose(c["c_mode"], torch.full_like(c["c_mode"], dc / 0.1), rtol=1e-9)
    assert torch.allclose(c["eta2"], 2.0 * c["c_mode"] * c["dt"], rtol=1e-12)


def test_fcps_horizon_matches_mass_terminal_retained_power():
    r_min = 5e-5
    cbar_T, info = resolve_fcps_cbar_T("match_mass_power", _LAM, _POW, 1.0, r_min=r_min)
    g = _LAM * _POW
    mass = float((_POW * r_min / (r_min + g)).sum() / _POW.sum())
    fcps = float((_POW * torch.exp(-2 * _LAM * cbar_T)).sum() / _POW.sum())
    assert abs(fcps - mass) / mass < 1e-8
    assert abs(info["mass_terminal_power_retained"] - mass) < 1e-15


def test_fcps_horizon_delta_is_definition_1():
    cbar_T, info = resolve_fcps_cbar_T("delta", _LAM, _POW, 1.0, delta=1e-3)
    assert np.isclose(cbar_T, np.log(1e3) / (2 * 0.03))
    assert np.isclose(info["fcps_terminal_max_mode_retained"], 1e-3)


def test_fcps_cbar_grid_is_uniform_in_cbar():
    sch = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=60.0, grid="cbar")
    t = sch.sample_times(17, device="cpu", dtype=torch.float64)
    c = sch.cbar_scalar(t)
    assert len(t) == 17 and float(t[0]) == 1e-3 and float(t[-1]) == 1.0
    inc = c[1:] - c[:-1]
    assert torch.allclose(inc, inc.mean().expand_as(inc), rtol=1e-8)
    assert torch.all(sch.sample_times(50, device="cpu")[1:] > sch.sample_times(50, device="cpu")[:-1])


def test_fcps_time_grid_and_conditioning_equal_mass():
    fcps = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=60.0, grid="time")
    mass = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=5e-5)
    for nfe in (6, 16, 25):
        assert torch.equal(fcps.sample_times(nfe, device="cpu"), mass.sample_times(nfe, device="cpu"))
    t = torch.linspace(1e-3, 1.0, 101, dtype=torch.float64)
    assert torch.allclose(fcps.condition(t), mass.condition(t), atol=1e-12)


def test_mass_analytic_rate_matches_cbar_derivative():
    mass = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=5e-5)
    t = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    h = 1e-6
    fd = (mass.cbar(_LAM, _POW, 1.0, t + h) - mass.cbar(_LAM, _POW, 1.0, t - h)) / (2 * h)
    assert torch.allclose(mass.rate(_LAM, _POW, 1.0, t), fd, rtol=1e-5)


def test_stiffness_report_mass_bounded_by_speed_fcps_is_not():
    mass = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=5e-5)
    cbar_T, _ = resolve_fcps_cbar_T("match_mass_power", _LAM, _POW, 1.0, r_min=5e-5)
    fcps = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=cbar_T)
    rm = schedule_stiffness_report(mass, _LAM, _POW, 1.0, nfes=(16,))
    rf = schedule_stiffness_report(fcps, _LAM, _POW, 1.0, nfes=(16,))
    M = abs(mass.log_slope)
    assert rm["L_S0"] <= M * (1 + 1e-9)          # Theorem 2 with B = 0
    assert rf["L_S0"] > 10 * rm["L_S0"]
    assert rm["per_nfe"][16]["unstable_steps"] == 0


def test_build_schedule_defaults_to_mass():
    sch, info = build_schedule({"r_max": "1e3", "r_min": "5e-5"})
    assert isinstance(sch, MASCorridor) and info == {}
    sch, info = build_schedule(
        {"schedule": "fcps", "r_min": "5e-5", "fcps_grid": "cbar"}, lam_gamma=_LAM, s_k=_POW
    )
    assert isinstance(sch, FCPSSchedule) and sch.grid == "cbar" and info["horizon_mode"] == "match_mass_power"


def test_checkpoint_schedule_mismatch_is_rejected():
    import pytest
    import run_fence_gad_mass_sctc as R

    class _M:
        def __init__(self, state): self._s = state
        def schedule_state(self): return self._s

    mass_state = MASCorridor(T=1.0, t_eps=1e-3, r_max=1e3, r_min=5e-5).describe()
    fcps_state = FCPSSchedule(T=1.0, t_eps=1e-3, cbar_T=60.0).describe()
    R.check_checkpoint_schedule(_M(mass_state), {}, "legacy.pt")          # published checkpoints: MASS
    R.check_checkpoint_schedule(_M(fcps_state), {"schedule": dict(fcps_state)}, "ok.pt")
    with pytest.raises(ValueError):
        R.check_checkpoint_schedule(_M(fcps_state), {}, "legacy.pt")
    with pytest.raises(ValueError):
        R.check_checkpoint_schedule(_M(fcps_state), {"schedule": mass_state}, "x.pt")
    with pytest.raises(ValueError):
        R.check_checkpoint_schedule(_M(fcps_state), {"schedule": {**fcps_state, "cbar_T": 61.0}}, "x.pt")
