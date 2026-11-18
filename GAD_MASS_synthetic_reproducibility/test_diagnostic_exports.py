from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np

from diagnostics import (
    _write_extreme_mode_mean_decay_csv,
    _write_modewise_log10_snr_csv,
    compute_forward_plot_arrays,
    export_generation_plot_data,
)
from schedules import build_schedules


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    n = 16
    num_times = 33
    k_extreme = 3
    lam = np.linspace(0.01, 2.0, n)
    power = np.linspace(1.0, 0.2, n)
    sigma = 1.0
    schedules = build_schedules(
        lam_gamma=lam,
        s_k=power,
        sigma=sigma,
        T=1.0,
        fcps_c0=30.0,
        fcps_cmin=0.1,
        fcps_alpha=4.0,
        r_max=1e4,
        r_min=1e-3,
        match_terminal_energy=False,
    )

    arrays, fcps_metrics = compute_forward_plot_arrays(
        lam_gamma=lam,
        s_k=power,
        sigma=sigma,
        schedule_name="FCPS",
        schedule_obj=schedules["FCPS"],
        num_times=num_times,
        num_extreme_modes=k_extreme,
    )
    _, uniform_metrics = compute_forward_plot_arrays(
        lam_gamma=lam,
        s_k=power,
        sigma=sigma,
        schedule_name="Uniform",
        schedule_obj=schedules["Uniform"],
        num_times=num_times,
        num_extreme_modes=k_extreme,
    )
    # Regression for the old floor-clipping artefact: alpha>0 FCPS and Uniform
    # follow different paths, so their temporal mean log-SNR ranges must differ.
    assert abs(fcps_metrics["mean_log_snr_range"] - uniform_metrics["mean_log_snr_range"]) > 1e-6
    np.testing.assert_array_equal(arrays["low_mode_indices"], np.arange(k_extreme))
    np.testing.assert_array_equal(arrays["high_mode_indices"], np.arange(n - k_extreme, n))
    assert arrays["forward_curve_times"][0] == 0.0
    np.testing.assert_allclose(arrays["forward_curve_a"][0], np.ones(n), atol=1e-12, rtol=0)
    assert arrays["log10_snr"].shape == (num_times, n)
    assert arrays["log10_snr_plot_clipped"].shape == (num_times, n)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        decay_csv = root / "extreme_mode_mean_decay_curves.csv"
        snr_csv = root / "modewise_log10_snr.csv"
        _write_extreme_mode_mean_decay_csv(decay_csv, arrays)
        _write_modewise_log10_snr_csv(snr_csv, arrays)
        assert count_csv_rows(decay_csv) == 2 * k_extreme * num_times
        assert count_csv_rows(snr_csv) == n * num_times

        rng = np.random.default_rng(7)
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        x_ref = rng.normal(size=(64, n))
        x_gen = 0.9 * x_ref + 0.1 * rng.normal(size=(64, n))
        paths = export_generation_plot_data(
            X_gen=x_gen,
            X_ref=x_ref,
            V=q,
            lam_gamma=lam,
            out_dir=root,
            prefix="test",
        )
        full_csv = Path(paths["full_spectrum_power_matching_csv"])
        assert full_csv.exists()
        assert count_csv_rows(full_csv) == n
        with np.load(paths["generation_plot_npz"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["full_spectrum_indices"], np.arange(n))
            for key in (
                "reference_power",
                "generated_power",
                "power_ratio",
                "log10_power_ratio",
                "absolute_power_error",
                "relative_power_error",
            ):
                assert data[key].shape == (n,)

    print("All diagnostic export checks passed.")


if __name__ == "__main__":
    main()
