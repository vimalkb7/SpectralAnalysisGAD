from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


GEN_METRICS = [
    "global_swd_mean",
    "mmd_rbf_z",
    "cov_rel_err",
    "spec_cov_rel_err",
    "band_ratio_l1",
    "graph_stationarity_gap_abs",
    "generated_finite_sample_fraction",
    "generated_nonfinite_flag",
]

FWD_METRICS = [
    "snr_log10_std_mean",
    "snr_log10_range_max",
    "rho_log10_std_mean",
    "stiffness_proxy_max",
    "stiffness_proxy_log10_max",
    "terminal_retained_signal_energy_mean",
    "terminal_noise_prior_rel_err",
    "terminal_rho_prior_rel_err",
]


def _read_run_config(path: Path) -> dict:
    cfg_path = path / "run_config.json"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("args", {})
    except Exception:
        return {}


def collect_csvs(root: Path, filename: str) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(root.rglob(filename)):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[skip] Could not read {csv_path}: {exc}")
            continue
        run_dir = csv_path.parent
        cfg = _read_run_config(run_dir)
        if "dataset_family" not in df.columns and "dataset_family" in cfg:
            df["dataset_family"] = cfg["dataset_family"]
        if "seed" not in df.columns and "seed" in cfg:
            df["seed"] = cfg["seed"]
        df["run_dir"] = str(run_dir)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_mean_std(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    available = [m for m in metrics if m in df.columns]
    if not available:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False)[available].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(c) for c in col if str(c) != ""]).strip("_") for col in agg.columns.to_flat_index()]
    return agg


def save_sampling_efficiency_plots(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    df = df[df["method"].astype(str).str.lower() == "em"].copy()
    if df.empty:
        return
    for dataset in sorted(df["dataset_family"].dropna().unique()):
        sub_d = df[df["dataset_family"] == dataset]
        for metric, ylabel in [
            ("global_swd_mean", "Global SWD ↓"),
            ("cov_rel_err", "Covariance relative error ↓"),
            ("spec_cov_rel_err", "Spectral covariance relative error ↓"),
            ("band_ratio_l1", "Band-ratio L1 ↓"),
        ]:
            if metric not in sub_d.columns:
                continue
            plt.figure(figsize=(7.0, 4.6))
            for schedule in sorted(sub_d["schedule"].dropna().unique()):
                sub = sub_d[sub_d["schedule"] == schedule]
                grouped = sub.groupby("steps")[metric].agg(["mean", "std"]).reset_index().sort_values("steps")
                plt.errorbar(grouped["steps"], grouped["mean"], yerr=grouped["std"], marker="o", linewidth=1.8, capsize=3, label=str(schedule))
            plt.xlabel("Reverse steps")
            plt.ylabel(ylabel)
            plt.title(f"{dataset}: {metric}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            safe_dataset = str(dataset).replace("/", "_")
            plt.savefig(out / f"sampling_efficiency_{safe_dataset}_{metric}.png", dpi=180)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TPAMI synthetic MAS-SNR results across seeds/runs.")
    parser.add_argument("--root", type=str, required=True, help="Root folder containing per-run output directories.")
    parser.add_argument("--out", type=str, default=None, help="Aggregation output folder. Defaults to <root>/aggregate.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    gen = collect_csvs(root, "results_summary.csv")
    if not gen.empty:
        gen.to_csv(out / "all_generation_results.csv", index=False)
        agg_gen = aggregate_mean_std(gen, ["dataset_family", "schedule", "method", "steps"], GEN_METRICS)
        agg_gen.to_csv(out / "aggregate_generation_mean_std.csv", index=False)
        paper = agg_gen[agg_gen["steps"].isin([100, 200, 300])].copy()
        paper.to_csv(out / "paper_generation_table_em_100_200_300_mean_std.csv", index=False)
        save_sampling_efficiency_plots(gen, out)
        print(f"Wrote generation aggregation to {out}")
    else:
        print("No results_summary.csv files found.")

    fwd = collect_csvs(root, "forward_diagnostics_summary_all.csv")
    if not fwd.empty:
        fwd.to_csv(out / "all_forward_diagnostics.csv", index=False)
        agg_fwd = aggregate_mean_std(fwd, ["dataset_family", "schedule"], FWD_METRICS)
        agg_fwd.to_csv(out / "aggregate_forward_stiffness_mean_std.csv", index=False)
        print(f"Wrote forward/stiffness aggregation to {out}")
    else:
        print("No forward_diagnostics_summary_all.csv files found.")


if __name__ == "__main__":
    main()
