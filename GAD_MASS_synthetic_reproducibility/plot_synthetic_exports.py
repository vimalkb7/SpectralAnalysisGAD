from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STYLE: dict[str, Any] = {
    "font_family": "DejaVu Sans",
    "font_size": 8.0,
    "axes_label_size": 8.0,
    "legend_size": 7.0,
    "line_width": 1.4,
    "marker_size": 3.2,
    "figure_width": 7.16,
    "single_column_width": 3.5,
    "dpi": 300,
    "grid_alpha": 0.18,
    "schedule_styles": {
        "FCPS": {"color": "#4D4D4D", "linestyle": "--", "marker": "o"},
        "Uniform": {"color": "#E69F00", "linestyle": ":", "marker": "^"},
        "Cosine": {"color": "#0072B2", "linestyle": "-.", "marker": "s"},
        "MAS-SNR": {"color": "#009E73", "linestyle": "-", "marker": "D"},
        "Isotropic-VP-Cosine": {"color": "#CC79A7", "linestyle": "--", "marker": "v"},
        "Isotropic-VP-Uniform": {"color": "#D55E00", "linestyle": ":", "marker": "x"},
    },
    "target_style": {"color": "#000000", "linestyle": "-", "marker": None},
}


def load_style(path: str | None) -> dict[str, Any]:
    style = json.loads(json.dumps(DEFAULT_STYLE))
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        for key, value in user.items():
            if key == "schedule_styles":
                style[key].update(value)
            else:
                style[key] = value
    return style


def apply_style(style: dict[str, Any]) -> None:
    plt.rcParams.update(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.labelsize": style["axes_label_size"],
            "axes.titlesize": style["axes_label_size"],
            "legend.fontsize": style["legend_size"],
            "xtick.labelsize": style["font_size"],
            "ytick.labelsize": style["font_size"],
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def save_figure(fig: plt.Figure, output_stem: str | Path, formats: list[str], style: dict[str, Any]) -> list[Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt.lower() == "png":
            kwargs["dpi"] = int(style["dpi"])
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def schedule_style(name: str, style: dict[str, Any]) -> dict[str, Any]:
    base = style["schedule_styles"].get(name, {})
    return {
        "color": base.get("color", None),
        "linestyle": base.get("linestyle", "-"),
        "marker": base.get("marker", None),
        "linewidth": float(style["line_width"]),
        "markersize": float(style["marker_size"]),
        "markevery": max(1, int(base.get("markevery", 24))),
    }


def add_panel_labels(axes: np.ndarray) -> None:
    labels = "abcdefghijklmnopqrstuvwxyz"
    for i, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.text(-0.13, 1.04, f"({labels[i]})", transform=ax.transAxes, fontweight="bold", va="bottom")


def plot_forward_bundle(args, style: dict[str, Any]) -> None:
    d = load_npz(args.input)
    modes = d["selected_modes"].astype(int)
    u = d["normalized_time"]
    curve_u = d.get("forward_curve_normalized_time", u)
    curve_a = d.get("forward_curve_a", d["a"])
    curve_nu = d.get("forward_curve_nu", d["nu"])
    fig, axes = plt.subplots(1, 2, figsize=(style["figure_width"], 2.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for j, k in enumerate(modes):
        color = cmap(j / max(len(modes) - 1, 1))
        label = rf"$k={k}$, $\lambda_k^\gamma={d['lam_gamma'][k]:.2g}$"
        axes[0].plot(curve_u, curve_a[:, k], color=color, linewidth=style["line_width"], label=label)
        axes[1].plot(curve_u, curve_nu[:, k], color=color, linewidth=style["line_width"], label=label)
        axes[1].axhline(d["prior_power"][k], color=color, linewidth=0.7, linestyle=":", alpha=0.7)
    axes[0].set(xlabel=r"Normalized diffusion time $t/T$", ylabel=r"Attenuation $a_{t,k}$")
    axes[1].set(xlabel=r"Normalized diffusion time $t/T$", ylabel=r"Noise variance $\nu_{t,k}$")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=style["grid_alpha"])
    axes[0].legend(ncol=2, frameon=False, loc="best")
    add_panel_labels(axes)
    save_figure(fig, Path(args.output_dir) / "forward_attenuation_variance", args.formats, style)

    fig, ax = plt.subplots(figsize=(style["single_column_width"], 2.7), constrained_layout=True)
    mesh = ax.pcolormesh(d["lam_gamma"], u, d["log10_snr"], shading="auto", cmap=args.cmap)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}\mathrm{SNR}_{t,k}$")
    ax.set(xlabel=r"Shifted Laplacian eigenvalue $\lambda_k^\gamma$", ylabel=r"Normalized time $t/T$")
    save_figure(fig, Path(args.output_dir) / "snr_heatmap", args.formats, style)


def plot_training_bundle(args, style: dict[str, Any]) -> None:
    d = load_npz(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(style["figure_width"], 2.45), constrained_layout=True)
    axes[0].plot(d["epoch"], d["train_loss"], linewidth=style["line_width"], label="Training")
    if len(d["validation_epoch"]):
        axes[0].plot(
            d["validation_epoch"],
            d["validation_loss"],
            linewidth=style["line_width"],
            marker="o",
            markersize=style["marker_size"],
            label="Validation",
        )
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(d["epoch"], d["learning_rate"], linewidth=style["line_width"])
    axes[1].set(xlabel="Epoch", ylabel="Learning rate")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=style["grid_alpha"])
    add_panel_labels(axes)
    save_figure(fig, Path(args.output_dir) / "training_curves", args.formats, style)


def _plot_spectrum_band(ax, d: dict[str, np.ndarray], indices: np.ndarray, label: str, style: dict[str, Any]) -> None:
    target = style["target_style"]
    ax.plot(
        d["lam_gamma"][indices],
        d["reference_power"][indices],
        color=target["color"],
        linestyle=target["linestyle"],
        linewidth=style["line_width"],
        label="Reference",
    )
    ax.plot(
        d["lam_gamma"][indices],
        d["generated_power"][indices],
        linewidth=style["line_width"],
        linestyle="--",
        marker="o",
        markersize=style["marker_size"],
        markevery=max(1, len(indices) // 6),
        label="Generated",
    )
    ax.set(xlabel=r"$\lambda_k^\gamma$", ylabel=r"Spectral power $\mathbb{E}[\tilde{x}_k^2]$")
    ax.set_yscale("log")
    ax.set_title(label)
    ax.grid(True, which="both", alpha=style["grid_alpha"])


def plot_generation_bundle(args, style: dict[str, Any]) -> None:
    d = load_npz(args.input)
    low = d["low_indices"].astype(int)
    high = d["high_indices"].astype(int)
    full = d["mode_index"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(style["figure_width"], 2.55), constrained_layout=True)
    _plot_spectrum_band(axes[0], d, low, "Low-frequency band", style)
    _plot_spectrum_band(axes[1], d, high, "High-frequency band", style)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    add_panel_labels(axes)
    save_figure(fig, Path(args.output_dir) / "bandlimited_spectral_match", args.formats, style)

    fig, ax = plt.subplots(figsize=(style["single_column_width"], 2.55), constrained_layout=True)
    _plot_spectrum_band(ax, d, full, "Full spectrum", style)
    ax.legend(frameon=False)
    save_figure(fig, Path(args.output_dir) / "full_spectrum_match", args.formats, style)

    fig, ax = plt.subplots(figsize=(style["single_column_width"], 2.4), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(d["lam_gamma"], d["log10_power_ratio"], linewidth=style["line_width"])
    ax.set(xlabel=r"$\lambda_k^\gamma$", ylabel=r"$\log_{10}(\widehat p_k/p_k)$")
    ax.grid(True, alpha=style["grid_alpha"])
    save_figure(fig, Path(args.output_dir) / "spectral_log_ratio", args.formats, style)


def resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    candidate = manifest_path.parent / p
    if candidate.exists():
        return candidate
    # Common case: manifest stores a path rooted at the experiment directory.
    candidate = manifest_path.parent / p.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve {value!r} from {manifest_path}")


def parse_label_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, received: {spec}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def load_forward_manifest(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for entry in manifest["entries"]:
        out[entry["schedule"]] = load_npz(resolve_manifest_path(path, entry["npz"]))
    return out


def plot_figure_a(args, style: dict[str, Any]) -> None:
    datasets = [parse_label_path(x) for x in args.dataset]
    nrows = len(datasets)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(style["figure_width"], max(2.25, 2.05 * nrows)),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
    )
    for row, (dataset_label, manifest_path) in enumerate(datasets):
        bundles = load_forward_manifest(manifest_path)
        for schedule in args.schedules:
            if schedule not in bundles:
                continue
            d = bundles[schedule]
            st = schedule_style(schedule, style)
            axes[row, 0].plot(d["normalized_time"], d["log10_snr_std"], label=schedule, **st)
            axes[row, 1].plot(d["normalized_time"], d[args.stiffness_key], label=schedule, **st)
        axes[row, 0].set_ylabel(rf"{dataset_label}\n$\mathrm{{Std}}_k[\log_{{10}}\mathrm{{SNR}}_{{t,k}}]$")
        axes[row, 1].set_ylabel(rf"{dataset_label}\nReverse-rate proxy")
        axes[row, 0].grid(True, alpha=style["grid_alpha"])
        axes[row, 1].grid(True, which="both", alpha=style["grid_alpha"])
        axes[row, 1].set_yscale("log")
    axes[-1, 0].set_xlabel(r"Normalized diffusion time $t/T$")
    axes[-1, 1].set_xlabel(r"Normalized diffusion time $t/T$")
    axes[0, 0].set_title("Mode-wise SNR dispersion")
    axes[0, 1].set_title("Reverse-time rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    add_panel_labels(axes)
    save_figure(fig, args.output, args.formats, style)


def load_figure_b_spec(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec["datasets"]


def plot_figure_b(args, style: dict[str, Any]) -> None:
    datasets = load_figure_b_spec(args.spec)
    nrows = len(datasets)
    fig, axes = plt.subplots(
        nrows,
        2,
        figsize=(style["figure_width"], max(2.3, 2.05 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, dataset in enumerate(datasets):
        entries = dataset["entries"]
        spec_base = Path(args.spec).resolve().parent
        loaded = []
        for entry in entries:
            data_path = Path(entry["path"])
            if not data_path.is_absolute():
                data_path = spec_base / data_path
            loaded.append((entry["schedule"], load_npz(data_path)))
        for col, band_key in enumerate(["low_indices", "high_indices"]):
            ax = axes[row, col]
            first = loaded[0][1]
            idx = first[band_key].astype(int)
            if args.representation == "power":
                ax.plot(
                    first["lam_gamma"][idx],
                    first["reference_power"][idx],
                    label="Reference",
                    color=style["target_style"]["color"],
                    linewidth=style["line_width"],
                )
                for schedule, d in loaded:
                    st = schedule_style(schedule, style)
                    ax.plot(d["lam_gamma"][idx], d["generated_power"][idx], label=schedule, **st)
                ax.set_yscale("log")
                ax.set_ylabel(rf"{dataset['label']}\nSpectral power")
            else:
                ax.axhline(0.0, color="black", linewidth=0.8)
                for schedule, d in loaded:
                    st = schedule_style(schedule, style)
                    ax.plot(d["lam_gamma"][idx], d["log10_power_ratio"][idx], label=schedule, **st)
                ax.set_ylabel(rf"{dataset['label']}\n$\log_{{10}}(\widehat p_k/p_k)$")
            ax.set_xlabel(r"$\lambda_k^\gamma$")
            ax.grid(True, which="both", alpha=style["grid_alpha"])
    axes[0, 0].set_title("Low-frequency modes")
    axes[0, 1].set_title("High-frequency modes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    add_panel_labels(axes)
    save_figure(fig, args.output, args.formats, style)


def plot_full_spectrum_comparison(args, style: dict[str, Any]) -> None:
    datasets = load_figure_b_spec(args.spec)
    nrows = len(datasets)
    ncols = max(len(dataset["entries"]) for dataset in datasets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(style["figure_width"], max(2.45, 2.25 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    spec_base = Path(args.spec).resolve().parent
    for row, dataset in enumerate(datasets):
        entries = dataset["entries"]
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(entries):
                ax.axis("off")
                continue
            entry = entries[col]
            data_path = Path(entry["path"])
            if not data_path.is_absolute():
                data_path = spec_base / data_path
            d = load_npz(data_path)
            idx = d.get("full_spectrum_indices", d["mode_index"]).astype(int)
            _plot_spectrum_band(
                ax,
                d,
                idx,
                f"{dataset['label']} | {entry['schedule']}",
                style,
            )
            if col == 0:
                ax.set_ylabel(r"Spectral power $\mathbb{E}[\tilde{x}_k^2]$")
            else:
                ax.set_ylabel("")
            ax.set_xlabel(r"Shifted Laplacian eigenvalue $\lambda_k^\gamma$")
            ax.legend(frameon=False, loc="best")
    save_figure(fig, args.output, args.formats, style)


def add_common_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--formats", default="pdf,svg", help="Comma-separated output formats.")
    parser.add_argument("--style_config", default=None, help="Optional JSON overriding fonts, colors, and line styles.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("forward", help="Render the original attenuation/variance and SNR heatmap diagnostics.")
    q.add_argument("--input", required=True, help="forward_plot_data.npz")
    q.add_argument("--output_dir", required=True)
    q.add_argument("--cmap", default="viridis")
    add_common_output_args(q)

    q = sub.add_parser("training", help="Render training and learning-rate curves.")
    q.add_argument("--input", required=True, help="training_plot_data.npz")
    q.add_argument("--output_dir", required=True)
    add_common_output_args(q)

    q = sub.add_parser("generation", help="Render existing spectral-recovery plots from one generation bundle.")
    q.add_argument("--input", required=True, help="*_generation_plot_data.npz")
    q.add_argument("--output_dir", required=True)
    add_common_output_args(q)

    q = sub.add_parser("figure-a", help="Render proposed Figure A from one or more forward manifests.")
    q.add_argument("--dataset", action="append", required=True, help="Repeat LABEL=/path/to/forward_plot_manifest.json")
    q.add_argument("--schedules", default="FCPS,Uniform,Cosine,MAS-SNR")
    q.add_argument("--stiffness_key", default="stiffness_optimal_max_abs", choices=["stiffness_optimal_max_abs", "stiffness_proxy_max_abs"])
    q.add_argument("--output", required=True, help="Output path without extension.")
    add_common_output_args(q)

    q = sub.add_parser("figure-b", help="Render proposed Figure B from a JSON specification.")
    q.add_argument("--spec", required=True, help="JSON listing datasets and generation NPZ entries.")
    q.add_argument("--representation", default="ratio", choices=["ratio", "power"])
    q.add_argument("--output", required=True, help="Output path without extension.")
    add_common_output_args(q)

    q = sub.add_parser(
        "figure-3-full",
        help="Render a Figure-3-like target/generated power comparison over the complete spectrum.",
    )
    q.add_argument("--spec", required=True, help="JSON listing datasets and generation NPZ entries.")
    q.add_argument("--output", required=True, help="Output path without extension.")
    add_common_output_args(q)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if hasattr(args, "schedules") and isinstance(args.schedules, str):
        args.schedules = [x.strip() for x in args.schedules.split(",") if x.strip()]
    style = load_style(args.style_config)
    apply_style(style)

    if args.command == "forward":
        plot_forward_bundle(args, style)
    elif args.command == "training":
        plot_training_bundle(args, style)
    elif args.command == "generation":
        plot_generation_bundle(args, style)
    elif args.command == "figure-a":
        plot_figure_a(args, style)
    elif args.command == "figure-b":
        plot_figure_b(args, style)
    elif args.command == "figure-3-full":
        plot_full_spectrum_comparison(args, style)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
