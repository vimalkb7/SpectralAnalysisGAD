from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.fcps_schedule import calibrate_fcps, fcps_cbar, mode_snr  # noqa: E402

# Published MASS settings (README, slurm/0[1-3]_*.sbatch).
PRESETS = {
    "amazon_game": ("content_k20_pu1000.npz", 1e-2),
    "yelp": ("content_k50_pu1000.npz", 1e-1),
    "citeulike_t": ("content_k20_pu10000.npz", 3e-2),
}


def report(name: str, asset: Path, args: argparse.Namespace, r_min: float) -> None:
    data = np.load(asset, allow_pickle=False)
    lam = np.maximum(data["eigvals_shifted"].astype(np.float64), 1e-12)
    s = np.maximum(data["s_real"].astype(np.float64), 1e-12)
    g = lam * s / args.sigma ** 2
    grid = np.linspace(args.t_eps, args.T_diff, args.steps)
    u = (grid - args.t_eps) / (args.T_diff - args.t_eps)
    r_mass = np.exp((1.0 - u) * np.log(args.r_max) + u * np.log(r_min))

    print(f"\n== {name}: {asset.name}  (n={lam.size}, cond(L_gamma)={lam.max() / lam.min():.0f}, "
          f"r_max={args.r_max:g}, r_min={r_min:g})")
    for alpha in args.alphas:
        try:
            p = calibrate_fcps(lam, s, args.sigma, args.r_max, r_min, args.t_eps, args.T_diff, alpha)
        except ValueError as exc:
            print(f"  alpha={alpha:g}: NOT ADMISSIBLE. {exc}")
            continue
        print(f"  alpha={alpha:g}: cbar_T={p['fcps_cbar_T']:.4g}  c_min={p['fcps_c_min']:.4g}  "
              f"(alpha_min={p['fcps_alpha_min_for_start_match']:.3f})")
        print("    step   MASS SNR (all modes)   FCPS SNR min / median / max")
        for i in sorted({0, 1, 2, 3, args.steps // 4, args.steps // 2, args.steps - 1}):
            cb = fcps_cbar(grid[i], p["fcps_c_min"], p["fcps_cbar_T"], alpha, args.T_diff)
            v = mode_snr(cb, lam, g)
            print(f"    {i:4d}   {r_mass[i]:18.4g}   {v.min():.3g} / {np.median(v):.3g} / {v.max():.3g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS) + ["all"], default=None)
    ap.add_argument("--graph_root", type=Path, default=ROOT / "graph_assets")
    ap.add_argument("--graph_asset_path", type=Path, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--r_max", type=float, default=1e4)
    ap.add_argument("--r_min", type=float, default=None, help="Required with --graph_asset_path")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--t_eps", type=float, default=1e-3)
    ap.add_argument("--T_diff", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    if args.graph_asset_path is not None:
        if args.r_min is None:
            ap.error("--r_min is required with --graph_asset_path")
        report(args.graph_asset_path.stem, args.graph_asset_path, args, args.r_min)
        return
    if args.preset is None:
        ap.error("give --preset or --graph_asset_path")
    names = list(PRESETS) if args.preset == "all" else [args.preset]
    for name in names:
        filename, r_min = PRESETS[name]
        asset = args.graph_root / filename
        if not asset.exists():
            print(f"\n== {name}: {asset} not built yet; skipping")
            continue
        report(name, asset, args, args.r_min if args.r_min is not None else r_min)


if __name__ == "__main__":
    main()
