from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import table4_tuning as T  # noqa: E402

GAMMAS = [3e-4, 1e-3, 3e-3, 1e-2]
SIGMAS = [0.25, 0.5, 1.0, 2.0]
SIGMA0 = 1.0
DELTA = 1e-3
POINT = ("dpmpp2m", 10)
DATASETS = ["stationary_gaussian"]
METRICS = ["global_swd_mean", "cov_rel_err", "full_spectrum_power_rel_l1", "band_ratio_l1", "smoothness_w1"]


def base(table4_root: Path, ds: str) -> dict:
    e = next(x for x in T.load_selection(table4_root)["entries"] if x["dataset"] == ds and x["schedule"] == "MAS-SNR")
    return {"gamma": float(e["params"]["gamma"]), "delta0": float(e["params"]["delta0"]), "config_id": e["config_id"]}


def configs(table4_root: Path, ds: str) -> list[dict]:
    b = base(table4_root, ds)
    out, seen = [], set()
    for g in GAMMAS:
        out.append({"id": f"gamma{g:g}_sigma{SIGMA0:g}", "gamma": g, "sigma": SIGMA0, "delta0": b["delta0"], "blocks": ["gamma"]})
        seen.add((g, SIGMA0))
    for s in SIGMAS:
        if (b["gamma"], s) in seen:
            next(c for c in out if (c["gamma"], c["sigma"]) == (b["gamma"], s))["blocks"].append("sigma")
            continue
        out.append({"id": f"gamma{b['gamma']:g}_sigma{s:g}", "gamma": b["gamma"], "sigma": s, "delta0": b["delta0"], "blocks": ["sigma"]})
    if (b["gamma"], SIGMA0) not in seen:
        raise SystemExit(f"selected gamma {b['gamma']} is not in the gamma grid {GAMMAS}")
    return out


def tasks(table4_root: Path) -> list[dict]:
    return [{"dataset": ds, "config": c, "seed": s} for ds in DATASETS for c in configs(table4_root, ds) for s in T.SEEDS]


def run_dir(root: Path, t: dict) -> Path:
    return root / "runs" / t["dataset"] / t["config"]["id"] / f"seed_{t['seed']}"


def driver_args(t: dict, problem: str) -> list[str]:
    c, r = t["config"], int(t["seed"])
    common = list(T.COMMON)
    common[common.index("--sigma") + 1] = f"{c['sigma']:g}"
    return ["--problem", problem, "--candidate_id", f"table9_{c['id']}", "--schedulers", "MAS-SNR",
            "--model_seed", str(3000 + r), "--train_seed", str(4000 + r), "--validation_seed", str(5000 + r),
            "--sample_seeds", str(6000 + r), "--plot_export_sample_seeds", str(6000 + r),
            "--evaluation_split", "test", "--study_phase", "table9_ablation", *common,
            "--gamma", f"{c['gamma']:g}", "--mass_start_a2_error", f"{c['delta0']:g}", "--mass_terminal_a2", f"{DELTA:g}"]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ms(v):
    v = [x for x in v if x is not None]
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)


def metrics_at(run: Path) -> dict | None:
    p = run / "results_raw.csv"
    if not p.is_file():
        return None
    for r in T.read_csv(p):
        if r.get("schedule") == "MAS-SNR" and r.get("method") == POINT[0] and int(float(r.get("nfe", -1))) == POINT[1]:
            if r.get("evaluation_split") != "test":
                raise SystemExit(f"{p}: row not scored on the test split")
            return r if str(r.get("stable", "true")).lower() in ("true", "1", "yes") else None
    return None


def theory(problem, c: dict) -> dict:
    """cond(L_gamma), endpoints, M and the schedule-only stiffness of one setting (as in Table III)."""
    from diagnostics import run_forward_diagnostics
    from schedules import build_schedules, mass_endpoints_from_tolerances
    from theory_checks import mass_M
    lam = problem.shifted_eigenvalues(c["gamma"]); s_k = problem.s_k_train
    r_max, r_min = mass_endpoints_from_tolerances(lam, s_k, c["sigma"], start_a2_error=c["delta0"], terminal_a2=DELTA)
    sch = build_schedules(lam_gamma=lam, s_k=s_k, sigma=c["sigma"], T=1.0, fcps_c0=100.0, fcps_cmin=0.1, fcps_alpha=4.0,
                          r_max=float(r_max), r_min=float(r_min), match_terminal_energy=False, include_isotropic_vp=False)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:          # the per-mode exports are large and not needed here
        summ = run_forward_diagnostics(X_train_np=problem.X_train, V_np=problem.V, lam_gamma_np=lam, sigma=c["sigma"], schedules=sch,
                                       scheduler_names=["MAS-SNR"], dataset_name="table9", out_dir=tmp, num_times=2048,
                                       num_extreme_modes=4, terminal_num_samples=512, seed=0, pmin=1e-10, t_min=1e-4,
                                       stiffness_h_proxy=0.0)
    L = float(next(r for r in summ if r["schedule"] == "MAS-SNR")["max_stiffness_proxy"])
    return {"cond_L_gamma": float(lam[-1] / lam[0]), "r_max": float(r_max), "r_min": float(r_min),
            "M": float(mass_M(sch["MAS-SNR"])), "L_S0": L}


def aggregate(root: Path, table4_root: Path) -> Path:
    from synthetic_problem import FrozenSyntheticProblem
    tt = tasks(table4_root)
    missing = [t for t in tt if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} Table IX runs not finished; first missing: {missing[0]['config']['id']} seed {missing[0]['seed']}")
    table = []
    for ds in DATASETS:
        problem = FrozenSyntheticProblem.load(root / "problems" / f"{T.DATASETS[ds]['problem']}.npz")
        b = base(table4_root, ds)
        for c in configs(table4_root, ds):
            th = theory(problem, c)
            rows = [metrics_at(run_dir(root, {"dataset": ds, "config": c, "seed": s})) for s in T.SEEDS]
            row = {"dataset": ds, "config_id": c["id"], "blocks": "+".join(c["blocks"]), "gamma": c["gamma"], "sigma": c["sigma"],
                   "delta0": c["delta0"], **th, "selected_in_table4": (c["gamma"], c["sigma"]) == (b["gamma"], SIGMA0)}
            for m in METRICS:
                mu, sd, n = ms([num(r.get(m)) if r else None for r in rows])
                row[f"{m}_mean"], row[f"{m}_std"], row["seeds_finite"] = mu, sd, n
            # the Table IV setting must reproduce the Table IV test runs
            if row["selected_in_table4"]:
                old = [metrics_at(T.run_dir(table4_root, {"phase": "test", "dataset": ds, "schedule": "MAS-SNR", "seed": s})) for s in T.SEEDS]
                d = [abs(num(o["global_swd_mean"]) - num(r["global_swd_mean"])) for o, r in zip(old, rows) if o and r]
                row["max_abs_swd_diff_vs_table4"] = max(d) if d else ""
            table.append(row)
    out = root / "table9"
    out.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in table:
        fields += [k for k in r if k not in fields]
    with (out / "table9_ablation.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, restval=""); w.writeheader(); w.writerows(table)
    f = lambda m, s: "–" if m is None else f"{m:.4f} ± {s:.4f}"
    lines = ["Table IX: ablation of the GAD channel parameters (Table IV setup, MASS, DPM-Solver++2M 10 steps, test split, "
             "mean ± std over 5 paired seeds). Each block varies one parameter; * = the Table IV setting.", "",
             "| Block | gamma | sigma | cond(L_gamma) | M | L_S^(0) ↓ | SWD ↓ | Spectrum err. ↓ |", "|---|---|---|---|---|---|---|---|"]
    for blk in ("gamma", "sigma"):
        for r in sorted((r for r in table if blk in r["blocks"].split("+")), key=lambda r: r[blk]):
            star = "*" if r["selected_in_table4"] else ""
            lines.append(f"| {blk} | {r['gamma']:g}{star} | {r['sigma']:g} | {r['cond_L_gamma']:.0f} | {r['M']:.2f} | {r['L_S0']:.2f} | "
                         f"{f(r['global_swd_mean_mean'], r['global_swd_mean_std'])} | "
                         f"{f(r['full_spectrum_power_rel_l1_mean'], r['full_spectrum_power_rel_l1_std'])} |")
    chk = [r.get("max_abs_swd_diff_vs_table4") for r in table if r.get("max_abs_swd_diff_vs_table4") not in (None, "")]
    if chk:
        lines += ["", f"Reproduction check: the Table IV setting matches the Table IV test runs to within {max(chk):.2e} SWD per seed."]
    lines += ["", "All five metrics per setting: table9_ablation.csv"]
    (out / "TABLE9.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle = root / "table9_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("TABLE9.md", "table9_ablation.csv"):
            z.write(out / name, arcname=f"table9/{name}")
    print((out / "TABLE9.md").read_text())
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("count", "status", "aggregate"):
        sp = sub.add_parser(name); sp.add_argument("--root", type=Path, required=True); sp.add_argument("--table4-root", type=Path, required=True)
    for name in ("describe", "args"):
        sp = sub.add_parser(name); sp.add_argument("--root", type=Path, required=True); sp.add_argument("--table4-root", type=Path, required=True)
        sp.add_argument("--index", type=int, required=True)
        if name == "args":
            sp.add_argument("--problem", required=True); sp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tt = tasks(a.table4_root)
    if a.cmd == "count":
        print(len(tt))
    elif a.cmd == "status":
        miss = [i for i, t in enumerate(tt) if not (run_dir(a.root, t) / "DONE.json").is_file()]
        print(f"table9: {len(tt) - len(miss)} of {len(tt)} done" + (f"; missing {miss}" if miss else ""))
        raise SystemExit(3 if miss else 0)
    elif a.cmd in ("describe", "args"):
        if not 0 <= a.index < len(tt):
            raise SystemExit(f"index {a.index} out of range 0-{len(tt) - 1}")
        t = tt[a.index]
        info = {"dataset": t["dataset"], "config": t["config"], "seed": t["seed"], "run_dir": str(run_dir(a.root, t)),
                "problem_name": T.DATASETS[t["dataset"]]["problem"], "problem_args": T.DATASETS[t["dataset"]]["problem_args"]}
        if a.cmd == "describe":
            print(json.dumps(info))
        else:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({**info, "driver_args": driver_args(t, a.problem)}, indent=2), encoding="utf-8")
            print(f"Table IX task {a.index}: {t['config']['id']} seed {t['seed']}")
    elif a.cmd == "aggregate":
        print(f"Upload: {aggregate(a.root, a.table4_root)}")


if __name__ == "__main__":
    main()
