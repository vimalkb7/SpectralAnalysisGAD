from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table4_tuning as T  # noqa: E402

METRICS = ["global_swd_mean", "cov_rel_err", "full_spectrum_power_rel_l1", "band_ratio_l1", "smoothness_w1"]
METRIC_LABELS = {"global_swd_mean": "SWD", "cov_rel_err": "Cov. Err.", "full_spectrum_power_rel_l1": "Spectrum",
                 "band_ratio_l1": "Band Ratio", "smoothness_w1": "Smoothness"}
DATA_LABEL = {"stationary_gaussian": "Stationary (broadband WSS)", "block_covariance": "Block covariance"}


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def ms(v):
    v = [x for x in v if x is not None]
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)


def run_row(run: Path, schedule: str, point: tuple) -> dict | None:
    p = run / "results_raw.csv"
    if not p.is_file():
        return None
    for r in T.read_csv(p):
        if r.get("schedule") == schedule and r.get("method") == point[0] and int(float(r.get("nfe", -1))) == point[1]:
            if r.get("evaluation_split") != "test":
                raise SystemExit(f"{p}: row not scored on the test split")
            if str(r.get("stable", "true")).lower() not in ("true", "1", "yes"):
                return None
            return r
    return None


def spectrum_rows(run: Path, row: dict) -> list[dict]:
    path = Path(row.get("generation_spectrum_csv") or "")
    if not path.is_file():                          # fall back to the run folder (results moved)
        path = run / "generation_diagnostics" / path.name
    return T.read_csv(path) if path.is_file() else []


def build(root: Path, phase: str, point: tuple, schedules: list[str], out: Path) -> Path:
    sel = T.load_selection(root)
    by = {(e["dataset"], e["schedule"]): e for e in sel["entries"]}
    vals = {}                                       # (ds, sc) -> seed -> row
    spec = []
    for ds in T.DATASET_ORDER:
        for sc in schedules:
            vals[(ds, sc)] = {}
            for s in T.SEEDS:
                run = T.run_dir(root, {"phase": phase, "dataset": ds, "schedule": sc, "seed": s})
                if not (run / "DONE.json").is_file():
                    raise SystemExit(f"missing run {run}; finish the Table IV {phase} phase first")
                r = run_row(run, sc, point)
                vals[(ds, sc)][s] = r
                if r is not None:
                    for m in spectrum_rows(run, r):
                        spec.append({"dataset": ds, "schedule": sc, "method": point[0], "steps": point[1], "seed": s,
                                     **{k: m.get(k) for k in ("mode", "lam_gamma", "band", "reference_power",
                                                               "generated_power", "log10_power_ratio")}})
    table, paired = [], []
    for ds in T.DATASET_ORDER:
        for sc in schedules:
            rows = vals[(ds, sc)]
            e = by[(ds, sc)]
            row = {"dataset": ds, "schedule": T.LABEL[sc], "method": point[0], "steps": point[1],
                   "config_id": e.get("config_id"), "params": json.dumps(e.get("params"))}
            for m in METRICS:
                mu, sd, n = ms([num(r.get(m)) if r else None for r in rows.values()])
                row[f"{m}_mean"], row[f"{m}_std"], row["seeds_finite"] = mu, sd, n
            table.append(row)
            if sc != "MAS-SNR" and "MAS-SNR" in schedules:
                mk = vals[(ds, "MAS-SNR")]
                for m in METRICS:
                    d = [num(mk[s][m]) - num(rows[s][m]) for s in T.SEEDS
                         if mk[s] and rows[s] and num(mk[s].get(m)) is not None and num(rows[s].get(m)) is not None]
                    if not d:
                        continue
                    mu, sd, _ = ms(d)
                    wins = sum(x < 0 for x in d)
                    paired.append({"dataset": ds, "baseline": T.LABEL[sc], "metric": METRIC_LABELS[m], "paired_seeds": len(d),
                                   "mean_diff_mass_minus_baseline": mu, "std_diff": sd, "seeds_mass_better": wins,
                                   "mass_better": bool(mu < 0 and wins >= 4 and len(d) == len(T.SEEDS))})
    out.mkdir(parents=True, exist_ok=True)
    T.write_csv(out / "table5_from_table4.csv", table)
    T.write_csv(out / "table5_from_table4_paired.csv", paired)

    def fmt(m, s):
        if m is None:
            return "diverged"
        return f"{m:.2f} ± {s:.2f}" if abs(m) >= 10 else f"{m:.4f} ± {s:.4f}"
    name = "DPM-Solver++2M" if point[0] == "dpmpp2m" else point[0].upper()
    lines = [f"Table V from the Table IV test runs: {name}, {point[1]} steps, test split, mean ± std over 5 paired seeds "
             "(one fixed graph); best per dataset in bold.", "",
             "| Dataset | Schedule | " + " | ".join(METRIC_LABELS[m] + " ↓" for m in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for ds in T.DATASET_ORDER:
        rows = [r for r in table if r["dataset"] == ds]
        best = {m: min(((r[f"{m}_mean"], r["schedule"]) for r in rows if r[f"{m}_mean"] is not None), default=(None, None))[1]
                for m in METRICS}
        for r in rows:
            cells = [(f"**{fmt(r[f'{m}_mean'], r[f'{m}_std'])}**" if best[m] == r["schedule"] else fmt(r[f"{m}_mean"], r[f"{m}_std"]))
                     for m in METRICS]
            lines.append(f"| {DATA_LABEL[ds]} | {r['schedule']} | " + " | ".join(cells) + " |")
    lines += ["", "Paired per seed (MASS - baseline; 'better' = mean favours MASS and >= 4/5 seeds):", "",
              "| Dataset | Baseline | Metric | Mean diff ± std | Seeds MASS better | Better |", "|---|---|---|---|---|---|"]
    lines += [f"| {p['dataset']} | {p['baseline']} | {p['metric']} | {p['mean_diff_mass_minus_baseline']:+.4g} ± {p['std_diff']:.3g} | "
              f"{p['seeds_mass_better']}/{p['paired_seeds']} | {'yes' if p['mass_better'] else 'no'} |" for p in paired]
    lines += ["", "Settings (selected on validation in the Table IV sweep):", ""]
    lines += [f"- {r['dataset']} {r['schedule']}: {r['config_id']} {r['params']}" for r in table]
    (out / "TABLE5_FROM_TABLE4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure 2 data
    if spec:
        T.write_csv(out / "figure2_spectral_power_long.csv", spec)
        groups = defaultdict(list)
        for r in spec:
            groups[(r["dataset"], r["schedule"], int(float(r["mode"])))].append(r)
        by_mode = []
        for (ds, sc, mode), rr in sorted(groups.items()):
            row = {"dataset": ds, "schedule": sc, "method": point[0], "steps": point[1], "mode": mode, "band": rr[0]["band"],
                   "num_seeds": len(rr)}
            for col in ("lam_gamma", "reference_power", "generated_power", "log10_power_ratio"):
                mu, sd, _ = ms([num(r.get(col)) for r in rr])
                row[f"{col}_mean"], row[f"{col}_std"] = mu, sd
            by_mode.append(row)
        T.write_csv(out / "figure2_spectral_power_by_mode.csv", by_mode)
    (out / "figure2_manifest.json").write_text(json.dumps({
        "what": f"Mode-wise power E[x_k^2] of generated signals ({name}, {point[1]} steps, test split) and of the test "
                "signals against lam_gamma; one fixed graph, so averaging over seeds by mode index is valid.",
        "bands": "the 'band' column marks the lowest and highest quarter of the spectrum",
        "schedules": [T.LABEL[s] for s in schedules]}, indent=2), encoding="utf-8")
    bundle = root / f"{out.name}_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():
                z.write(p, arcname=f"table5_from_table4/{p.name}")
        for p in (root / "selection" / "selection.json", root / "selection" / "SELECTION.md"):
            if p.is_file():
                z.write(p, arcname=f"table5_from_table4/{p.name}")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Table IV results folder (with selection/ and test/)")
    ap.add_argument("--phase", default="test", choices=("test", "ddim"), help="which Table IV runs to read")
    ap.add_argument("--point", default="dpmpp2m:10", help="sampler and steps, e.g. dpmpp2m:10")
    ap.add_argument("--schedules", default="MAS-SNR,Isotropic-VP-Cosine,Cosine")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    method, steps = a.point.split(":")
    schedules = [s.strip() for s in a.schedules.split(",") if s.strip()]
    bad = [s for s in schedules if s not in T.SCHEDULES]
    if bad:
        raise SystemExit(f"unknown schedules {bad}; choose from {T.SCHEDULES}")
    out = a.out or a.root / "table5_from_table4"
    bundle = build(a.root, a.phase, (method, int(steps)), schedules, out)
    print((out / "TABLE5_FROM_TABLE4.md").read_text())
    print(f"Upload: {bundle}")


if __name__ == "__main__":
    main()
