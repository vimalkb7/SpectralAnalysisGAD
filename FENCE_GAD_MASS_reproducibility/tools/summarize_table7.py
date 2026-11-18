from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

SEEDS = [126, 42, 2024, 2025, 3407]
DATASETS = ["PEMS04", "PEMS08"]
METRICS = ("MAE", "RMSE", "MAPE")


def run_name(ds: str, label: str, seed: int) -> str:
    return f"{ds}_SC_TC_GAD_{label}_V3_clean_residual_blend_cov_fbg_seed{seed}"


def result_path(root: Path, ds: str, label: str, seed: int, nfe: int) -> Path:
    return root / ds.lower() / run_name(ds, label, seed) / "table7_locked_test" / f"{ds}_seed{seed}_NFE{nfe}_TEST.csv"


def expected_kind(label: str) -> str:
    return "mass" if label == "MASS" else "fcps"


def load_label(root: Path, label: str, nfes: list[int], allow_missing: bool) -> dict[tuple[str, int, int], dict]:
    out: dict[tuple[str, int, int], dict] = {}
    for ds in DATASETS:
        for nfe in nfes:
            for seed in SEEDS:
                p = result_path(root, ds, label, seed, nfe)
                if not p.exists():
                    if allow_missing:
                        continue
                    raise SystemExit(f"Missing {p}")
                rows = list(csv.DictReader(open(p, newline="")))
                if not rows:
                    raise SystemExit(f"Empty {p}")
                r = rows[-1]
                kind = (r.get("Schedule") or "mass").strip().lower()  # pre-FCPS result files are MASS
                if kind != expected_kind(label):
                    raise SystemExit(f"{p} has Schedule={kind}, expected {expected_kind(label)} for label {label}")
                rec = {"Dataset": ds, "NFE": nfe, "Seed": seed}
                for m in METRICS:
                    rec[m] = float(r[m])
                rec["Nonfinite_Predictions"] = int(float(r.get("Nonfinite_Predictions") or 0))
                rec["ResultFile"] = str(p)
                out[(ds, nfe, seed)] = rec
    return out


def finite(rec: dict) -> bool:
    return rec["Nonfinite_Predictions"] == 0 and all(math.isfinite(rec[m]) for m in METRICS)


def aggregate(records: dict, label: str, nfes: list[int], expected: dict) -> list[dict]:
    agg = []
    for ds in DATASETS:
        for nfe in nfes:
            rows = [records[(ds, nfe, s)] for s in SEEDS if (ds, nfe, s) in records]
            if not rows:
                continue
            ok = [r for r in rows if finite(r)]
            a: dict = {"Dataset": ds, "NFE": nfe, "Seeds": ",".join(str(r["Seed"]) for r in rows)}
            cell = expected.get("datasets", {}).get(ds, {}).get("expected_mean_std", {}).get(str(nfe)) if label == "MASS" else None
            for m in METRICS:
                vals = [r[m] for r in ok]
                a[m] = statistics.mean(vals) if vals else float("nan")
                a[m + "_Std"] = statistics.stdev(vals) if len(vals) > 1 else float("nan")
                if cell is not None:
                    e = cell[m]
                    a[m + "_ExpectedMean"] = e[0]
                    a[m + "_ExpectedStd"] = e[1]
                    a[m + "_MeanDelta"] = a[m] - e[0]
            if label != "MASS" or len(ok) < len(rows):
                # Mean/std above are over finite seeds only; this column says how many.
                a["Finite_Seeds"] = f"{len(ok)}/{len(rows)}"
            agg.append(a)
    return agg


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def paired(records: dict, other: dict, label: str, other_label: str, nfes: list[int]) -> list[dict]:
    """Per (dataset, NFE): LABEL minus OTHER on the same training seed."""
    rows = []
    for ds in DATASETS:
        for nfe in nfes:
            keys = [(ds, nfe, s) for s in SEEDS if (ds, nfe, s) in records and (ds, nfe, s) in other]
            if not keys:
                continue
            r: dict = {"Dataset": ds, "NFE": nfe, "Paired_Seeds": len(keys),
                       f"{label}_Diverged": sum(not finite(records[k]) for k in keys),
                       f"{other_label}_Diverged": sum(not finite(other[k]) for k in keys)}
            both = [k for k in keys if finite(records[k]) and finite(other[k])]
            for m in METRICS:
                d = [records[k][m] - other[k][m] for k in both]
                r[f"{m}_Diff_Mean"] = statistics.mean(d) if d else float("nan")
                r[f"{m}_Diff_Std"] = statistics.stdev(d) if len(d) > 1 else float("nan")
                # A diverged run counts as losing to a finite one.
                r[f"{m}_{other_label}_Better"] = sum(
                    (finite(other[k]) and not finite(records[k]))
                    or (finite(other[k]) and finite(records[k]) and other[k][m] < records[k][m])
                    for k in keys
                )
            rows.append(r)
    return rows


def steps_to_match(agg: list[dict], other_agg: list[dict], label: str, other_label: str) -> list[dict]:
    """Smallest LABEL NFE whose seed-mean is <= the OTHER seed-mean at each OTHER NFE."""
    out = []
    for o in other_agg:
        mine = sorted((a for a in agg if a["Dataset"] == o["Dataset"]), key=lambda a: a["NFE"])
        if not mine:
            continue
        row: dict = {"Dataset": o["Dataset"], f"{other_label}_NFE": o["NFE"],
                     f"{label}_NFEs_Evaluated": ",".join(str(a["NFE"]) for a in mine)}
        for m in METRICS:
            hit = next((a["NFE"] for a in mine if math.isfinite(a[m]) and a[m] <= o[m]), None)
            row[f"{m}_{label}_NFE_To_Match"] = hit if hit is not None else f">{mine[-1]['NFE']}"
            row[f"{m}_Step_Ratio"] = (hit / o["NFE"]) if hit is not None else ""
        out.append(row)
    return out


def fmt(a: dict, m: str) -> str:
    return f"{a[m]:.4f}±{a[m + '_Std']:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root")
    ap.add_argument("--label", default="MASS", help="Run label: MASS, FCPS, FCPSCBAR, ...")
    ap.add_argument("--nfes", default="16,25")
    ap.add_argument("--compare", default=None, help="Label to compare against, e.g. MASS")
    ap.add_argument("--compare-nfes", default=None, help="NFEs of the comparison label (default: 16,25)")
    ap.add_argument("--allow-missing", action="store_true", help="Summarize whatever cells exist (progress checks).")
    args = ap.parse_args()

    root = Path(args.out_root).expanduser().resolve()
    label = args.label.upper()
    nfes = [int(x) for x in args.nfes.split(",") if x.strip()]
    expected = json.load(open(Path(__file__).resolve().parents[1] / "configs" / "PUBLISHED_TABLE_VII.json"))

    records = load_label(root, label, nfes, args.allow_missing)
    agg = aggregate(records, label, nfes, expected)
    per = [{k: v for k, v in r.items() if label != "MASS" or k != "Nonfinite_Predictions"}
           for r in sorted(records.values(), key=lambda r: (DATASETS.index(r["Dataset"]), nfes.index(r["NFE"]), SEEDS.index(r["Seed"])))]
    stem = "TABLE_VII" if label == "MASS" else f"TABLE_VII_{label}"
    agg_path = root / ("TABLE_VII_REPRODUCED.csv" if label == "MASS" else f"{stem}.csv")
    write_rows(root / f"{stem}_PER_SEED.csv", per)
    write_rows(agg_path, agg)
    for a in agg:
        extra = f" [finite seeds {a['Finite_Seeds']}]" if "Finite_Seeds" in a else ""
        prefix = "" if label == "MASS" else f"{label} "  # MASS keeps the original output format
        print(f"{prefix}{a['Dataset']} NFE={a['NFE']}: MAE {fmt(a, 'MAE')}; RMSE {fmt(a, 'RMSE')}; MAPE {fmt(a, 'MAPE')}{extra}")
    print(agg_path)

    if args.compare:
        other_label = args.compare.upper()
        other_nfes = [int(x) for x in (args.compare_nfes or "16,25").split(",") if x.strip()]
        other = load_label(root, other_label, sorted(set(other_nfes) | set(nfes)), allow_missing=True)
        if not other:
            print(f"No {other_label} results under {root}; skipping comparison.")
            return
        other_agg = aggregate(other, other_label, other_nfes, expected)
        pair_rows = paired(records, other, label, other_label, nfes)
        match_rows = steps_to_match(agg, other_agg, label, other_label)
        write_rows(root / f"{stem}_vs_{other_label}_PAIRED.csv", pair_rows)
        write_rows(root / f"{stem}_vs_{other_label}_STEPS_TO_MATCH.csv", match_rows)
        for r in pair_rows:
            print(f"{label}-{other_label} {r['Dataset']} NFE={r['NFE']} ({r['Paired_Seeds']} paired seeds): "
                  + "; ".join(f"{m} {r[f'{m}_Diff_Mean']:+.4f} ({other_label} better {r[f'{m}_{other_label}_Better']}/{r['Paired_Seeds']})" for m in METRICS))
        for r in match_rows:
            print(f"{r['Dataset']}: {label} steps to match {other_label}@{r[f'{other_label}_NFE']}: "
                  + ", ".join(f"{m} {r[f'{m}_{label}_NFE_To_Match']}" for m in METRICS))


if __name__ == "__main__":
    main()
