from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


# Every synthetic experiment uses five seeds (Tables III, IV, V, IX and
# Figures 1-2); each aggregator checks that exact count unless told otherwise.
DEFAULT_NUM_SEEDS = 5
DATASETS = ("stationary_gaussian", "block_covariance")

METRICS = (
    "global_swd_mean",
    "cov_rel_err",
    "full_spectrum_power_rel_l1",
    "band_ratio_l1",
    "smoothness_w1",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty aggregate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def is_stable(row: dict) -> bool:
    return str(row.get("stable", "true")).lower() in {"true", "1", "yes"}


def summarize(values: list[float]) -> tuple[float, float]:
    if not values:
        raise RuntimeError("No finite values available for an expected result")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed_"):
            return int(part.removeprefix("seed_"))
    raise RuntimeError(f"No seed_N component in {path}")


def _check_seed_dirs(files: list[Path], base: Path, num_seeds: int, what: str, depth: int = 0) -> None:
    """Each dataset (path component ``depth`` under base) must have seeds 0..num_seeds-1 exactly."""
    seen: dict[str, set[int]] = defaultdict(set)
    for path in files:
        seen[path.relative_to(base).parts[depth]].add(seed_from_path(path))
    want = set(range(num_seeds))
    bad = {k: sorted(v) for k, v in seen.items() if v != want}
    if bad:
        raise RuntimeError(f"{what}: expected seeds {sorted(want)} everywhere, found {bad} under {base}")


def aggregate_table3(root: Path, num_seeds: int = DEFAULT_NUM_SEEDS) -> Path:
    base = root / "table3_figure1"
    files = sorted(base.glob("*/seed_*/forward_diagnostics/forward_diagnostics_summary_*.csv"))
    expected = len(DATASETS) * num_seeds
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Table III summaries ({num_seeds} seeds), found {len(files)} under {base}")
    _check_seed_dirs(files, base, num_seeds, "Table III")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in files:
        dataset = path.relative_to(base).parts[0]
        for row in read_csv(path):
            grouped[(dataset, row["schedule"])].append(row)

    output: list[dict] = []
    for (dataset, schedule), rows in sorted(grouped.items()):
        result = {"dataset": dataset, "schedule": schedule, "num_seeds": len(rows)}
        for metric in ("mean_log_snr_range", "max_log_snr_range", "max_stiffness_proxy"):
            vals = [v for row in rows if (v := number(row.get(metric))) is not None]
            mean, std = summarize(vals)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = std
        output.append(result)
    out = base / "table3_reproduced.csv"
    write_csv(out, output)
    return out


def aggregate_table4(root: Path, num_seeds: int = DEFAULT_NUM_SEEDS) -> Path:
    base = root / "table4"
    files = sorted(base.glob("*/*/seed_*/results_raw.csv"))
    expected = len(DATASETS) * 4 * num_seeds
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Table IV runs ({num_seeds} paired seeds), found {len(files)} under {base}")
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for path in files:
        dataset = path.relative_to(base).parts[0]
        for row in read_csv(path):
            if not is_stable(row):
                raise RuntimeError(f"Unstable Table IV row in {path}: {row}")
            key = (dataset, row["schedule"], row["method"], int(float(row["nfe"])))
            value = number(row.get("global_swd_mean"))
            if value is not None:
                grouped[key].append(value)

    output: list[dict] = []
    for (dataset, schedule, method, nfe), values in sorted(grouped.items()):
        if len(values) != num_seeds:
            raise RuntimeError(f"Expected {num_seeds} paired Table IV values for {(dataset, schedule, method, nfe)}, got {len(values)}")
        mean, std = summarize(values)
        output.append({
            "dataset": dataset,
            "schedule": schedule,
            "method": method,
            "nfe": nfe,
            "num_seeds": len(values),
            "global_swd_mean": mean,
            "global_swd_std": std,
        })
    out = base / "table4_reproduced.csv"
    write_csv(out, output)
    return out


DEFAULT_TABLE5_OPERATING_POINTS = "Cosine:em:2500,MAS-SNR:em:50,Isotropic-VP-Cosine:em:50,Isotropic-VP-Cosine:em:2500"
TABLE5_DATASETS = DATASETS


def parse_operating_points(spec: str) -> list[tuple[str, str, int]]:
    """Parse 'Schedule:method:steps,...' into (schedule, method, steps) triples.

    'Schedule:steps' is accepted as shorthand for method 'em'.
    """
    points: list[tuple[str, str, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) == 2:
            schedule, method, steps = parts[0], "em", parts[1]
        elif len(parts) == 3:
            schedule, method, steps = parts
        else:
            raise ValueError(f"Bad operating point {item!r}; expected Schedule:method:steps")
        points.append((schedule.strip(), method.strip(), int(float(steps))))
    if not points:
        raise ValueError("No operating points given")
    return points


def _table5_files(root: Path, num_seeds: int) -> tuple[Path, list[Path]]:
    base = root / "table5_figure2"
    files = sorted(base.glob("*/seed_*/results_summary.csv"))
    expected = num_seeds * len(TABLE5_DATASETS)
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Table V summaries ({num_seeds} seeds), found {len(files)} under {base}")
    _check_seed_dirs(files, base, num_seeds, "Table V")
    return base, files


def aggregate_table5(
    root: Path,
    operating_points: str = DEFAULT_TABLE5_OPERATING_POINTS,
    allow_unstable: bool = False,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> Path:
    """Aggregate Table V.

    Writes two files. ``table5_all_operating_points.csv`` summarises every
    (dataset, schedule, method, steps) present in the runs, with the number of
    seeds whose sampler stayed finite; it is what a step sweep for the coarsest
    stable grid is read from. ``table5_reproduced.csv`` holds the paper rows
    for ``operating_points`` and is strict: each point must be present in all
    seeds and (unless ``allow_unstable``) stable in all of them.
    """
    base, files = _table5_files(root, num_seeds)
    grouped: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for path in files:
        dataset = path.relative_to(base).parts[0]
        for row in read_csv(path):
            if row.get("method") is None or row.get("steps") in (None, ""):
                continue
            key = (dataset, row["schedule"], row["method"], int(float(row["steps"])))
            grouped[key].append(row)

    def summarise_group(rows: list[dict]) -> dict:
        stable_rows = [r for r in rows if is_stable(r)]
        result = {"num_seeds": len(rows), "num_seeds_stable": len(stable_rows)}
        for metric in METRICS:
            vals = [v for r in stable_rows if (v := number(r.get(metric))) is not None]
            if vals:
                mean, std = summarize(vals)
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
            else:
                result[f"{metric}_mean"] = ""
                result[f"{metric}_std"] = ""
        return result

    sweep_output: list[dict] = []
    for (dataset, schedule, method, steps), rows in sorted(grouped.items()):
        sweep_output.append({"dataset": dataset, "schedule": schedule, "method": method, "nfe": steps, **summarise_group(rows)})
    sweep_out = base / "table5_all_operating_points.csv"
    write_csv(sweep_out, sweep_output)

    points = parse_operating_points(operating_points)
    output: list[dict] = []
    problems: list[str] = []
    for dataset in TABLE5_DATASETS:
        for schedule, method, steps in points:
            rows = grouped.get((dataset, schedule, method, steps), [])
            if len(rows) != num_seeds:
                available = sorted({(k[1], k[2], k[3]) for k in grouped if k[0] == dataset})
                problems.append(
                    f"{dataset}: {schedule}/{method}/{steps} found in {len(rows)} of {num_seeds} seeds; "
                    f"available points: {available}"
                )
                continue
            summary = summarise_group(rows)
            if summary["num_seeds_stable"] != num_seeds:
                msg = f"{dataset}: {schedule}/{method}/{steps} unstable in {num_seeds - summary['num_seeds_stable']} seed(s)"
                if not allow_unstable:
                    problems.append(msg + " (pass --allow-unstable to report the stable seeds only)")
                    continue
            output.append({"dataset": dataset, "schedule": schedule, "method": method, "nfe": steps, **summary})
    if problems:
        raise RuntimeError("Table V operating points incomplete:\n  " + "\n  ".join(problems) + f"\nThe full sweep is in {sweep_out}")
    out = base / "table5_reproduced.csv"
    write_csv(out, output)
    return out


FIGURE2_LONG_COLUMNS = (
    "dataset", "seed", "schedule", "method", "steps", "mode", "lam_gamma", "band",
    "reference_power", "generated_power", "log10_power_ratio",
)


def aggregate_figure2(
    root: Path,
    operating_points: str = DEFAULT_TABLE5_OPERATING_POINTS,
    num_seeds: int = DEFAULT_NUM_SEEDS,
) -> Path:
    """Collect the Figure 2 spectral-power exports of every Table V run.

    Produces, under ``table5_figure2``:
      figure2_spectral_power_long.csv     every mode of every (seed, schedule, method, steps)
      figure2_spectral_power_by_mode.csv  mean/std across seeds by mode index, for the paper's
                                           operating points (each seed has its own graph, so
                                           eigenvalues are averaged by mode index as well)
      figure2_manifest.json               column descriptions and what the figure shows
    These three files are all that is needed to draw Figure 2 outside this package.
    """
    base = root / "table5_figure2"
    files = sorted(base.glob("*/seed_*/figure2_spectral_power_long.csv"))
    expected = num_seeds * len(TABLE5_DATASETS)
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Figure 2 exports ({num_seeds} seeds), found {len(files)} under {base}")
    _check_seed_dirs(files, base, num_seeds, "Figure 2")

    long_rows: list[dict] = []
    for path in files:
        dataset = path.relative_to(base).parts[0]
        seed = seed_from_path(path)
        for row in read_csv(path):
            out_row = {col: row.get(col, "") for col in FIGURE2_LONG_COLUMNS}
            out_row["dataset"] = row.get("dataset") or dataset
            out_row["seed"] = row.get("seed") or seed
            long_rows.append(out_row)
    long_out = base / "figure2_spectral_power_long.csv"
    write_csv(long_out, long_rows)

    points = set(parse_operating_points(operating_points))
    by_mode: dict[tuple[str, str, str, int, int], list[dict]] = defaultdict(list)
    for row in long_rows:
        key = (row["dataset"], row["schedule"], row["method"], int(float(row["steps"])), int(float(row["mode"])))
        if (key[1], key[2], key[3]) in points:
            by_mode[key].append(row)
    by_mode_rows: list[dict] = []
    for (dataset, schedule, method, steps, mode), rows in sorted(by_mode.items()):
        entry = {"dataset": dataset, "schedule": schedule, "method": method, "nfe": steps, "mode": mode, "num_seeds": len(rows), "band": rows[0]["band"]}
        for col in ("lam_gamma", "reference_power", "generated_power", "log10_power_ratio"):
            vals = [v for r in rows if (v := number(r.get(col))) is not None]
            mean, std = summarize(vals)
            entry[f"{col}_mean"] = mean
            entry[f"{col}_std"] = std
        by_mode_rows.append(entry)
    if by_mode_rows:
        write_csv(base / "figure2_spectral_power_by_mode.csv", by_mode_rows)

    present = sorted({(r["dataset"], r["schedule"], r["method"], int(float(r["steps"]))) for r in long_rows})
    manifest = {
        "files": {
            "figure2_spectral_power_long.csv": "one row per (dataset, seed, schedule, method, steps, mode)",
            "figure2_spectral_power_by_mode.csv": "mean and sample std across seeds by mode index, operating points only",
        },
        "what_figure_2_shows": (
            "Mode-wise spectral power E[x_tilde_k^2] against the shifted Laplacian eigenvalue lambda_gamma_k. "
            "One panel per dataset. Reference (held-out test) power as a black line; one line per schedule at "
            "its Table V operating point. Log-scale y axis. Shade band=='low' (lowest quarter of the spectrum) "
            "and band=='high' (highest quarter); those two bands enter the Table V band-ratio metric."
        ),
        "operating_points": [{"schedule": s, "method": m, "steps": n} for (s, m, n) in sorted(points)],
        "operating_points_present": [{"dataset": d, "schedule": s, "method": m, "steps": n} for (d, s, m, n) in present],
        "columns": {
            "dataset": "synthetic data family",
            "seed": "graph/data/training seed; each seed has its own graph, hence its own eigenvalues",
            "schedule": "forward schedule (Cosine, MAS-SNR, Isotropic-VP-Cosine, ...)",
            "method": "reverse integrator",
            "steps": "reverse steps = denoiser evaluations",
            "mode": "graph-frequency index k in ascending eigenvalue order",
            "lam_gamma": "lambda_k + gamma",
            "band": "low / middle / high",
            "reference_power": "mean squared GFT coefficient of the held-out test signals",
            "generated_power": "mean squared GFT coefficient of the generated signals",
            "log10_power_ratio": "log10(generated_power / reference_power)",
        },
        "seeds_per_dataset": num_seeds,
        "note": "Unstable sampler configurations have no rows; see table5_all_operating_points.csv.",
    }
    (base / "figure2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return long_out


TABLE9_NUM_CONFIGS = 7


def aggregate_table9(root: Path, num_seeds: int = DEFAULT_NUM_SEEDS, sampling_seeds: int = DEFAULT_NUM_SEEDS) -> Path:
    base = root / "table9"
    files = sorted(base.glob("*/seed_*/results_raw.csv"))
    expected = TABLE9_NUM_CONFIGS * num_seeds
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} Table IX model runs ({num_seeds} models per setting), found {len(files)} under {base}")
    by_config: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        config_id = path.relative_to(base).parts[0]
        rows = [r for r in read_csv(path) if is_stable(r) and r.get("method") == "dpmpp2m" and int(float(r["nfe"])) == 10]
        if len(rows) != sampling_seeds:
            raise RuntimeError(f"Expected {sampling_seeds} sampling seeds in {path}, got {len(rows)}")
        cfg = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        theory = read_csv(path.parent / "theory_stiffness_summary.csv")
        mass_theory = next(row for row in theory if row["schedule"] == "MAS-SNR")
        model_row = {
            "model_seed": int(cfg["training"]["model_seed"]),
            "gamma": float(cfg["gamma"]),
            "sigma": float(cfg["sigma"]),
            "condition_number": float(cfg["lambda_gamma_condition_number"]),
            "schedule_only_stiffness": float(mass_theory["measured_stiffness"]),
        }
        for metric in ("global_swd_mean", "full_spectrum_power_rel_l1"):
            model_row[metric] = statistics.fmean(float(row[metric]) for row in rows)
        by_config[config_id].append(model_row)

    output: list[dict] = []
    for config_id, models in sorted(by_config.items()):
        if len(models) != num_seeds:
            raise RuntimeError(f"Expected {num_seeds} independent models for {config_id}, got {len(models)}")
        row = {
            "config_id": config_id,
            "gamma": models[0]["gamma"],
            "sigma": models[0]["sigma"],
            "condition_number": models[0]["condition_number"],
            "schedule_only_stiffness": models[0]["schedule_only_stiffness"],
            "num_models": num_seeds,
            "sampling_seeds_per_model": sampling_seeds,
        }
        for metric in ("global_swd_mean", "full_spectrum_power_rel_l1"):
            mean, std = summarize([model[metric] for model in models])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        output.append(row)
    out = base / "table9_reproduced.csv"
    write_csv(out, output)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, choices=("table3", "table4", "table5", "table9", "figure2"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--operating-points",
        default=DEFAULT_TABLE5_OPERATING_POINTS,
        help="Table V / Figure 2 rows as 'Schedule:method:steps,...' (Schedule:steps means method em).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Seeds per dataset (Tables III, V, Figure 2), paired seeds (Table IV) or models per setting (Table IX). Default 5.",
    )
    parser.add_argument(
        "--sampling-seeds-per-model",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Table IX: sampling seeds per trained model. Default 5.",
    )
    parser.add_argument(
        "--allow-unstable",
        action="store_true",
        help="Table V: report an operating point from its stable seeds instead of failing when a seed diverged.",
    )
    args = parser.parse_args()
    root = args.results_root.resolve()
    if args.table == "table5":
        output = aggregate_table5(root, operating_points=args.operating_points, allow_unstable=args.allow_unstable,
                                  num_seeds=args.num_seeds)
    elif args.table == "figure2":
        output = aggregate_figure2(root, operating_points=args.operating_points, num_seeds=args.num_seeds)
    elif args.table == "table9":
        output = aggregate_table9(root, num_seeds=args.num_seeds, sampling_seeds=args.sampling_seeds_per_model)
    elif args.table == "table3":
        output = aggregate_table3(root, num_seeds=args.num_seeds)
    else:
        output = aggregate_table4(root, num_seeds=args.num_seeds)
    print(output)


if __name__ == "__main__":
    main()
