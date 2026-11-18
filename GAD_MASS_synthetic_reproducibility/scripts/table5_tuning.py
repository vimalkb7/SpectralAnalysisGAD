from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

PROTOCOL_VERSION = "table5-tuning-v1"
DATASETS = {
    "stationary_gaussian": {"sigma": 1.0, "gamma_default": 1e-3,
                            "data_args": ["--stationary_spectrum", "power_law", "--power_law_p", "1.0"]},
    "block_covariance": {"sigma": 0.75, "gamma_default": 3e-3,
                         "data_args": ["--num_blocks", "4", "--rho", "0.8"]},
}
DATASET_ORDER = ["stationary_gaussian", "block_covariance"]
SEEDS = [0, 1, 2, 3, 4]
EM_STEPS = [50, 100, 250, 500, 1000, 2500]
SELECTION_BUDGETS = [50, 2500]
DELTA = 1e-3
T0 = 1e-3
METRICS = ["global_swd_mean", "cov_rel_err", "full_spectrum_power_rel_l1", "band_ratio_l1", "smoothness_w1"]
METRIC_LABELS = {"global_swd_mean": "SWD", "cov_rel_err": "Cov. Err.", "full_spectrum_power_rel_l1": "Spectrum",
                 "band_ratio_l1": "Band Ratio", "smoothness_w1": "Smoothness"}
METHOD_LABEL = {"MAS-SNR": "MASS", "Cosine": "Cosine", "Isotropic-VP-Cosine": "Isotropic VP"}
BASELINES = ["Cosine", "Isotropic-VP-Cosine"]

# Everything the three methods share: data, network, training, sampler, metrics.
COMMON_ARGS = [
    "--run_mode", "train_and_sample", "--dtype", "float32", "--n", "128", "--k", "8", "--T", "1.0",
    "--num_train", "10000", "--num_val", "2000", "--num_test", "2000",
    "--hidden_dim", "256", "--num_layers", "8", "--time_embed_dim", "256", "--cheb_order", "8",
    "--num_epochs", "200", "--batch_size", "128", "--lr", "3e-4", "--min_lr", "1e-6",
    "--weight_decay", "0", "--grad_clip_norm", "1", "--train_t_min", "1e-2",
    "--val_method", "ddim", "--val_steps", "200", "--val_num_samples", "512", "--val_every_epochs", "10",
    "--methods", "em", "--reverse_steps", ",".join(str(s) for s in EM_STEPS), "--t0", f"{T0:g}",
    "--sampler_init_mode", "algorithm2", "--metric_num_projections", "128", "--metric_num_seeds", "5",
    "--band_low_frac", "0.25", "--band_high_frac", "0.25",
]


def grid() -> list[dict]:
    """The pre-registered grid, in a fixed order (the order breaks final ties)."""
    out = []
    for gamma in (3e-4, 1e-3, 3e-3, 1e-2):
        for d0 in (1e-3, 1e-2):
            out.append({"id": f"mass_gamma{gamma:g}_delta0{d0:g}", "method": "MAS-SNR",
                        "params": {"gamma": gamma, "delta0": d0, "delta_T": DELTA}})
    for gamma in (1e-3, 3e-3, 1e-2):
        for eps in (1e-3, 1e-2, 1e-1, 0.5):
            cbar = math.log(1.0 / DELTA) / (2.0 * gamma)
            out.append({"id": f"cosine_gamma{gamma:g}_eps{eps:g}", "method": "Cosine",
                        "params": {"gamma": gamma, "cosine_s": 0.008, "cosine_terminal_epsilon": eps, "cbar_T": cbar}})
    for s in (0.002, 0.008, 0.02, 0.05):
        for eps in (1e-4, 1e-3, 1e-2):
            out.append({"id": f"iso_s{s:g}_eps{eps:g}", "method": "Isotropic-VP-Cosine",
                        "params": {"iso_cosine_s": s, "iso_cosine_terminal_epsilon": eps}})
    return out


GRID = grid()
GRID_BY_ID = {c["id"]: c for c in GRID}


def protocol_dict() -> dict:
    return {
        "version": PROTOCOL_VERSION, "datasets": DATASETS, "seeds": SEEDS, "em_steps": EM_STEPS,
        "selection_budgets": SELECTION_BUDGETS, "delta": DELTA, "t0": T0, "metrics": METRICS,
        "grid": GRID, "common_args": COMMON_ARGS,
        "selection_rule": "per dataset, method and budget: eligible = delta-complete and finite in all seeds; "
                          "mean over seeds per metric; rank per metric among eligible (average ties); lowest mean rank; "
                          "ties by mean SWD, then grid order",
        "claim_rule": "MASS better on a metric vs a baseline if the mean paired difference favours MASS and >= 4 of 5 seeds do",
    }


def protocol_sha() -> str:
    return hashlib.sha256(json.dumps(protocol_dict(), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ tasks
def tune_tasks() -> list[dict]:
    tasks = []
    for ds in DATASET_ORDER:
        for cfg in GRID:
            for seed in SEEDS:
                tasks.append({"phase": "tune", "dataset": ds, "config_id": cfg["id"], "method": cfg["method"], "seed": seed})
    return tasks


def load_selection(root: Path) -> dict:
    p = root / "selection" / "selection.json"
    if not p.is_file():
        raise SystemExit(f"No selection yet: {p} (run the tuning phase and the selection first)")
    sel = json.loads(p.read_text(encoding="utf-8"))
    if sel.get("protocol_sha256") != protocol_sha():
        raise SystemExit("selection.json was made under a different protocol; refusing to use it")
    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != sel.get("selection_sha256"):
        raise SystemExit("selection.json was modified after it was frozen; refusing to use it")
    return sel


def test_tasks(root: Path) -> list[dict]:
    sel = load_selection(root)
    pairs = sorted({(e["dataset"], e["config_id"]) for e in sel["entries"] if e.get("config_id")},
                   key=lambda x: (DATASET_ORDER.index(x[0]), [c["id"] for c in GRID].index(x[1])))
    tasks = []
    for ds, cid in pairs:
        for seed in SEEDS:
            tasks.append({"phase": "test", "dataset": ds, "config_id": cid, "method": GRID_BY_ID[cid]["method"], "seed": seed})
    return tasks


def tasks_for(phase: str, root: Path) -> list[dict]:
    return tune_tasks() if phase == "tune" else test_tasks(root)


def run_dir(root: Path, task: dict) -> Path:
    return root / task["phase"] / task["dataset"] / task["config_id"] / f"seed_{task['seed']}"


# ------------------------------------------------------------------ plan (needs torch and the package)
def build_plan(task: dict, code_dir: Path) -> dict:
    """Driver arguments plus the delta-completeness and EM-stability checks of one task."""
    sys.path.insert(0, str(code_dir))
    import numpy as np
    import torch
    from datasets import make_synthetic_dataset_numpy
    from graphs import eigh_sym, rgg_knn_graph, shifted_laplacian
    from schedules import build_schedules, channel_params_for_times_torch, mass_endpoints_from_tolerances

    ds, seed, cfg = task["dataset"], int(task["seed"]), GRID_BY_ID[task["config_id"]]
    method, prm, dcfg = cfg["method"], dict(cfg["params"]), DATASETS[task["dataset"]]
    sigma = dcfg["sigma"]
    gamma = prm.get("gamma", dcfg["gamma_default"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, _, L = rgg_knn_graph(n=128, k=8, seed=seed)                      # as in the driver
    lam, V = eigh_sym(shifted_laplacian(L, gamma=gamma))
    X, _ = make_synthetic_dataset_numpy(family=ds, V=V, lam_gamma=lam, N=14000, seed=seed,
                                        spectrum="power_law", power_law_p=1.0, k_cut=16, strength=3.0,
                                        num_blocks=4, rho=0.8, num_bursts=2, burst_amp=3.0)
    s_k = np.mean((X[:10000] @ V) ** 2, axis=0)

    args = ["--seed", str(seed), "--dataset_family", ds, *dcfg["data_args"], "--sigma", f"{sigma:g}",
            "--gamma", f"{gamma:g}", "--schedulers", method, "--diagnostics_schedulers", method]
    r_max = r_min = None
    cosine_cbar, cosine_s, cosine_eps = 300.0, 0.008, 1e-3
    iso_s, iso_eps = 0.02, 0.01
    if method == "MAS-SNR":
        r_max, r_min = mass_endpoints_from_tolerances(lam, s_k, sigma, start_a2_error=prm["delta0"], terminal_a2=DELTA)
        args += ["--r_max", repr(float(r_max)), "--r_min", repr(float(r_min))]
    elif method == "Cosine":
        cosine_cbar, cosine_s, cosine_eps = prm["cbar_T"], prm["cosine_s"], prm["cosine_terminal_epsilon"]
        args += ["--cosine_cbar_T", repr(float(cosine_cbar)), "--cosine_s", f"{cosine_s:g}",
                 "--cosine_terminal_epsilon", f"{cosine_eps:g}"]
    else:
        iso_s, iso_eps = prm["iso_cosine_s"], prm["iso_cosine_terminal_epsilon"]
        args += ["--iso_cosine_s", f"{iso_s:g}", "--iso_cosine_terminal_epsilon", f"{iso_eps:g}"]
    args += ["--evaluation_split", "val" if task["phase"] == "tune" else "test"]

    sched = build_schedules(lam_gamma=lam, s_k=s_k, sigma=sigma, T=1.0, fcps_c0=2300.0, fcps_cmin=0.1, fcps_alpha=4.0,
                            r_max=float(r_max if r_max is not None else 1e4),
                            r_min=float(r_min if r_min is not None else 1e-3), match_terminal_energy=False,
                            cosine_cbar_T=float(cosine_cbar), cosine_s=float(cosine_s),
                            cosine_terminal_epsilon=float(cosine_eps),
                            iso_cosine_s=float(iso_s), iso_cosine_terminal_epsilon=float(iso_eps))
    obj = sched[method]
    lam_t = torch.tensor(lam, dtype=torch.float64)
    s_t = torch.tensor(s_k, dtype=torch.float64)
    endT = channel_params_for_times_torch(method, obj, lam_t, s_t, sigma, torch.tensor([1.0], dtype=torch.float64))
    start = channel_params_for_times_torch(method, obj, lam_t, s_t, sigma, torch.tensor([T0], dtype=torch.float64))
    a2_T_max = float(torch.as_tensor(endT["a2"]).max())
    a2_t0_min = float(torch.as_tensor(start["a2"]).min())
    eligible = a2_T_max <= DELTA * (1 + 1e-6)

    grids = {}
    for steps in EM_STEPS:
        g = torch.linspace(1.0, T0, steps + 1, dtype=torch.float64)
        t, dt = g[:-1], (g[:-1] - g[1:])[:, None]
        p = channel_params_for_times_torch(method, obj, lam_t, s_t, sigma, t)
        zeta = p["diffusion_sq"] / torch.clamp(p["nu"], min=1e-300) - p["drift_rate"]
        zdt = zeta * dt
        logf = torch.log10(torch.clamp(torch.abs(1.0 - zdt), min=1e-300))
        prefix = torch.cumprod((p["a2"] < 1e-12).to(torch.float64), dim=0)
        peak = float(torch.clamp(torch.cumsum(logf * prefix, dim=0).max(), min=0.0))
        grids[str(steps)] = {"max_zeta_dt": float(zdt.max()), "em_stable_schedule_only": bool(float(zdt.max()) <= 2.0),
                             "float32_overflow_certain": bool(peak >= 38.0)}
    diverges_at_all_budgets = all(grids[str(b)]["float32_overflow_certain"] for b in SELECTION_BUDGETS)
    if not eligible:
        action = "skip_ineligible"
    elif diverges_at_all_budgets:
        action = "skip_diverged"
    else:
        action = "run"
    return {
        "protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_sha(), "task": task, "config": cfg,
        "driver_args": COMMON_ARGS + args, "action": action,
        "delta_complete": eligible, "max_a2_at_T": a2_T_max, "min_a2_at_t0": a2_t0_min,
        "mass_r_max": r_max, "mass_r_min": r_min, "cosine_cbar_T": cosine_cbar if method == "Cosine" else None,
        "lambda_gamma_min": float(lam[0]), "lambda_gamma_max": float(lam[-1]), "em_grids": grids,
    }


# ------------------------------------------------------------------ small IO helpers
def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def write_csv(p: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def is_stable(r: dict) -> bool:
    return str(r.get("stable", "true")).strip().lower() in {"true", "1", "yes"}


def mean_std(v: list[float]) -> tuple[float, float]:
    return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def rows_at(run: Path, method: str, steps: int, split: str) -> dict | None:
    """The result row of one run at (em, steps), or None when missing."""
    p = run / "results_summary.csv"
    if not p.is_file():
        return None
    for r in read_csv(p):
        if r.get("schedule") == method and r.get("method") == "em" and int(float(r.get("steps", -1))) == steps:
            if r.get("evaluation_split", split) != split:
                raise SystemExit(f"{p}: row scored on split {r.get('evaluation_split')!r}, expected {split!r}")
            return r
    return None


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# ------------------------------------------------------------------ selection (validation only)
def select(root: Path) -> Path:
    missing = [t for t in tune_tasks() if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} tuning runs are not finished; first missing: {missing[0]}")
    all_rows, entries = [], []
    for ds in DATASET_ORDER:
        for method in ("MAS-SNR", "Cosine", "Isotropic-VP-Cosine"):
            cfgs = [c for c in GRID if c["method"] == method]
            for b in SELECTION_BUDGETS:
                cand = []
                for c in cfgs:
                    plan = json.loads((run_dir(root, {"phase": "tune", "dataset": ds, "config_id": c["id"], "seed": 0})
                                       / "plan.json").read_text(encoding="utf-8"))
                    per_seed = [rows_at(run_dir(root, {"phase": "tune", "dataset": ds, "config_id": c["id"], "seed": s}),
                                        method, b, "val") for s in SEEDS]
                    stable = [r is not None and is_stable(r) and all(num(r.get(m)) is not None for m in METRICS)
                              for r in per_seed]
                    means = {m: (statistics.fmean(num(r[m]) for r in per_seed) if all(stable) else None) for m in METRICS}
                    eligible = bool(plan["delta_complete"]) and all(stable)
                    row = {"dataset": ds, "method": method, "budget": b, "config_id": c["id"],
                           "delta_complete": plan["delta_complete"], "seeds_stable": sum(stable), "eligible": eligible,
                           **{f"val_{m}": means[m] for m in METRICS}}
                    all_rows.append(row)
                    if eligible:
                        cand.append(row)
                if not cand:
                    entries.append({"dataset": ds, "method": method, "budget": b, "config_id": None,
                                    "reason": "no delta-complete configuration stayed finite in all seeds",
                                    "num_eligible": 0, "num_configs": len(cfgs)})
                    continue
                ranks = {m: average_ranks([r[f"val_{m}"] for r in cand]) for m in METRICS}
                for i, r in enumerate(cand):
                    r["mean_rank"] = statistics.fmean(ranks[m][i] for m in METRICS)
                order = [c["id"] for c in GRID]
                best = min(cand, key=lambda r: (r["mean_rank"], r["val_global_swd_mean"], order.index(r["config_id"])))
                entries.append({"dataset": ds, "method": method, "budget": b, "config_id": best["config_id"],
                                "params": GRID_BY_ID[best["config_id"]]["params"], "mean_rank": best["mean_rank"],
                                "val_means": {m: best[f"val_{m}"] for m in METRICS},
                                "num_eligible": len(cand), "num_configs": len(cfgs)})
    out_dir = root / "selection"
    write_csv(out_dir / "tuning_summary.csv", all_rows)
    body = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_sha(), "entries": entries}
    body["selection_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    (out_dir / "selection.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    lines = [f"# Table V tuning: selected configurations ({PROTOCOL_VERSION})", "",
             "Selected on the validation split only. Frozen; sha256 " + body["selection_sha256"], "",
             "| Dataset | Method | Budget | Selected configuration | Eligible / grid | Mean rank |", "|---|---|---|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['dataset']} | {METHOD_LABEL[e['method']]} | {e['budget']} | {e['config_id'] or 'none: ' + e['reason']} "
                     f"| {e['num_eligible']} / {e['num_configs']} | {e.get('mean_rank', '')} |")
    (out_dir / "SELECTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "selection.json"


# ------------------------------------------------------------------ final table (test split, once)
def fmt(m: float | None, s: float | None) -> str:
    if m is None:
        return "–"
    if abs(m) >= 10:
        return f"{m:.2f} ± {s:.2f}"
    return f"{m:.4f} ± {s:.4f}"


def final(root: Path) -> Path:
    sel = load_selection(root)
    missing = [t for t in test_tasks(root) if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} test runs are not finished; first missing: {missing[0]}")
    out = root / "final"
    per_seed = {}   # (ds, method, b) -> {seed: row}
    table = []
    for e in sel["entries"]:
        key = (e["dataset"], e["method"], e["budget"])
        if not e.get("config_id"):
            table.append({"dataset": e["dataset"], "method": e["method"], "budget": e["budget"], "config_id": "",
                          "note": e["reason"]})
            continue
        vals = {}
        for s in SEEDS:
            r = rows_at(run_dir(root, {"phase": "test", "dataset": e["dataset"], "config_id": e["config_id"], "seed": s}),
                        e["method"], e["budget"], "test")
            vals[s] = r
        per_seed[key] = vals
        stable = [s for s, r in vals.items() if r is not None and is_stable(r)]
        row = {"dataset": e["dataset"], "method": e["method"], "budget": e["budget"], "config_id": e["config_id"],
               "seeds_stable": len(stable)}
        for m in METRICS:
            v = [num(vals[s][m]) for s in stable if num(vals[s].get(m)) is not None]
            mu, sd = mean_std(v) if v else (None, None)
            row[f"{m}_mean"], row[f"{m}_std"] = mu, sd
        if len(stable) < len(SEEDS):
            row["note"] = f"sampler diverged on the test split in {len(SEEDS) - len(stable)} seed(s)"
        table.append(row)
    write_csv(out / "table5_tuned_final.csv", table)

    paired = []
    comparisons = [(b, b) for b in SELECTION_BUDGETS] + [(50, 2500)]
    for ds in DATASET_ORDER:
        for bm, bb in comparisons:
            mk = (ds, "MAS-SNR", bm)
            for base in BASELINES:
                bk = (ds, base, bb)
                if mk not in per_seed or bk not in per_seed:
                    continue
                for m in METRICS:
                    d = []
                    for s in SEEDS:
                        a, c = per_seed[mk][s], per_seed[bk][s]
                        if a is None or c is None or not (is_stable(a) and is_stable(c)):
                            continue
                        x, y = num(a.get(m)), num(c.get(m))
                        if x is not None and y is not None:
                            d.append(x - y)
                    if not d:
                        continue
                    mu, sd = mean_std(d)
                    wins = sum(v < 0 for v in d)
                    paired.append({"dataset": ds, "mass_budget": bm, "baseline": base, "baseline_budget": bb, "metric": m,
                                   "paired_seeds": len(d), "mean_diff_mass_minus_baseline": mu, "std_diff": sd,
                                   "seeds_mass_better": wins,
                                   "mass_better": bool(mu < 0 and wins >= 4 and len(d) == len(SEEDS))})
    write_csv(out / "table5_tuned_paired.csv", paired)

    # Markdown rendering of the main table.
    lines = ["| Dataset | Schedule | Steps | " + " | ".join(METRIC_LABELS[m] + " ↓" for m in METRICS) + " |",
             "|---|---|---|" + "---|" * len(METRICS)]
    for ds in DATASET_ORDER:
        for b in SELECTION_BUDGETS:
            rows = [r for r in table if r["dataset"] == ds and r["budget"] == b]
            best = {}
            for m in METRICS:
                v = [(r.get(f"{m}_mean"), r["method"]) for r in rows if r.get(f"{m}_mean") is not None]
                best[m] = min(v)[1] if v else None
            for r in rows:
                cells = []
                for m in METRICS:
                    c = fmt(r.get(f"{m}_mean"), r.get(f"{m}_std")) if r.get(f"{m}_mean") is not None else "–"
                    cells.append(f"**{c}**" if best[m] == r["method"] and c != "–" else c)
                note = "" if not r.get("note") else f" ({r['note']})"
                lines.append(f"| {ds} | {METHOD_LABEL[r['method']]}{note} | {b} | " + " | ".join(cells) + " |")
    lines += ["", "Paired comparison (MASS - baseline, per seed; 'better' = mean favours MASS and >= 4/5 seeds):", "",
              "| Dataset | MASS steps | Baseline | Baseline steps | Metric | Mean diff ± std | Seeds MASS better | Better |",
              "|---|---|---|---|---|---|---|---|"]
    for p in paired:
        lines.append(f"| {p['dataset']} | {p['mass_budget']} | {METHOD_LABEL[p['baseline']]} | {p['baseline_budget']} | "
                     f"{METRIC_LABELS[p['metric']]} | {p['mean_diff_mass_minus_baseline']:+.4g} ± {p['std_diff']:.3g} | "
                     f"{p['seeds_mass_better']}/{p['paired_seeds']} | {'yes' if p['mass_better'] else 'no'} |")
    (out / "TABLE5_TUNED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Full EM sweep of the selected configurations (test split) and Figure 2 data.
    sweep, fig = [], []
    seen = set()
    for e in sel["entries"]:
        if not e.get("config_id") or (e["dataset"], e["config_id"]) in seen:
            continue
        seen.add((e["dataset"], e["config_id"]))
        for steps in EM_STEPS:
            vals = [rows_at(run_dir(root, {"phase": "test", "dataset": e["dataset"], "config_id": e["config_id"], "seed": s}),
                            e["method"], steps, "test") for s in SEEDS]
            st = [r for r in vals if r is not None and is_stable(r)]
            row = {"dataset": e["dataset"], "method": e["method"], "config_id": e["config_id"], "steps": steps,
                   "seeds_stable": len(st)}
            for m in METRICS:
                v = [num(r[m]) for r in st if num(r.get(m)) is not None]
                row[f"{m}_mean"], row[f"{m}_std"] = mean_std(v) if v else ("", "")
            sweep.append(row)
    write_csv(out / "table5_tuned_em_sweep.csv", sweep)
    for e in sel["entries"]:
        if not e.get("config_id"):
            continue
        for s in SEEDS:
            p = run_dir(root, {"phase": "test", "dataset": e["dataset"], "config_id": e["config_id"], "seed": s}) \
                / "figure2_spectral_power_long.csv"
            if p.is_file():
                for r in read_csv(p):
                    if int(float(r["steps"])) == e["budget"]:
                        fig.append({**r, "config_id": e["config_id"], "selected_for_budget": e["budget"]})
    if fig:
        write_csv(out / "figure2_tuned_spectral_power_long.csv", fig)

    bundle = root / "table5_tuned_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [out / "TABLE5_TUNED.md", out / "table5_tuned_final.csv", out / "table5_tuned_paired.csv",
                  out / "table5_tuned_em_sweep.csv", out / "figure2_tuned_spectral_power_long.csv",
                  root / "selection" / "selection.json", root / "selection" / "SELECTION.md",
                  root / "selection" / "tuning_summary.csv", root / "PROTOCOL.md"]:
            if p.is_file():
                z.write(p, arcname=f"table5_tuned/{p.name}")
    return bundle


# ------------------------------------------------------------------ protocol document
def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_protocol(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pj = root / "protocol.json"
    sha = protocol_sha()
    if pj.is_file():
        old = json.loads(pj.read_text(encoding="utf-8")).get("protocol_sha256")
        if old != sha:
            raise SystemExit(f"{pj} holds a different protocol ({old}); use a new results folder")
        return
    _atomic_write(pj, json.dumps({"protocol_sha256": sha, **protocol_dict()}, indent=2))
    counts = defaultdict(int)
    for c in GRID:
        counts[c["method"]] += 1
    lines = [f"# Table V tuning protocol ({PROTOCOL_VERSION})", "", f"sha256: {sha}", "",
             __doc__.split("Subcommands")[0].strip(), "",
             "## Grid", "", "| Configuration | Method | Parameters |", "|---|---|---|"]
    for c in GRID:
        lines.append(f"| {c['id']} | {METHOD_LABEL[c['method']]} | {json.dumps(c['params'])} |")
    lines += ["", "Configurations per method: " + ", ".join(f"{METHOD_LABEL[k]} {v}" for k, v in counts.items()),
              "", "Shared driver arguments: `" + " ".join(COMMON_ARGS) + "`"]
    _atomic_write(root / "PROTOCOL.md", "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("protocol", "select", "final"):
        sp = sub.add_parser(name)
        sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("count")
    sp.add_argument("--phase", choices=("tune", "test"), required=True)
    sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("status")
    sp.add_argument("--phase", choices=("tune", "test"), required=True)
    sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("describe")
    sp.add_argument("--phase", choices=("tune", "test"), required=True)
    sp.add_argument("--index", type=int, required=True)
    sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("plan")
    sp.add_argument("--phase", choices=("tune", "test"), required=True)
    sp.add_argument("--index", type=int, required=True)
    sp.add_argument("--root", type=Path, required=True)
    sp.add_argument("--code-dir", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("record-skip")
    sp.add_argument("--plan", type=Path, required=True)
    sp.add_argument("--run-dir", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "protocol":
        write_protocol(a.root)
        print(a.root / "PROTOCOL.md")
    elif a.cmd == "count":
        print(len(tasks_for(a.phase, a.root)))
    elif a.cmd == "status":
        t = tasks_for(a.phase, a.root)
        missing = [i for i, task in enumerate(t) if not (run_dir(a.root, task) / "DONE.json").is_file()]
        print(f"{a.phase}: {len(t) - len(missing)} of {len(t)} done")
        if missing:
            spans, start = [], missing[0]
            for prev, cur in zip(missing, missing[1:] + [None]):
                if cur != prev + 1:
                    spans.append(f"{start}" if start == prev else f"{start}-{prev}")
                    start = cur
            print("missing: " + ",".join(spans))
        raise SystemExit(0 if not missing else 3)
    elif a.cmd == "describe":
        t = tasks_for(a.phase, a.root)
        if not 0 <= a.index < len(t):
            raise SystemExit(f"index {a.index} out of range 0-{len(t) - 1} for phase {a.phase}")
        task = t[a.index]
        print(json.dumps({**task, "run_dir": str(run_dir(a.root, task))}))
    elif a.cmd == "plan":
        t = tasks_for(a.phase, a.root)
        if not 0 <= a.index < len(t):
            raise SystemExit(f"index {a.index} out of range 0-{len(t) - 1} for phase {a.phase}")
        plan = build_plan(t[a.index], a.code_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        g = plan["em_grids"]
        print(f"plan: {plan['task']['dataset']} {plan['config']['id']} seed {plan['task']['seed']} -> {plan['action']}")
        print(f"  delta-complete: {plan['delta_complete']} (max a^2 at T = {plan['max_a2_at_T']:.3g}; "
              f"min a^2 at t0 = {plan['min_a2_at_t0']:.6f})")
        if plan["mass_r_max"] is not None:
            print(f"  MASS endpoints from eq. (41): r_max = {plan['mass_r_max']:.4g}, r_min = {plan['mass_r_min']:.4g}")
        if plan["cosine_cbar_T"] is not None:
            print(f"  Cosine cbar_T = {plan['cosine_cbar_T']:.4g}")
        print("  EM steps:  " + "  ".join(f"{k}: {'ok' if v['em_stable_schedule_only'] else ('overflow' if v['float32_overflow_certain'] else 'unstable')}"
                                         f" ({v['max_zeta_dt']:.3g})" for k, v in g.items()))
    elif a.cmd == "record-skip":
        plan = json.loads(a.plan.read_text(encoding="utf-8"))
        task = plan["task"]
        split = "val" if task["phase"] == "tune" else "test"
        reason = {"skip_ineligible": "not delta-complete (protocol)",
                  "skip_diverged": "float32 overflow certain at every selection budget (schedule-only, denoiser-independent)"}[plan["action"]]
        rows = [{"dataset": task["dataset"], "seed": task["seed"], "schedule": task["method"], "method": "em",
                 "steps": s, "nfe": s, "evaluation_split": split, "stable": False, "failure_reason": f"skipped: {reason}"}
                for s in EM_STEPS]
        write_csv(a.run_dir / "results_summary.csv", rows)
        print(f"recorded skip ({reason}) in {a.run_dir / 'results_summary.csv'}")
    elif a.cmd == "select":
        print(select(a.root))
    elif a.cmd == "final":
        print(final(a.root))


if __name__ == "__main__":
    main()
