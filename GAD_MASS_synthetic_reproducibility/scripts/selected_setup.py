from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import table5_tuning as T  # noqa: E402

DATASETS = T.DATASET_ORDER
METHODS = ["MAS-SNR", "Cosine", "Isotropic-VP-Cosine"]
LABEL = {"MAS-SNR": "MASS", "Cosine": "Cosine", "Isotropic-VP-Cosine": "Isotropic VP", "FCPS": "FCPS"}
FILE_NAME = {"MAS-SNR": "mass.json", "Cosine": "cosine.json", "Isotropic-VP-Cosine": "isotropic_vp_cosine.json"}
DELTA = T.DELTA
SEEDS = T.SEEDS

TABLE3_SCHEDULES = ["FCPS", "Cosine", "MAS-SNR"]
TABLE3_SIGMA = {"stationary_gaussian": 1.0, "block_covariance": 0.75}
TABLE3_COMMON = [
    "--run_mode", "forward_diagnostics", "--dtype", "float64", "--n", "128", "--k", "8", "--T", "1.0",
    "--num_train", "10000", "--num_val", "2000", "--num_test", "2000",
    "--forward_diag_num_times", "2048", "--forward_diag_num_extreme_modes", "4",
    "--forward_diag_num_samples", "2000", "--forward_diag_t_min", "1e-4",
    "--forward_diag_pmin", "1e-10", "--stiffness_h_proxy", "0.0",
]
TABLE4_SCHEDULES = ["Isotropic-VP-Cosine", "FCPS", "Cosine", "MAS-SNR"]
TABLE4_SAFE = {"Isotropic-VP-Cosine": "isotropic_vp_cosine", "FCPS": "fcps", "Cosine": "cosine", "MAS-SNR": "mass"}
TABLE4_DATASETS = {"stationary_gaussian": "broadband_wss", "block_covariance": "block_covariance"}
TABLE4_COMMON = [
    "--evaluation_split", "test", "--study_phase", "locked_test", "--sigma", "1.0",
    "--methods", "ddim,dpmpp2m", "--ddim_steps", "25", "--dpm_steps", "10,25",
    "--num_eval_samples", "1024", "--num_epochs", "40", "--batch_size", "128",
    "--hidden_dim", "64", "--num_layers", "4", "--time_embed_dim", "64", "--cheb_order", "3",
    "--lr", "3e-4", "--min_lr", "1e-6", "--train_t_min", "1e-2", "--val_every_epochs", "2",
    "--val_num_samples", "1024", "--t0", "1e-3", "--sampler_init_mode", "algorithm2",
    "--dtype", "float32", "--skip_forward_diagnostics", "--plot_export_steps", "10,25",
]


# ------------------------------------------------------------------ helpers
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


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


# ------------------------------------------------------------------ export
def export(sweep_root: Path, out: Path) -> Path:
    sel = T.load_selection(sweep_root)                       # checks protocol and selection hashes
    final_md = sweep_root / "final" / "TABLE5_TUNED.md"
    if not final_md.is_file():
        raise SystemExit(f"The sweep has no final table yet ({final_md}); wait for the test phase")
    final_rows = read_csv(sweep_root / "final" / "table5_tuned_final.csv")
    paired = read_csv(sweep_root / "final" / "table5_tuned_paired.csv") if (sweep_root / "final" / "table5_tuned_paired.csv").is_file() else []
    out.mkdir(parents=True, exist_ok=True)
    entries = {(e["dataset"], e["method"], int(e["budget"])): e for e in sel["entries"]}

    def final_row(ds, method, b):
        for r in final_rows:
            if r["dataset"] == ds and r["method"] == method and int(float(r["budget"])) == b:
                return r
        return None

    uniform = {"protocol_sha256": sel["protocol_sha256"], "selection_sha256": sel["selection_sha256"],
               "rule": "per dataset and method: the configuration selected at 50 steps, else at 2500 steps",
               "delta": DELTA, "datasets": {}}
    per_method = {m: {"method": m, "label": LABEL[m], "protocol_sha256": sel["protocol_sha256"],
                      "selection_sha256": sel["selection_sha256"], "rule": uniform["rule"], "datasets": {}}
                  for m in METHODS}
    for ds in DATASETS:
        uniform["datasets"][ds] = {}
        for m in METHODS:
            budgets = {}
            for b in T.SELECTION_BUDGETS:
                e = entries.get((ds, m, b), {})
                fr = final_row(ds, m, b) if e.get("config_id") else None
                budgets[str(b)] = {
                    "config_id": e.get("config_id"), "params": e.get("params"), "reason": e.get("reason"),
                    "validation_means": e.get("val_means"), "mean_rank": e.get("mean_rank"),
                    "num_eligible": e.get("num_eligible"), "num_configs": e.get("num_configs"),
                    "test_metrics": ({mt: {"mean": num(fr.get(f"{mt}_mean")), "std": num(fr.get(f"{mt}_std"))}
                                      for mt in T.METRICS} if fr else None),
                    "test_seeds_stable": (int(fr["seeds_stable"]) if fr and fr.get("seeds_stable") not in (None, "") else None),
                }
            primary_b = next((b for b in T.SELECTION_BUDGETS if budgets[str(b)]["config_id"]), None)
            primary = budgets[str(primary_b)] if primary_b else None
            endpoints = {}
            if m == "MAS-SNR" and primary:
                for s in SEEDS:
                    plan = T.run_dir(sweep_root, {"phase": "test", "dataset": ds, "config_id": primary["config_id"],
                                                  "seed": s}) / "plan.json"
                    if plan.is_file():
                        pl = json.loads(plan.read_text(encoding="utf-8"))
                        endpoints[str(s)] = {"r_max": pl.get("mass_r_max"), "r_min": pl.get("mass_r_min")}
            per_method[m]["datasets"][ds] = {"primary_budget": primary_b,
                                             "primary_config_id": primary["config_id"] if primary else None,
                                             "primary_params": primary["params"] if primary else None,
                                             "by_budget": budgets,
                                             **({"mass_endpoints_per_seed_table5": endpoints} if endpoints else {})}
            uniform["datasets"][ds][m] = ({"config_id": primary["config_id"], "selected_at_budget": primary_b,
                                           **primary["params"]} if primary else None)
        cos = uniform["datasets"][ds]["Cosine"]
        uniform["datasets"][ds]["FCPS"] = ({"gamma": cos["gamma"], "cbar_T": cos["cbar_T"], "fcps_alpha": 4.0,
                                            "fcps_cmin_table3": 0.1, "fcps_cmin_table4": 1e-4,
                                            "note": "not tuned in the sweep; shares gamma and cbar_T with the selected Cosine"}
                                           if cos else None)
    for m in METHODS:
        (out / FILE_NAME[m]).write_text(json.dumps(per_method[m], indent=2), encoding="utf-8")
    (out / "uniform_parameters.json").write_text(json.dumps(uniform, indent=2), encoding="utf-8")

    # Metrics: copies of the sweep's final files.
    mdir = out / "metrics"
    mdir.mkdir(exist_ok=True)
    for name in ("TABLE5_TUNED.md", "table5_tuned_final.csv", "table5_tuned_paired.csv", "table5_tuned_em_sweep.csv"):
        src = sweep_root / "final" / name
        if src.is_file():
            shutil.copyfile(src, mdir / name)
    for name in ("selection.json", "SELECTION.md", "tuning_summary.csv"):
        src = sweep_root / "selection" / name
        if src.is_file():
            shutil.copyfile(src, mdir / name)
    if (sweep_root / "PROTOCOL.md").is_file():
        shutil.copyfile(sweep_root / "PROTOCOL.md", mdir / "PROTOCOL.md")

    # Where MASS is best, and where not (test split).
    check, cells, wins = [], 0, 0
    for ds in DATASETS:
        for b in T.SELECTION_BUDGETS:
            for mt in T.METRICS:
                vals = {}
                for m in METHODS:
                    fr = final_row(ds, m, b)
                    if fr and num(fr.get(f"{mt}_mean")) is not None:
                        vals[m] = num(fr[f"{mt}_mean"])
                if "MAS-SNR" not in vals:
                    check.append({"dataset": ds, "budget": b, "metric": mt, "mass_best": "", "note": "MASS not available"})
                    continue
                best = min(vals, key=vals.get)
                claims = {p["baseline"]: p for p in paired if p["dataset"] == ds and int(p["mass_budget"]) == b
                          and int(p["baseline_budget"]) == b and p["metric"] == mt}
                row = {"dataset": ds, "budget": b, "metric": T.METRIC_LABELS[mt], "best_method_by_mean": LABEL[best],
                       "mass_best": best == "MAS-SNR", **{f"{LABEL[m]}_mean": v for m, v in vals.items()}}
                for base in ("Cosine", "Isotropic-VP-Cosine"):
                    if base in claims:
                        c = claims[base]
                        row[f"MASS_better_than_{LABEL[base]}_claim_rule"] = c["mass_better"]
                        row[f"seeds_MASS_better_than_{LABEL[base]}"] = f"{c['seeds_mass_better']}/{c['paired_seeds']}"
                    elif base not in vals:
                        row[f"MASS_better_than_{LABEL[base]}_claim_rule"] = "no selected configuration"
                check.append(row)
                cells += 1
                wins += int(best == "MAS-SNR")
    write_csv(out / "MASS_best_check.csv", check)

    # Figure 2 data (test split), per seed and averaged by mode index.
    fdir = out / "figure2"
    fdir.mkdir(exist_ok=True)
    fig = sweep_root / "final" / "figure2_tuned_spectral_power_long.csv"
    by_mode_rows = []
    if fig.is_file():
        shutil.copyfile(fig, fdir / "figure2_spectral_power_long.csv")
        groups = defaultdict(list)
        for r in read_csv(fig):
            groups[(r["dataset"], r["schedule"], int(float(r["steps"])), int(float(r["mode"])))].append(r)
        for (ds, s, steps, mode), rr in sorted(groups.items()):
            row = {"dataset": ds, "schedule": s, "steps": steps, "mode": mode, "band": rr[0]["band"], "num_seeds": len(rr)}
            for col in ("lam_gamma", "reference_power", "generated_power", "log10_power_ratio"):
                mu, sd, _ = ms([num(r.get(col)) for r in rr])
                row[f"{col}_mean"], row[f"{col}_std"] = mu, sd
            by_mode_rows.append(row)
        write_csv(fdir / "figure2_spectral_power_by_mode.csv", by_mode_rows)
    (fdir / "figure2_manifest.json").write_text(json.dumps({
        "what_figure_2_shows": "Mode-wise power E[x_k^2] of generated signals (Euler-Maruyama, test split) against the "
                               "shifted Laplacian eigenvalue, with the test signals as reference; one panel per dataset; "
                               "shade band 'low' and 'high'. Each schedule at the configuration selected for that budget.",
        "files": {"figure2_spectral_power_long.csv": "per seed and mode",
                  "figure2_spectral_power_by_mode.csv": "mean and std over seeds by mode index (each seed has its own "
                                                        "graph, so for block covariance plot one seed from the long file)"},
    }, indent=2), encoding="utf-8")

    lines = ["# Selected Table V setup", "",
             f"Frozen selection sha256 {sel['selection_sha256']} (protocol {sel['protocol_sha256'][:12]}...).",
             "Selected on the validation split only, per method, by the pre-registered rule; not chosen by outcome.", "",
             f"MASS has the best test mean in {wins} of {cells} (dataset, budget, metric) cells; see MASS_best_check.csv.", "",
             "| Dataset | Method | Parameters carried to Tables III / IV |", "|---|---|---|"]
    for ds in DATASETS:
        for m in METHODS + ["FCPS"]:
            u = uniform["datasets"][ds].get(m)
            lines.append(f"| {ds} | {LABEL[m]} | {json.dumps(u) if u else 'no configuration selected'} |")
    lines += ["", "Files: mass.json, cosine.json, isotropic_vp_cosine.json (per method), uniform_parameters.json (read by the "
              "Table III / IV reruns), metrics/ (final Table V and the selection), figure2/ (Figure 2 data)."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    bundle = out.parent / "selected_setup_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, arcname=f"selected_setup/{p.relative_to(out)}")
    print(f"MASS best (test mean) in {wins} of {cells} cells")
    return out


def load_uniform(path: Path) -> dict:
    u = json.loads(path.read_text(encoding="utf-8"))
    for k in ("protocol_sha256", "selection_sha256", "datasets"):
        if k not in u:
            raise SystemExit(f"{path} is not a uniform_parameters.json")
    return u


# ------------------------------------------------------------------ Table III / Figure 1
def table3_tasks() -> list[dict]:
    return [{"dataset": ds, "seed": s, "schedule": sc} for ds in DATASETS for s in SEEDS for sc in TABLE3_SCHEDULES]


def table3_run_dir(root: Path, task: dict) -> Path:
    return root / "table3_figure1_selected" / task["dataset"] / f"seed_{task['seed']}" / task["schedule"]


def _data_and_schedules(ds: str, seed: int, gamma: float, sigma: float, code_dir: Path, **sched_kw):
    sys.path.insert(0, str(code_dir))
    import numpy as np
    import torch
    from datasets import make_synthetic_dataset_numpy
    from graphs import eigh_sym, rgg_knn_graph, shifted_laplacian
    from schedules import build_schedules
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, _, L = rgg_knn_graph(n=128, k=8, seed=seed)
    lam, V = eigh_sym(shifted_laplacian(L, gamma=gamma))
    X, _ = make_synthetic_dataset_numpy(family=ds, V=V, lam_gamma=lam, N=14000, seed=seed,
                                        spectrum="power_law", power_law_p=1.0, k_cut=16, strength=3.0,
                                        num_blocks=4, rho=0.8, num_bursts=2, burst_amp=3.0)
    s_k = np.mean((X[:10000] @ V) ** 2, axis=0)
    kw = dict(lam_gamma=lam, s_k=s_k, sigma=sigma, T=1.0, fcps_c0=2300.0, fcps_cmin=0.1, fcps_alpha=4.0,
              r_max=1e4, r_min=1e-3, match_terminal_energy=False, cosine_cbar_T=300.0, cosine_s=0.008,
              cosine_terminal_epsilon=1e-3)
    kw.update(sched_kw)
    return lam, s_k, build_schedules(**kw)


def table3_plan(uniform: dict, index: int, code_dir: Path) -> dict:
    tasks = table3_tasks()
    if not 0 <= index < len(tasks):
        raise SystemExit(f"index {index} out of range 0-{len(tasks) - 1}")
    task = tasks[index]
    ds, seed, sc = task["dataset"], task["seed"], task["schedule"]
    prm = uniform["datasets"][ds].get(sc)
    base = {"task": task, "uniform_selection_sha256": uniform["selection_sha256"]}
    if not prm:
        return {**base, "action": "not_selected", "driver_args": []}
    sys.path.insert(0, str(code_dir))
    from schedules import mass_endpoints_from_tolerances
    from theory_checks import mass_M, scalar_stiffness_lower_bound
    sigma = TABLE3_SIGMA[ds]
    gamma = float(prm["gamma"])
    args = [*TABLE3_COMMON, "--seed", str(seed), "--dataset_family", ds, *T.DATASETS[ds]["data_args"],
            "--gamma", f"{gamma:g}", "--sigma", f"{sigma:g}", "--schedulers", sc, "--diagnostics_schedulers", sc]
    theory = {}
    if sc == "MAS-SNR":
        lam, s_k, _ = _data_and_schedules(ds, seed, gamma, sigma, code_dir)
        r_max, r_min = mass_endpoints_from_tolerances(lam, s_k, sigma, start_a2_error=float(prm["delta0"]), terminal_a2=DELTA)
        _, _, sch = _data_and_schedules(ds, seed, gamma, sigma, code_dir, r_max=float(r_max), r_min=float(r_min))
        args += ["--r_max", repr(float(r_max)), "--r_min", repr(float(r_min))]
        theory = {"mass_r_max": float(r_max), "mass_r_min": float(r_min), "mass_M": float(mass_M(sch["MAS-SNR"]))}
    elif sc == "Cosine":
        cbar, eps = float(prm["cbar_T"]), float(prm["cosine_terminal_epsilon"])
        lam, s_k, sch = _data_and_schedules(ds, seed, gamma, sigma, code_dir, cosine_cbar_T=cbar,
                                            cosine_s=float(prm["cosine_s"]), cosine_terminal_epsilon=eps)
        args += ["--cosine_cbar_T", repr(cbar), "--cosine_s", f"{float(prm['cosine_s']):g}", "--cosine_terminal_epsilon", f"{eps:g}"]
        theory = scalar_stiffness_lower_bound("Cosine", sch["Cosine"], lam, s_k, sigma)
    else:  # FCPS
        cbar = float(prm["cbar_T"])
        lam, s_k, sch = _data_and_schedules(ds, seed, gamma, sigma, code_dir, fcps_c0=cbar,
                                            fcps_cmin=float(prm["fcps_cmin_table3"]), fcps_alpha=float(prm["fcps_alpha"]))
        args += ["--fcps_c0", repr(cbar), "--fcps_cmin", f"{float(prm['fcps_cmin_table3']):g}", "--fcps_alpha", f"{float(prm['fcps_alpha']):g}"]
        theory = scalar_stiffness_lower_bound("FCPS", sch["FCPS"], lam, s_k, sigma)
    theory = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in theory.items()}
    return {**base, "action": "run", "driver_args": args, "params": prm, "gamma": gamma, "sigma": sigma,
            "lambda_gamma_max": float(lam[-1]), "theory": theory}


def table3_aggregate(root: Path, uniform: dict, fig1_seed: int, max_times: int) -> Path:
    base = root / "table3_figure1_selected"
    missing = [t for t in table3_tasks() if not (table3_run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} Table III runs not finished; first missing: {missing[0]}")
    per = defaultdict(dict)       # (ds, seed) -> schedule -> dict
    for t in table3_tasks():
        rd = table3_run_dir(root, t)
        plan = json.loads((rd / "plan.json").read_text(encoding="utf-8"))
        entry = {"plan": plan, "L": None}
        if plan["action"] == "run":
            summ = rd / "forward_diagnostics" / f"forward_diagnostics_summary_{t['dataset']}.csv"
            rows = [r for r in read_csv(summ) if r["schedule"] == t["schedule"]]
            entry["L"] = num(rows[0]["max_stiffness_proxy"]) if rows else None
            entry["summary"] = rows[0] if rows else {}
        per[(t["dataset"], t["seed"])][t["schedule"]] = entry
    per_seed, table = [], []
    for ds in DATASETS:
        for sc in TABLE3_SCHEDULES:
            Ls, meas, guar, extra = [], [], [], defaultdict(list)
            for s in SEEDS:
                e, mass = per[(ds, s)][sc], per[(ds, s)]["MAS-SNR"]
                row = {"dataset": ds, "seed": s, "schedule": sc, "L_S0": e["L"]}
                Ls.append(e["L"])
                if sc != "MAS-SNR" and e["L"] is not None and mass["L"]:
                    th, mth = e["plan"]["theory"], mass["plan"]["theory"]
                    row["measured_reduction"] = e["L"] / mass["L"]
                    row["guaranteed_reduction"] = th["scalar_stiffness_lower_bound"] / mth["mass_M"]
                    row["scalar_c_T"] = th["scalar_c_T"]
                    row["prop5_condition_snr_le_1_3"] = th["scalar_theorem_condition_snr_le_1_over_3"]
                    meas.append(row["measured_reduction"]); guar.append(row["guaranteed_reduction"])
                if sc == "MAS-SNR" and e["plan"]["action"] == "run":
                    row["mass_M_bound"] = e["plan"]["theory"]["mass_M"]
                    row["mass_r_max"] = e["plan"]["theory"]["mass_r_max"]; row["mass_r_min"] = e["plan"]["theory"]["mass_r_min"]
                    extra["M"].append(row["mass_M_bound"])
                for k in ("mean_log_snr_range", "max_log_snr_range"):
                    if e.get("summary"):
                        row[k] = num(e["summary"].get(k)); extra[k].append(row[k])
                per_seed.append(row)
            L = ms(Ls); me = ms(meas); gu = ms(guar)
            prm = uniform["datasets"][ds].get(sc)
            table.append({"dataset": ds, "schedule": LABEL[sc], "parameters": json.dumps(prm) if prm else "not selected",
                          "num_seeds": L[2], "L_S0_mean": L[0], "L_S0_std": L[1],
                          "measured_reduction_mean": me[0] if sc != "MAS-SNR" else (1.0 if L[2] else None),
                          "measured_reduction_std": me[1] if sc != "MAS-SNR" else None,
                          "guaranteed_reduction_mean": gu[0], "guaranteed_reduction_std": gu[1],
                          "MASS_upper_bound_M": ms(extra["M"])[0] if sc == "MAS-SNR" else None,
                          "mean_log_snr_range_mean": ms(extra["mean_log_snr_range"])[0],
                          "max_log_snr_range_mean": ms(extra["max_log_snr_range"])[0]})
    out = base
    write_csv(out / "table3_selected_paper_columns.csv", table)
    write_csv(out / "table3_selected_per_seed.csv", per_seed)

    def f(m, s):
        if m is None:
            return "–"
        e = int(math.floor(math.log10(abs(m)))) if m else 0
        if abs(e) < 3:
            return f"{m:.2f} ± {s:.2f}" if s is not None else f"{m:.2f}"
        return f"({m / 10**e:.2f} ± {(s or 0) / 10**e:.2f})×10^{e}"
    lines = ["| Dataset | Schedule | L_S^(0) ↓ | Measured reduction ↑ | Guaranteed reduction ↑ |", "|---|---|---|---|---|"]
    for r in table:
        if r["schedule"] == "MASS":
            g = f"L_MASS^(0) ≤ {r['MASS_upper_bound_M']:.2f}" if r["MASS_upper_bound_M"] else "–"
            lines.append(f"| {r['dataset']} | MASS | {f(r['L_S0_mean'], r['L_S0_std'])} | 1 | {g} |")
        else:
            lines.append(f"| {r['dataset']} | {r['schedule']} | {f(r['L_S0_mean'], r['L_S0_std'])} | "
                         f"{f(r['measured_reduction_mean'], r['measured_reduction_std'])} | "
                         f"≥ {f(r['guaranteed_reduction_mean'], r['guaranteed_reduction_std'])} |")
    (out / "TABLE3_SELECTED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure 1 export: one seed, heat maps thinned in time.
    decay, heat = [], []
    for ds in DATASETS:
        for sc in TABLE3_SCHEDULES:
            t = {"dataset": ds, "seed": fig1_seed, "schedule": sc}
            fd = table3_run_dir(root, t) / "forward_diagnostics"
            sds = [p for p in fd.iterdir() if p.is_dir()] if fd.is_dir() else []
            for sd in sds:
                if (sd / "extreme_mode_mean_decay_curves.csv").is_file():
                    rows = read_csv(sd / "extreme_mode_mean_decay_curves.csv")
                    times = sorted({float(r["time"]) for r in rows})
                    keep = set(times[::max(1, math.ceil(len(times) / 512))]) | {times[0], times[-1]}
                    decay += [{"dataset": ds, "seed": fig1_seed, "schedule": sc,
                               **{k: r[k] for k in ("time", "normalized_time", "frequency_band", "band_rank", "mode",
                                                    "lam_gamma", "mean_decay_a_tk")}} for r in rows if float(r["time"]) in keep]
                if sc in ("Cosine", "MAS-SNR") and (sd / "modewise_log10_snr.csv").is_file():
                    rows = read_csv(sd / "modewise_log10_snr.csv")
                    times = sorted({float(r["time"]) for r in rows})
                    keep = set(times[::max(1, math.ceil(len(times) / max_times))]) | {times[0], times[-1]}
                    heat += [{"dataset": ds, "seed": fig1_seed, "schedule": sc,
                              **{k: r[k] for k in ("time", "normalized_time", "mode", "lam_gamma", "log10_snr_tk")}}
                             for r in rows if float(r["time"]) in keep]
    if decay:
        write_csv(out / "figure1_mean_decay.csv", decay)
    if heat:
        write_csv(out / "figure1_log10_snr_heatmap.csv", heat)
    (out / "figure1_manifest.json").write_text(json.dumps({
        "seed": fig1_seed, "parameters": "uniform_parameters.json of the Table V selection",
        "figure1_mean_decay.csv": "a_{t,k} of the 4 lowest and 4 highest graph frequencies; Cosine solid, MASS dashed (FCPS too)",
        "figure1_log10_snr_heatmap.csv": f"log10 SNR over lam_gamma and time for Cosine and MASS, <= {max_times} times; clip at -30",
    }, indent=2), encoding="utf-8")
    bundle = root / "table3_fig1_selected_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("TABLE3_SELECTED.md", "table3_selected_paper_columns.csv", "table3_selected_per_seed.csv",
                     "figure1_mean_decay.csv", "figure1_log10_snr_heatmap.csv", "figure1_manifest.json"):
            if (out / name).is_file():
                z.write(out / name, arcname=f"table3_fig1_selected/{name}")
    return bundle


# ------------------------------------------------------------------ Table IV
def table4_task(index: int) -> dict:
    if not 0 <= index < 40:
        raise SystemExit(f"index {index} out of range 0-39")
    ds = DATASETS[index // 20]
    sc = TABLE4_SCHEDULES[(index % 20) // 5]
    rep = index % 5
    return {"dataset": ds, "problem": TABLE4_DATASETS[ds], "schedule": sc, "safe": TABLE4_SAFE[sc], "rep": rep}


def table4_args(uniform: dict, index: int, problem_path: str) -> dict:
    t = table4_task(index)
    prm = uniform["datasets"][t["dataset"]].get(t["schedule"])
    if not prm:
        return {**t, "action": "not_selected", "driver_args": []}
    r = t["rep"]
    args = ["--problem", problem_path, "--candidate_id", f"selected_{t['safe']}", "--schedulers", t["schedule"],
            "--model_seed", str(3000 + r), "--train_seed", str(4000 + r), "--validation_seed", str(5000 + r),
            "--sample_seeds", str(6000 + r), "--plot_export_sample_seeds", str(6000 + r), *TABLE4_COMMON]
    sc = t["schedule"]
    if sc == "MAS-SNR":
        args += ["--gamma", f"{float(prm['gamma']):g}", "--mass_start_a2_error", f"{float(prm['delta0']):g}",
                 "--mass_terminal_a2", f"{DELTA:g}"]
    elif sc == "Cosine":
        args += ["--gamma", f"{float(prm['gamma']):g}", "--scalar_cbar_T", repr(float(prm["cbar_T"])),
                 "--cosine_s", f"{float(prm['cosine_s']):g}", "--cosine_terminal_epsilon", f"{float(prm['cosine_terminal_epsilon']):g}"]
    elif sc == "FCPS":
        args += ["--gamma", f"{float(prm['gamma']):g}", "--scalar_cbar_T", repr(float(prm["cbar_T"])),
                 "--fcps_alpha", f"{float(prm['fcps_alpha']):g}", "--fcps_cmin", f"{float(prm['fcps_cmin_table4']):g}"]
    else:
        args += ["--gamma", "1e-3", "--iso_cosine_s", f"{float(prm['iso_cosine_s']):g}",
                 "--iso_cosine_terminal_epsilon", f"{float(prm['iso_cosine_terminal_epsilon']):g}"]
    return {**t, "action": "run", "driver_args": args, "params": prm}


# ------------------------------------------------------------------ Table V under DPM-Solver++2M (and EM)
# Same data, network, training and metrics as the Table V sweep; every schedule
# with the parameters of uniform_parameters (FCPS/Cosine from
# add_scalar_baselines.py). The final samples are scored on the TEST split under
# DPM-Solver++2M at 10, 25, 50 steps and Euler-Maruyama at 50 and 2500 steps.
TABLE5DPM_SCHEDULES = ["Isotropic-VP-Cosine", "FCPS", "Cosine", "MAS-SNR"]
TABLE5DPM_STEPS = [10, 25, 50, 2500]
TABLE5DPM_POINTS = [("dpmpp2m", 10), ("dpmpp2m", 25), ("dpmpp2m", 50), ("em", 50), ("em", 2500)]


def _table5dpm_common() -> list[str]:
    args = list(T.COMMON_ARGS)
    args[args.index("--methods") + 1] = "em,dpmpp2m"
    args[args.index("--reverse_steps") + 1] = ",".join(str(s) for s in TABLE5DPM_STEPS)
    return args + ["--evaluation_split", "test"]


def table5dpm_task(index: int) -> dict:
    if not 0 <= index < 40:
        raise SystemExit(f"index {index} out of range 0-39")
    ds = DATASETS[index // 20]
    sc = TABLE5DPM_SCHEDULES[(index % 20) // 5]
    return {"dataset": ds, "schedule": sc, "safe": TABLE4_SAFE[sc], "seed": index % 5}


def table5dpm_run_dir(root: Path, task: dict) -> Path:
    return root / "table5_dpm" / task["dataset"] / task["safe"] / f"seed_{task['seed']}"


def table5dpm_plan(uniform: dict, index: int, code_dir: Path) -> dict:
    t = table5dpm_task(index)
    ds, sc, seed = t["dataset"], t["schedule"], t["seed"]
    prm = uniform["datasets"][ds].get(sc)
    base = {"task": t, "uniform_selection_sha256": uniform["selection_sha256"]}
    if not prm:
        return {**base, "action": "not_selected", "driver_args": []}
    sigma = TABLE3_SIGMA[ds]
    mass = uniform["datasets"][ds]["MAS-SNR"]
    gamma = float(prm.get("gamma", mass["gamma"]))
    args = [*_table5dpm_common(), "--seed", str(seed), "--dataset_family", ds, *T.DATASETS[ds]["data_args"],
            "--sigma", f"{sigma:g}", "--gamma", f"{gamma:g}", "--schedulers", sc, "--diagnostics_schedulers", sc]
    extra = {}
    if sc == "MAS-SNR":
        sys.path.insert(0, str(code_dir))
        from schedules import mass_endpoints_from_tolerances
        lam, s_k, _ = _data_and_schedules(ds, seed, gamma, sigma, code_dir)
        r_max, r_min = mass_endpoints_from_tolerances(lam, s_k, sigma, start_a2_error=float(prm["delta0"]), terminal_a2=DELTA)
        args += ["--r_max", repr(float(r_max)), "--r_min", repr(float(r_min))]
        extra = {"mass_r_max": float(r_max), "mass_r_min": float(r_min)}
    elif sc == "Cosine":
        args += ["--cosine_cbar_T", repr(float(prm["cbar_T"])), "--cosine_s", f"{float(prm['cosine_s']):g}",
                 "--cosine_terminal_epsilon", f"{float(prm['cosine_terminal_epsilon']):g}"]
    elif sc == "FCPS":
        args += ["--fcps_c0", repr(float(prm["cbar_T"])), "--fcps_cmin", f"{float(prm['fcps_cmin_table3']):g}",
                 "--fcps_alpha", f"{float(prm['fcps_alpha']):g}"]
    else:
        args += ["--iso_cosine_s", f"{float(prm['iso_cosine_s']):g}",
                 "--iso_cosine_terminal_epsilon", f"{float(prm['iso_cosine_terminal_epsilon']):g}"]
    return {**base, "action": "run", "driver_args": args, "params": prm, "gamma": gamma, "sigma": sigma, **extra}


def table5dpm_aggregate(root: Path, uniform: dict) -> Path:
    tasks = [table5dpm_task(i) for i in range(40)]
    missing = [t for t in tasks if not (table5dpm_run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} Table V (DPM) runs not finished; first missing: {missing[0]}")
    vals = defaultdict(dict)       # (ds, schedule, method, steps) -> seed -> row (stable only)
    unstable = defaultdict(int)
    for t in tasks:
        p = table5dpm_run_dir(root, t) / "results_summary.csv"
        if not p.is_file():
            continue
        for r in read_csv(p):
            if r.get("schedule") != t["schedule"]:
                continue
            if r.get("evaluation_split", "test") != "test":
                raise SystemExit(f"{p}: row not scored on the test split")
            key = (t["dataset"], t["schedule"], r["method"], int(float(r["steps"])))
            if T.is_stable(r) and all(num(r.get(m)) is not None for m in T.METRICS):
                vals[key][t["seed"]] = r
            else:
                unstable[key] += 1
    out = root / "table5_dpm"
    table, paired = [], []
    for ds in DATASETS:
        for meth, steps in TABLE5DPM_POINTS:
            for sc in TABLE5DPM_SCHEDULES:
                key = (ds, sc, meth, steps)
                rows = vals.get(key, {})
                row = {"dataset": ds, "schedule": LABEL[sc], "sampler": "DPM-Solver++2M" if meth == "dpmpp2m" else "Euler-Maruyama",
                       "steps": steps, "seeds_stable": len(rows), "seeds_unstable": unstable.get(key, 0),
                       "parameters": json.dumps(uniform["datasets"][ds].get(sc))}
                for m in T.METRICS:
                    mu, sd, _ = ms([num(r[m]) for r in rows.values()])
                    row[f"{m}_mean"], row[f"{m}_std"] = mu, sd
                table.append(row)
                if sc != "MAS-SNR":
                    mk = vals.get((ds, "MAS-SNR", meth, steps), {})
                    common = sorted(set(mk) & set(rows))
                    for m in T.METRICS:
                        d = [num(mk[s][m]) - num(rows[s][m]) for s in common]
                        if len(d) < 1:
                            continue
                        mu, sd, _ = ms(d)
                        wins = sum(x < 0 for x in d)
                        paired.append({"dataset": ds, "sampler": row["sampler"], "steps": steps, "baseline": LABEL[sc],
                                       "metric": T.METRIC_LABELS[m], "paired_seeds": len(d),
                                       "mean_diff_mass_minus_baseline": mu, "std_diff": sd, "seeds_mass_better": wins,
                                       "mass_better": bool(mu < 0 and wins >= 4 and len(d) == len(SEEDS))})
    write_csv(out / "table5_dpm_final.csv", table)
    write_csv(out / "table5_dpm_paired.csv", paired)

    def fmt(m, s):
        if m is None:
            return "diverged"
        return f"{m:.2f} ± {s:.2f}" if abs(m) >= 10 else f"{m:.4f} ± {s:.4f}"
    lines = ["| Dataset | Sampler | Steps | Schedule | " + " | ".join(T.METRIC_LABELS[m] + " ↓" for m in T.METRICS) + " |",
             "|---|---|---|---|" + "---|" * len(T.METRICS)]
    for ds in DATASETS:
        for meth, steps in TABLE5DPM_POINTS:
            rows = [r for r in table if r["dataset"] == ds and r["steps"] == steps
                    and r["sampler"] == ("DPM-Solver++2M" if meth == "dpmpp2m" else "Euler-Maruyama")]
            best = {m: min(((r[f"{m}_mean"], r["schedule"]) for r in rows if r[f"{m}_mean"] is not None), default=(None, None))[1]
                    for m in T.METRICS}
            for r in rows:
                note = "" if r["seeds_stable"] == len(SEEDS) else f" ({r['seeds_stable']}/5 seeds finite)"
                cells = [(f"**{fmt(r[f'{m}_mean'], r[f'{m}_std'])}**" if best[m] == r["schedule"] and r[f"{m}_mean"] is not None
                          else fmt(r[f"{m}_mean"], r[f"{m}_std"])) for m in T.METRICS]
                lines.append(f"| {ds} | {r['sampler']} | {steps} | {r['schedule']}{note} | " + " | ".join(cells) + " |")
    lines += ["", "Paired comparison (MASS - baseline per seed; 'better' = mean favours MASS and >= 4/5 seeds):", "",
              "| Dataset | Sampler | Steps | Baseline | Metric | Mean diff ± std | Seeds MASS better | Better |",
              "|---|---|---|---|---|---|---|---|"]
    for p in paired:
        lines.append(f"| {p['dataset']} | {p['sampler']} | {p['steps']} | {p['baseline']} | {p['metric']} | "
                     f"{p['mean_diff_mass_minus_baseline']:+.4g} ± {p['std_diff']:.3g} | {p['seeds_mass_better']}/{p['paired_seeds']} | "
                     f"{'yes' if p['mass_better'] else 'no'} |")
    (out / "TABLE5_DPM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure 2 data: every (schedule, sampler, steps), per seed and by mode.
    fig = []
    for t in tasks:
        p = table5dpm_run_dir(root, t) / "figure2_spectral_power_long.csv"
        if p.is_file():
            fig += [r for r in read_csv(p) if r.get("schedule") == t["schedule"]]
    if fig:
        write_csv(out / "figure2_dpm_spectral_power_long.csv", fig)
        groups = defaultdict(list)
        for r in fig:
            groups[(r["dataset"], r["schedule"], r["method"], int(float(r["steps"])), int(float(r["mode"])))].append(r)
        by_mode = []
        for (ds, sc, meth, steps, mode), rr in sorted(groups.items()):
            row = {"dataset": ds, "schedule": sc, "method": meth, "steps": steps, "mode": mode, "band": rr[0]["band"], "num_seeds": len(rr)}
            for col in ("lam_gamma", "reference_power", "generated_power", "log10_power_ratio"):
                mu, sd, _ = ms([num(r.get(col)) for r in rr])
                row[f"{col}_mean"], row[f"{col}_std"] = mu, sd
            by_mode.append(row)
        write_csv(out / "figure2_dpm_spectral_power_by_mode.csv", by_mode)
    (out / "figure2_dpm_manifest.json").write_text(json.dumps({
        "what": "Mode-wise power of generated signals (test split) against lam_gamma, reference = test signals; one row per "
                "(dataset, schedule, method, steps, mode); method dpmpp2m or em. For block covariance plot one seed.",
        "parameters": "uniform_parameters (selected Table V setup; FCPS/Cosine by the scalar-baseline rule)"}, indent=2), encoding="utf-8")
    bundle = root / "table5_dpm_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("TABLE5_DPM.md", "table5_dpm_final.csv", "table5_dpm_paired.csv", "figure2_dpm_spectral_power_long.csv",
                     "figure2_dpm_spectral_power_by_mode.csv", "figure2_dpm_manifest.json"):
            if (out / name).is_file():
                z.write(out / name, arcname=f"table5_dpm/{name}")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("export"); sp.add_argument("--sweep-root", type=Path, required=True); sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("table3-plan"); sp.add_argument("--params", type=Path, required=True); sp.add_argument("--index", type=int, required=True)
    sp.add_argument("--root", type=Path, required=True); sp.add_argument("--code-dir", type=Path, required=True); sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("table3-rundir"); sp.add_argument("--index", type=int, required=True); sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("table3-aggregate"); sp.add_argument("--params", type=Path, required=True); sp.add_argument("--root", type=Path, required=True)
    sp.add_argument("--fig1-seed", type=int, default=0); sp.add_argument("--fig1-max-times", type=int, default=256)
    sp = sub.add_parser("table3-status"); sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("table4-args"); sp.add_argument("--params", type=Path, required=True); sp.add_argument("--index", type=int, required=True)
    sp.add_argument("--problem", required=True); sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("table4-task"); sp.add_argument("--index", type=int, required=True)
    sp = sub.add_parser("table5dpm-plan"); sp.add_argument("--params", type=Path, required=True); sp.add_argument("--index", type=int, required=True)
    sp.add_argument("--code-dir", type=Path, required=True); sp.add_argument("--out", type=Path, required=True)
    sp = sub.add_parser("table5dpm-rundir"); sp.add_argument("--index", type=int, required=True); sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("table5dpm-status"); sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("table5dpm-aggregate"); sp.add_argument("--params", type=Path, required=True); sp.add_argument("--root", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        print(export(a.sweep_root, a.out))
    elif a.cmd == "table3-plan":
        plan = table3_plan(load_uniform(a.params), a.index, a.code_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        t = plan["task"]
        print(f"Table III plan: {t['dataset']} seed {t['seed']} {t['schedule']} -> {plan['action']}")
        if plan["action"] == "run":
            print(f"  parameters: {json.dumps(plan['params'])}; theory: {json.dumps(plan['theory'])}")
    elif a.cmd == "table3-rundir":
        print(table3_run_dir(a.root, table3_tasks()[a.index]))
    elif a.cmd == "table3-status":
        miss = [i for i, t in enumerate(table3_tasks()) if not (table3_run_dir(a.root, t) / "DONE.json").is_file()]
        print(f"table3: {len(table3_tasks()) - len(miss)} of {len(table3_tasks())} done" + (f"; missing {miss}" if miss else ""))
        raise SystemExit(3 if miss else 0)
    elif a.cmd == "table3-aggregate":
        print(table3_aggregate(a.root, load_uniform(a.params), a.fig1_seed, a.fig1_max_times))
    elif a.cmd == "table5dpm-plan":
        plan = table5dpm_plan(load_uniform(a.params), a.index, a.code_dir)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        t = plan["task"]
        print(f"Table V (DPM) plan: {t['dataset']} {t['schedule']} seed {t['seed']} -> {plan['action']}")
        if plan["action"] == "run":
            print(f"  parameters: {json.dumps(plan['params'])}" + (f"; r_max = {plan['mass_r_max']:.4g}, r_min = {plan['mass_r_min']:.4g}" if "mass_r_max" in plan else ""))
    elif a.cmd == "table5dpm-rundir":
        print(table5dpm_run_dir(a.root, table5dpm_task(a.index)))
    elif a.cmd == "table5dpm-status":
        miss = [i for i in range(40) if not (table5dpm_run_dir(a.root, table5dpm_task(i)) / "DONE.json").is_file()]
        print(f"table5_dpm: {40 - len(miss)} of 40 done" + (f"; missing {miss}" if miss else ""))
        raise SystemExit(3 if miss else 0)
    elif a.cmd == "table5dpm-aggregate":
        print(table5dpm_aggregate(a.root, load_uniform(a.params)))
    elif a.cmd == "table4-task":
        print(json.dumps(table4_task(a.index)))
    elif a.cmd == "table4-args":
        spec = table4_args(load_uniform(a.params), a.index, a.problem)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Table IV task {a.index}: {spec['dataset']} ({spec['problem']}) {spec['schedule']} rep {spec['rep']} -> {spec['action']}")


if __name__ == "__main__":
    main()
