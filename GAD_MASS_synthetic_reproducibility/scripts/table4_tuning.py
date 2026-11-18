from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import zipfile
from pathlib import Path

PROTOCOL_VERSION = "table4-tuning-v1"
DATASETS = {
    "stationary_gaussian": {"problem": "broadband_wss", "problem_args": [
        "--family", "broadband_wss", "--n", "128", "--k", "8", "--graph_seed", "123", "--data_seed", "456",
        "--num_train", "10000", "--num_val", "2000", "--num_test", "2000",
        "--broadband_exponent", "1.5", "--broadband_delta", "0.05", "--broadband_high_band_target", "0.20"]},
    "block_covariance": {"problem": "block_covariance", "problem_args": [
        "--family", "block_covariance", "--n", "128", "--k", "8", "--graph_seed", "123", "--data_seed", "789",
        "--num_train", "10000", "--num_val", "2000", "--num_test", "2000"]},
}
DATASET_ORDER = ["stationary_gaussian", "block_covariance"]
SCHEDULES = ["Isotropic-VP-Cosine", "FCPS", "Cosine", "MAS-SNR"]
SAFE = {"Isotropic-VP-Cosine": "isotropic_vp_cosine", "FCPS": "fcps", "Cosine": "cosine", "MAS-SNR": "mass"}
LABEL = {"Isotropic-VP-Cosine": "Isotropic-VP-Cosine", "FCPS": "FCPS", "Cosine": "Cosine", "MAS-SNR": "MASS"}
SEEDS = [0, 1, 2, 3, 4]
DELTA = 1e-3
GAMMAS = [1e-3, 3e-3, 1e-2]
SELECTION_POINT = ("dpmpp2m", 10)
POINTS = [("dpmpp2m", 10), ("ddim", 25), ("dpmpp2m", 25)]
# Extra phase "ddim" (added after the protocol was frozen; not part of protocol_sha):
# the selected configurations, retrained with the same seeds, sampled with DDIM
# at 10 and 25 steps and DPM-Solver++2M at 10 and 25 steps on the test split.
DDIM_POINTS = [("ddim", 10), ("ddim", 25), ("dpmpp2m", 10), ("dpmpp2m", 25)]

COMMON = [
    "--sigma", "1.0", "--methods", "ddim,dpmpp2m", "--ddim_steps", "25", "--dpm_steps", "10,25",
    "--num_eval_samples", "1024", "--num_epochs", "40", "--batch_size", "128",
    "--hidden_dim", "64", "--num_layers", "4", "--time_embed_dim", "64", "--cheb_order", "3",
    "--lr", "3e-4", "--min_lr", "1e-6", "--train_t_min", "1e-2", "--val_every_epochs", "2",
    "--val_num_samples", "1024", "--t0", "1e-3", "--sampler_init_mode", "algorithm2",
    "--dtype", "float32", "--skip_forward_diagnostics", "--plot_export_steps", "10,25",
]


def grid() -> list[dict]:
    out = []
    for g in GAMMAS:
        cbar = math.log(1.0 / DELTA) / (2.0 * g)
        for a, c in ((4.0, 1e-4), (4.0, 0.1), (2.0, 1e-4), (8.0, 1e-4)):
            out.append({"id": f"fcps_gamma{g:g}_alpha{a:g}_cmin{c:g}", "schedule": "FCPS",
                        "params": {"gamma": g, "cbar_T": cbar, "fcps_alpha": a, "fcps_cmin": c}})
    for g in GAMMAS:
        cbar = math.log(1.0 / DELTA) / (2.0 * g)
        for e in (1e-3, 1e-2, 0.1, 0.5):
            out.append({"id": f"cosine_gamma{g:g}_eps{e:g}", "schedule": "Cosine",
                        "params": {"gamma": g, "cbar_T": cbar, "cosine_s": 0.008, "cosine_terminal_epsilon": e}})
    for g in GAMMAS:
        for d0 in (1e-3, 3e-3, 1e-2, 3e-2):
            out.append({"id": f"mass_gamma{g:g}_delta0{d0:g}", "schedule": "MAS-SNR",
                        "params": {"gamma": g, "delta0": d0, "delta_T": DELTA}})
    for s in (0.002, 0.008, 0.02, 0.05):
        for e in (1e-4, 1e-3, 1e-2):
            out.append({"id": f"iso_s{s:g}_eps{e:g}", "schedule": "Isotropic-VP-Cosine",
                        "params": {"iso_cosine_s": s, "iso_cosine_terminal_epsilon": e}})
    return out


GRID = grid()
GRID_BY_ID = {c["id"]: c for c in GRID}
GRID_ORDER = [c["id"] for c in GRID]


def protocol_dict() -> dict:
    return {"version": PROTOCOL_VERSION, "datasets": DATASETS, "seeds": SEEDS, "delta": DELTA, "grid": GRID,
            "common_args": COMMON, "selection_point": SELECTION_POINT, "points": POINTS,
            "selection_rule": "per dataset and schedule: finite in all seeds at the selection point; lowest mean "
                              "validation SWD there; ties by grid order"}


def protocol_sha() -> str:
    return hashlib.sha256(json.dumps(protocol_dict(), sort_keys=True).encode()).hexdigest()


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_protocol(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pj, sha = root / "protocol.json", protocol_sha()
    if pj.is_file():
        old = json.loads(pj.read_text(encoding="utf-8")).get("protocol_sha256")
        if old != sha:
            raise SystemExit(f"{pj} holds a different protocol ({old}); use a new results folder")
        return
    _atomic(pj, json.dumps({"protocol_sha256": sha, **protocol_dict()}, indent=2))
    lines = [f"# Table IV tuning protocol ({PROTOCOL_VERSION})", "", f"sha256: {sha}", "",
             __doc__.split("Subcommands")[0].strip(), "", "| Configuration | Schedule | Parameters |", "|---|---|---|"]
    lines += [f"| {c['id']} | {LABEL[c['schedule']]} | {json.dumps(c['params'])} |" for c in GRID]
    _atomic(root / "PROTOCOL.md", "\n".join(lines) + "\n")


# ------------------------------------------------------------------ tasks
def tune_tasks() -> list[dict]:
    return [{"phase": "tune", "dataset": ds, "config_id": c["id"], "schedule": c["schedule"], "seed": s}
            for ds in DATASET_ORDER for c in GRID for s in SEEDS]


def load_selection(root: Path) -> dict:
    p = root / "selection" / "selection.json"
    if not p.is_file():
        raise SystemExit(f"No selection yet: {p}")
    sel = json.loads(p.read_text(encoding="utf-8"))
    if sel.get("protocol_sha256") != protocol_sha():
        raise SystemExit("selection.json was made under a different protocol")
    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != sel.get("selection_sha256"):
        raise SystemExit("selection.json was modified after it was frozen")
    return sel


def test_tasks(root: Path) -> list[dict]:
    sel = {(e["dataset"], e["schedule"]): e for e in load_selection(root)["entries"]}
    return [{"phase": "test", "dataset": ds, "schedule": sc, "config_id": sel[(ds, sc)].get("config_id"), "seed": s}
            for ds in DATASET_ORDER for sc in SCHEDULES for s in SEEDS]


def tasks_for(phase: str, root: Path) -> list[dict]:
    if phase == "tune":
        return tune_tasks()
    tt = test_tasks(root)
    return tt if phase == "test" else [{**t, "phase": "ddim"} for t in tt]


def run_dir(root: Path, t: dict) -> Path:
    if t["phase"] == "tune":
        return root / "tune" / t["dataset"] / t["config_id"] / f"seed_{t['seed']}"
    return root / t["phase"] / t["dataset"] / SAFE[t["schedule"]] / f"seed_{t['seed']}"


def driver_args(t: dict, problem: str) -> list[str]:
    cfg = GRID_BY_ID[t["config_id"]]
    p, r = cfg["params"], int(t["seed"])
    split = "val" if t["phase"] == "tune" else "test"
    args = ["--problem", problem, "--candidate_id", t["config_id"], "--schedulers", t["schedule"],
            "--model_seed", str(3000 + r), "--train_seed", str(4000 + r), "--validation_seed", str(5000 + r),
            "--sample_seeds", str(6000 + r), "--plot_export_sample_seeds", str(6000 + r),
            "--evaluation_split", split, "--study_phase", f"table4_{t['phase']}", *COMMON]
    if t["phase"] == "ddim":
        args[args.index("--ddim_steps") + 1] = "10,25"
    sc = t["schedule"]
    if sc == "MAS-SNR":
        args += ["--gamma", f"{p['gamma']:g}", "--mass_start_a2_error", f"{p['delta0']:g}", "--mass_terminal_a2", f"{DELTA:g}"]
    elif sc == "FCPS":
        args += ["--gamma", f"{p['gamma']:g}", "--scalar_cbar_T", repr(float(p["cbar_T"])),
                 "--fcps_alpha", f"{p['fcps_alpha']:g}", "--fcps_cmin", f"{p['fcps_cmin']:g}"]
    elif sc == "Cosine":
        args += ["--gamma", f"{p['gamma']:g}", "--scalar_cbar_T", repr(float(p["cbar_T"])),
                 "--cosine_s", f"{p['cosine_s']:g}", "--cosine_terminal_epsilon", f"{p['cosine_terminal_epsilon']:g}"]
    else:
        args += ["--gamma", "1e-3", "--iso_cosine_s", f"{p['iso_cosine_s']:g}",
                 "--iso_cosine_terminal_epsilon", f"{p['iso_cosine_terminal_epsilon']:g}"]
    return args


# ------------------------------------------------------------------ reading results
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


def swd_at(run: Path, schedule: str, point: tuple, split: str):
    """SWD of one run at (method, nfe), or None if missing or not finite."""
    p = run / "results_raw.csv"
    if not p.is_file():
        return None
    for r in read_csv(p):
        if r.get("schedule") == schedule and r.get("method") == point[0] and int(float(r.get("nfe", -1))) == point[1]:
            if r.get("evaluation_split", split) != split:
                raise SystemExit(f"{p}: row scored on {r.get('evaluation_split')!r}, expected {split!r}")
            if str(r.get("stable", "true")).lower() not in ("true", "1", "yes"):
                return None
            return num(r.get("global_swd_mean"))
    return None


def ms(v):
    return (statistics.fmean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (None, None)


# ------------------------------------------------------------------ selection and final table
def select(root: Path) -> Path:
    missing = [t for t in tune_tasks() if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} tuning runs are not finished; first missing: {missing[0]}")
    summary, entries = [], []
    for ds in DATASET_ORDER:
        for sc in SCHEDULES:
            cand = []
            for c in (c for c in GRID if c["schedule"] == sc):
                row = {"dataset": ds, "schedule": LABEL[sc], "config_id": c["id"], "params": json.dumps(c["params"])}
                finite_all = True
                for pt in POINTS:
                    v = [swd_at(run_dir(root, {"phase": "tune", "dataset": ds, "config_id": c["id"], "seed": s}), sc, pt, "val")
                         for s in SEEDS]
                    ok = [x for x in v if x is not None]
                    mu, sd = ms(ok)
                    row[f"val_swd_{pt[0]}{pt[1]}_mean"], row[f"val_swd_{pt[0]}{pt[1]}_std"] = mu, sd
                    row[f"seeds_finite_{pt[0]}{pt[1]}"] = len(ok)
                    if pt == SELECTION_POINT and len(ok) < len(SEEDS):
                        finite_all = False
                row["eligible"] = finite_all
                summary.append(row)
                if finite_all:
                    cand.append(row)
            key = f"val_swd_{SELECTION_POINT[0]}{SELECTION_POINT[1]}_mean"
            if cand:
                best = min(cand, key=lambda r: (r[key], GRID_ORDER.index(r["config_id"])))
                entries.append({"dataset": ds, "schedule": sc, "config_id": best["config_id"],
                                "params": GRID_BY_ID[best["config_id"]]["params"], "val_swd_mean": best[key],
                                "num_eligible": len(cand)})
            else:
                entries.append({"dataset": ds, "schedule": sc, "config_id": None, "num_eligible": 0,
                                "reason": "no configuration finite in all seeds at the selection point"})
    out = root / "selection"
    write_csv(out / "tuning_summary.csv", summary)
    body = {"protocol_version": PROTOCOL_VERSION, "protocol_sha256": protocol_sha(), "entries": entries}
    body["selection_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    _atomic(out / "selection.json", json.dumps(body, indent=2))
    lines = [f"# Table IV tuning: selected configurations ({PROTOCOL_VERSION})", "",
             "Selected on the validation split (mean SWD, DPM-Solver++2M, 10 steps). Frozen; sha256 " + body["selection_sha256"], "",
             "| Dataset | Schedule | Selected configuration | Validation SWD | Eligible / 12 |", "|---|---|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['dataset']} | {LABEL[e['schedule']]} | {e['config_id'] or e['reason']} | "
                     f"{e['val_swd_mean']:.5f} | {e['num_eligible']} |" if e["config_id"] else
                     f"| {e['dataset']} | {LABEL[e['schedule']]} | {e['reason']} | – | 0 |")
    _atomic(out / "SELECTION.md", "\n".join(lines) + "\n")
    return out / "selection.json"


def final(root: Path) -> Path:
    sel = load_selection(root)
    tt = test_tasks(root)
    missing = [t for t in tt if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} test runs are not finished; first missing: {missing[0]}")
    by = {(e["dataset"], e["schedule"]): e for e in sel["entries"]}
    vals = {}
    for ds in DATASET_ORDER:
        for sc in SCHEDULES:
            for pt in POINTS:
                vals[(ds, sc, pt)] = {s: swd_at(run_dir(root, {"phase": "test", "dataset": ds, "schedule": sc, "seed": s}), sc, pt, "test")
                                      for s in SEEDS}
    rows, paired = [], []
    for ds in DATASET_ORDER:
        for pt in POINTS:
            for sc in SCHEDULES:
                v = [x for x in vals[(ds, sc, pt)].values() if x is not None]
                mu, sd = ms(v)
                rows.append({"dataset": ds, "schedule": LABEL[sc], "method": pt[0], "nfe": pt[1], "swd_mean": mu, "swd_std": sd,
                             "seeds_finite": len(v), "config_id": by[(ds, sc)].get("config_id") or "",
                             "params": json.dumps(by[(ds, sc)].get("params"))})
                if sc != "MAS-SNR":
                    a, b = vals[(ds, "MAS-SNR", pt)], vals[(ds, sc, pt)]
                    d = [a[s] - b[s] for s in SEEDS if a[s] is not None and b[s] is not None]
                    if d:
                        m, s_ = ms(d)
                        paired.append({"dataset": ds, "method": pt[0], "nfe": pt[1], "baseline": LABEL[sc], "paired_seeds": len(d),
                                       "mean_diff_mass_minus_baseline": m, "std_diff": s_, "seeds_mass_better": sum(x < 0 for x in d)})
    out = root / "final"
    write_csv(out / "table4_tuned_final.csv", rows)
    write_csv(out / "table4_tuned_paired.csv", paired)

    def get(ds, sc, pt):
        return next(r for r in rows if r["dataset"] == ds and r["schedule"] == LABEL[sc] and (r["method"], r["nfe"]) == pt)
    f = lambda r: "diverged" if r["swd_mean"] is None else f"{r['swd_mean']:.5f} ± {r['swd_std']:.5f}"
    lines = ["(a) DPM-Solver++2M, 10 steps (test SWD, mean ± std over 5 seeds)", "",
             "| Schedule | Stationary (broadband WSS) | Block covariance |", "|---|---|---|"]
    for sc in SCHEDULES:
        lines.append(f"| {LABEL[sc]} | {f(get('stationary_gaussian', sc, ('dpmpp2m', 10)))} | {f(get('block_covariance', sc, ('dpmpp2m', 10)))} |")
    rank_lines = []
    for ds in DATASET_ORDER:
        rr = sorted((get(ds, sc, ("dpmpp2m", 10)) for sc in SCHEDULES), key=lambda r: (r["swd_mean"] is None, r["swd_mean"] or 0))
        rank_lines.append(f"{ds}: " + " < ".join(r["schedule"] for r in rr))
        base = [r for r in rr if r["schedule"] != "MASS" and r["swd_mean"] is not None]
        m = get(ds, "MAS-SNR", ("dpmpp2m", 10))
        if base and m["swd_mean"] is not None:
            rank_lines.append(f"  MASS vs best baseline ({base[0]['schedule']}): {100 * (1 - m['swd_mean'] / base[0]['swd_mean']):.1f}% lower SWD")
    lines += ["", "Ranking (lower SWD first):", *rank_lines, "",
              "(b) FCPS vs MASS at 25 steps", "", "| Schedule | Integrator | Stationary | Block covariance |", "|---|---|---|---|"]
    for pt, name in ((("ddim", 25), "DDIM"), (("dpmpp2m", 25), "DPM-Solver++2M")):
        for sc in ("FCPS", "MAS-SNR"):
            lines.append(f"| {LABEL[sc]} | {name} | {f(get('stationary_gaussian', sc, pt))} | {f(get('block_covariance', sc, pt))} |")
    lines += ["", "Paired per seed (MASS - baseline):", "", "| Dataset | Point | Baseline | Mean diff ± std | Seeds MASS better |", "|---|---|---|---|---|"]
    lines += [f"| {p['dataset']} | {p['method']} {p['nfe']} | {p['baseline']} | {p['mean_diff_mass_minus_baseline']:+.5f} ± {p['std_diff']:.5f} | "
              f"{p['seeds_mass_better']}/{p['paired_seeds']} |" for p in paired]
    lines += ["", "Selected configurations (validation):", ""]
    lines += [f"- {e['dataset']} {LABEL[e['schedule']]}: {e.get('config_id') or e.get('reason')} {json.dumps(e.get('params'))}" for e in sel["entries"]]
    _atomic(out / "TABLE4_TUNED.md", "\n".join(lines) + "\n")
    bundle = root / "table4_tuned_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in (out / "TABLE4_TUNED.md", out / "table4_tuned_final.csv", out / "table4_tuned_paired.csv",
                  root / "selection" / "selection.json", root / "selection" / "SELECTION.md",
                  root / "selection" / "tuning_summary.csv", root / "PROTOCOL.md"):
            if p.is_file():
                z.write(p, arcname=f"table4_tuned/{p.name}")
    return bundle


def final_ddim(root: Path) -> Path:
    sel = load_selection(root)
    tt = tasks_for("ddim", root)
    missing = [t for t in tt if not (run_dir(root, t) / "DONE.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} DDIM runs are not finished; first missing: {missing[0]}")
    by = {(e["dataset"], e["schedule"]): e for e in sel["entries"]}
    vals = {(ds, sc, pt): {s: swd_at(run_dir(root, {"phase": "ddim", "dataset": ds, "schedule": sc, "seed": s}), sc, pt, "test")
                           for s in SEEDS}
            for ds in DATASET_ORDER for sc in SCHEDULES for pt in DDIM_POINTS}
    rows, paired, checks = [], [], []
    for ds in DATASET_ORDER:
        for pt in DDIM_POINTS:
            for sc in SCHEDULES:
                v = [x for x in vals[(ds, sc, pt)].values() if x is not None]
                mu, sd = ms(v)
                rows.append({"dataset": ds, "schedule": LABEL[sc], "method": pt[0], "nfe": pt[1], "swd_mean": mu, "swd_std": sd,
                             "seeds_finite": len(v), "config_id": by[(ds, sc)].get("config_id") or ""})
                if sc != "MAS-SNR":
                    a, b = vals[(ds, "MAS-SNR", pt)], vals[(ds, sc, pt)]
                    d = [a[s] - b[s] for s in SEEDS if a[s] is not None and b[s] is not None]
                    if d:
                        m, s_ = ms(d)
                        paired.append({"dataset": ds, "method": pt[0], "nfe": pt[1], "baseline": LABEL[sc], "paired_seeds": len(d),
                                       "mean_diff_mass_minus_baseline": m, "std_diff": s_, "seeds_mass_better": sum(x < 0 for x in d)})
                # the DPM-Solver++ points must reproduce the earlier test phase (same configuration and seeds)
                if pt[0] == "dpmpp2m" and (run_dir(root, {"phase": "test", "dataset": ds, "schedule": sc, "seed": 0}) / "DONE.json").is_file():
                    old = [swd_at(run_dir(root, {"phase": "test", "dataset": ds, "schedule": sc, "seed": s}), sc, pt, "test") for s in SEEDS]
                    new = [vals[(ds, sc, pt)][s] for s in SEEDS]
                    diffs = [abs(x - y) for x, y in zip(old, new) if x is not None and y is not None]
                    checks.append({"dataset": ds, "schedule": LABEL[sc], "point": f"{pt[0]} {pt[1]}",
                                   "max_abs_diff_vs_test_phase": max(diffs) if diffs else None})
    out = root / "ddim"
    write_csv(out / "table4_ddim_final.csv", rows)
    write_csv(out / "table4_ddim_paired.csv", paired)
    write_csv(out / "table4_ddim_reproduction_check.csv", checks)
    f = lambda r: "diverged" if r["swd_mean"] is None else f"{r['swd_mean']:.5f} ± {r['swd_std']:.5f}"
    lines = ["Selected configurations (Table IV tuning), test SWD, mean ± std over 5 paired seeds", "",
             "| Integrator | Steps | Schedule | Stationary (broadband WSS) | Block covariance |", "|---|---|---|---|---|"]
    for pt in DDIM_POINTS:
        for sc in SCHEDULES:
            r = {ds: next(x for x in rows if x["dataset"] == ds and x["schedule"] == LABEL[sc] and (x["method"], x["nfe"]) == pt)
                 for ds in DATASET_ORDER}
            name = "DDIM" if pt[0] == "ddim" else "DPM-Solver++2M"
            lines.append(f"| {name} | {pt[1]} | {LABEL[sc]} | {f(r['stationary_gaussian'])} | {f(r['block_covariance'])} |")
    lines += ["", "Ranking per point (lower SWD first):", ""]
    for ds in DATASET_ORDER:
        for pt in DDIM_POINTS:
            rr = sorted((x for x in rows if x["dataset"] == ds and (x["method"], x["nfe"]) == pt),
                        key=lambda x: (x["swd_mean"] is None, x["swd_mean"] or 0))
            lines.append(f"- {ds} {pt[0]} {pt[1]}: " + " < ".join(x["schedule"] for x in rr))
    lines += ["", "Paired per seed (MASS - baseline):", "", "| Dataset | Point | Baseline | Mean diff ± std | Seeds MASS better |",
              "|---|---|---|---|---|"]
    lines += [f"| {p['dataset']} | {p['method']} {p['nfe']} | {p['baseline']} | {p['mean_diff_mass_minus_baseline']:+.5f} ± "
              f"{p['std_diff']:.5f} | {p['seeds_mass_better']}/{p['paired_seeds']} |" for p in paired]
    worst = max((c["max_abs_diff_vs_test_phase"] for c in checks if c["max_abs_diff_vs_test_phase"] is not None), default=None)
    lines += ["", f"Reproduction check: largest per-seed |SWD difference| of the DPM-Solver++ points against the earlier test phase = "
              f"{worst:.2e}" if worst is not None else "", ""]
    _atomic(out / "TABLE4_DDIM.md", "\n".join(lines) + "\n")
    bundle = root / "table4_ddim_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in (out / "TABLE4_DDIM.md", out / "table4_ddim_final.csv", out / "table4_ddim_paired.csv",
                  out / "table4_ddim_reproduction_check.csv", root / "selection" / "selection.json", root / "selection" / "SELECTION.md"):
            if p.is_file():
                z.write(p, arcname=f"table4_ddim/{p.name}")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("protocol", "select", "final", "final-ddim"):
        sub.add_parser(name).add_argument("--root", type=Path, required=True)
    for name in ("count", "status"):
        sp = sub.add_parser(name); sp.add_argument("--phase", choices=("tune", "test", "ddim"), required=True)
        sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("describe"); sp.add_argument("--phase", choices=("tune", "test", "ddim"), required=True)
    sp.add_argument("--index", type=int, required=True); sp.add_argument("--root", type=Path, required=True)
    sp = sub.add_parser("args"); sp.add_argument("--phase", choices=("tune", "test", "ddim"), required=True)
    sp.add_argument("--index", type=int, required=True); sp.add_argument("--root", type=Path, required=True)
    sp.add_argument("--problem", required=True); sp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "protocol":
        write_protocol(a.root); print(a.root / "PROTOCOL.md")
    elif a.cmd == "count":
        print(len(tasks_for(a.phase, a.root)))
    elif a.cmd == "status":
        t = tasks_for(a.phase, a.root)
        miss = [i for i, x in enumerate(t) if not (run_dir(a.root, x) / "DONE.json").is_file()]
        print(f"{a.phase}: {len(t) - len(miss)} of {len(t)} done")
        if miss:
            spans, start = [], miss[0]
            for prev, cur in zip(miss, miss[1:] + [None]):
                if cur != prev + 1:
                    spans.append(f"{start}" if start == prev else f"{start}-{prev}"); start = cur
            print("missing: " + ",".join(spans))
        raise SystemExit(0 if not miss else 3)
    elif a.cmd in ("describe", "args"):
        t = tasks_for(a.phase, a.root)
        if not 0 <= a.index < len(t):
            raise SystemExit(f"index {a.index} out of range 0-{len(t) - 1} for phase {a.phase}")
        task = t[a.index]
        info = {**task, "run_dir": str(run_dir(a.root, task)), "problem_name": DATASETS[task["dataset"]]["problem"],
                "problem_args": DATASETS[task["dataset"]]["problem_args"]}
        if a.cmd == "describe":
            print(json.dumps(info))
        else:
            spec = {**info, "action": "run" if task.get("config_id") else "not_selected",
                    "driver_args": driver_args(task, a.problem) if task.get("config_id") else [],
                    "params": GRID_BY_ID[task["config_id"]]["params"] if task.get("config_id") else None,
                    "protocol_sha256": protocol_sha()}
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            print(f"Table IV {a.phase} task {a.index}: {task['dataset']} {LABEL[task['schedule']]} "
                  f"{task.get('config_id')} seed {task['seed']} -> {spec['action']}")
    elif a.cmd == "select":
        print(select(a.root))
    elif a.cmd == "final":
        print(final(a.root))
    elif a.cmd == "final-ddim":
        print(final_ddim(a.root))


if __name__ == "__main__":
    main()