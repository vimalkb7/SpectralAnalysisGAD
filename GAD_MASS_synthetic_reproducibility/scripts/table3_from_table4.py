from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import table4_tuning as T  # noqa: E402

SCHEDULES = ["FCPS", "Cosine", "MAS-SNR"]
LABEL = {"FCPS": "FCPS", "Cosine": "Cosine", "MAS-SNR": "MASS"}
DELTA = 1e-3
SIGMA = 1.0                       # Table IV
NUM_TIMES, T_MIN = 2048, 1e-4


def read_csv(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def build_one(problem, sc: str, prm: dict):
    """The forward process of schedule sc exactly as run_detailed_synthetic_study builds it."""
    import numpy as np
    from schedules import build_schedules, mass_endpoints_from_tolerances
    gamma = float(prm["gamma"])
    lam = problem.shifted_eigenvalues(gamma)
    s_k = problem.s_k_train
    cbar = float(prm.get("cbar_T", math.log(1.0 / DELTA) / (2.0 * gamma)))
    r_max, r_min = mass_endpoints_from_tolerances(lam, s_k, SIGMA, start_a2_error=float(prm.get("delta0", 1e-3)),
                                                  terminal_a2=DELTA)
    sch = build_schedules(lam_gamma=lam, s_k=s_k, sigma=SIGMA, T=1.0, fcps_c0=cbar,
                          fcps_cmin=float(prm.get("fcps_cmin", 0.1)), fcps_alpha=float(prm.get("fcps_alpha", 4.0)),
                          r_max=float(r_max), r_min=float(r_min), match_terminal_energy=False, uniform_cbar_T=cbar,
                          cosine_cbar_T=cbar, cosine_s=float(prm.get("cosine_s", 0.008)),
                          cosine_terminal_epsilon=float(prm.get("cosine_terminal_epsilon", 1e-3)), include_isotropic_vp=False)
    return lam, np.asarray(s_k), sch, {"gamma": gamma, "cbar_T": cbar if sc != "MAS-SNR" else None,
                                       "r_max": float(r_max) if sc == "MAS-SNR" else None,
                                       "r_min": float(r_min) if sc == "MAS-SNR" else None}


def run(table4_root: Path, out: Path, fig1_max_times: int = 256) -> Path:
    from diagnostics import run_forward_diagnostics
    from synthetic_problem import FrozenSyntheticProblem
    from theory_checks import mass_M, scalar_stiffness_lower_bound
    sel = T.load_selection(table4_root)
    by = {(e["dataset"], e["schedule"]): e for e in sel["entries"]}
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for ds in T.DATASET_ORDER:
        pname = T.DATASETS[ds]["problem"]
        ppath = table4_root / "problems" / f"{pname}.npz"
        if not ppath.is_file():
            raise SystemExit(f"Table IV problem not found: {ppath}")
        problem = FrozenSyntheticProblem.load(ppath)
        for sc in SCHEDULES:
            e = by[(ds, sc)]
            if not e.get("config_id"):
                raise SystemExit(f"no Table IV selection for {ds} {sc}")
            lam, s_k, sch, extra = build_one(problem, sc, e["params"])
            od = out / "forward_diagnostics" / ds / sc
            summ = run_forward_diagnostics(X_train_np=problem.X_train, V_np=problem.V, lam_gamma_np=lam, sigma=SIGMA,
                                           schedules=sch, scheduler_names=[sc], dataset_name=ds, out_dir=od,
                                           num_times=NUM_TIMES, num_extreme_modes=4, terminal_num_samples=2000, seed=0,
                                           pmin=1e-10, t_min=T_MIN, stiffness_h_proxy=0.0)
            srow = next(r for r in summ if r["schedule"] == sc)
            th = {"mass_M": float(mass_M(sch[sc]))} if sc == "MAS-SNR" else \
                {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                 for k, v in scalar_stiffness_lower_bound(sc, sch[sc], lam, s_k, SIGMA).items()}
            rows[(ds, sc)] = {"L": float(srow["max_stiffness_proxy"]), "theory": th, "extra": extra,
                              "cond": float(lam[-1] / lam[0]), "lam_max": float(lam[-1]), "config_id": e["config_id"],
                              "params": e["params"], "out_dir": od}
            print(f"{ds:20s} {LABEL[sc]:6s} L_S^(0) = {rows[(ds, sc)]['L']:.6g}  {json.dumps(th)}")
    table = []
    for ds in T.DATASET_ORDER:
        m = rows[(ds, "MAS-SNR")]
        for sc in SCHEDULES:
            r = rows[(ds, sc)]
            row = {"dataset": ds, "schedule": LABEL[sc], "config_id": r["config_id"], "params": json.dumps(r["params"]),
                   "gamma": r["extra"]["gamma"], "cond_L_gamma": r["cond"], "L_S0": r["L"]}
            if sc == "MAS-SNR":
                row.update({"measured_reduction": 1.0, "MASS_bound_M": r["theory"]["mass_M"],
                            "mass_r_max": r["extra"]["r_max"], "mass_r_min": r["extra"]["r_min"]})
            else:
                row.update({"measured_reduction": r["L"] / m["L"],
                            "guaranteed_reduction": r["theory"]["scalar_stiffness_lower_bound"] / m["theory"]["mass_M"],
                            "scalar_c_T": r["theory"]["scalar_c_T"], "cbar_T": r["extra"]["cbar_T"],
                            "prop5_condition_holds": r["theory"]["scalar_theorem_condition_snr_le_1_over_3"]})
            table.append(row)
    fields: list[str] = []
    for r in table:
        fields += [k for k in r if k not in fields]
    with (out / "table3_table4setup.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, restval=""); w.writeheader(); w.writerows(table)

    def sci(v):
        if v is None:
            return "–"
        e = int(math.floor(math.log10(abs(v))))
        return f"{v:.2f}" if -2 <= e < 3 else f"{v / 10**e:.2f}×10^{e}"
    lines = ["Table III on the Table IV setup (one graph, seed 123; settings selected in the Table IV sweep; sigma = 1)", "",
             "| Dataset | Schedule | gamma | cond(L_gamma) | L_S^(0) ↓ | Measured reduction ↑ | Guaranteed reduction ↑ |",
             "|---|---|---|---|---|---|---|"]
    for r in table:
        if r["schedule"] == "MASS":
            g = f"L_MASS^(0) ≤ {r['MASS_bound_M']:.2f}"
            lines.append(f"| {r['dataset']} | MASS | {r['gamma']:g} | {r['cond_L_gamma']:.0f} | {r['L_S0']:.2f} | 1 | {g} |")
        else:
            lines.append(f"| {r['dataset']} | {r['schedule']} | {r['gamma']:g} | {r['cond_L_gamma']:.0f} | {sci(r['L_S0'])} | "
                         f"{sci(r['measured_reduction'])} | ≥ {sci(r['guaranteed_reduction'])} |")
    lines += ["", "Proposition 5's condition holds for every scalar schedule: "
              + str(all(r.get("prop5_condition_holds") in (True, "True") for r in table if r["schedule"] != "MASS")), "",
              "Settings:", *[f"- {r['dataset']} {r['schedule']}: {r['config_id']} {r['params']}" for r in table]]
    (out / "TABLE3_TABLE4SETUP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure 1 export
    decay, heat = [], []
    for ds in T.DATASET_ORDER:
        for sc in SCHEDULES:
            od = rows[(ds, sc)]["out_dir"]
            for sub in [p for p in od.iterdir() if p.is_dir()]:
                f = sub / "extreme_mode_mean_decay_curves.csv"
                if f.is_file():
                    rr = read_csv(f); times = sorted({float(x["time"]) for x in rr})
                    keep = set(times[::max(1, math.ceil(len(times) / 512))]) | {times[0], times[-1]}
                    decay += [{"dataset": ds, "schedule": LABEL[sc], **{k: x[k] for k in ("time", "normalized_time", "frequency_band",
                               "band_rank", "mode", "lam_gamma", "mean_decay_a_tk")}} for x in rr if float(x["time"]) in keep]
                f = sub / "modewise_log10_snr.csv"
                if sc in ("Cosine", "MAS-SNR") and f.is_file():
                    rr = read_csv(f); times = sorted({float(x["time"]) for x in rr})
                    keep = set(times[::max(1, math.ceil(len(times) / fig1_max_times))]) | {times[0], times[-1]}
                    heat += [{"dataset": ds, "schedule": LABEL[sc], **{k: x[k] for k in ("time", "normalized_time", "mode",
                              "lam_gamma", "log10_snr_tk")}} for x in rr if float(x["time"]) in keep]
    for name, rr in (("figure1_mean_decay.csv", decay), ("figure1_log10_snr_heatmap.csv", heat)):
        if rr:
            with (out / name).open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    (out / "figure1_manifest.json").write_text(json.dumps({
        "setup": "Table IV problems (graph seed 123), Table IV selected settings, sigma = 1",
        "figure1_mean_decay.csv": "a_{t,k} of the 4 lowest and 4 highest graph frequencies (FCPS, Cosine, MASS)",
        "figure1_log10_snr_heatmap.csv": f"log10 SNR over lam_gamma and time for Cosine and MASS (<= {fig1_max_times} times; clip at -30)"},
        indent=2), encoding="utf-8")
    bundle = out / "table3_table4setup_upload_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("TABLE3_TABLE4SETUP.md", "table3_table4setup.csv", "figure1_mean_decay.csv",
                     "figure1_log10_snr_heatmap.csv", "figure1_manifest.json"):
            if (out / name).is_file():
                z.write(out / name, arcname=f"table3_table4setup/{name}")
        z.write(table4_root / "selection" / "selection.json", arcname="table3_table4setup/selection.json")
    print((out / "TABLE3_TABLE4SETUP.md").read_text())
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table4-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fig1-max-times", type=int, default=256)
    a = ap.parse_args()
    print(f"Upload: {run(a.table4_root, a.out, a.fig1_max_times)}")


if __name__ == "__main__":
    main()
