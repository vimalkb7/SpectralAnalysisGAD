from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE))
import selected_setup as S
import table5_tuning as T 
import test_table5_tuning as TT


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        T.write_protocol(root)
        for t in T.tune_tasks():
            TT.fabricate_run(root, "tune", t["dataset"], t["config_id"], t["seed"])
        T.select(root)
        for t in T.test_tasks(root):
            TT.fabricate_run(root, "test", t["dataset"], t["config_id"], t["seed"])
        for t in T.test_tasks(root):
            rows = [{"dataset": t["dataset"], "seed": t["seed"], "schedule": t["method"], "method": "em", "steps": n,
                     "mode": k, "lam_gamma": 0.5 * k, "band": "low" if k < 2 else ("high" if k >= 6 else "middle"),
                     "reference_power": 1.0 / (k + 1), "generated_power": 1.1 / (k + 1), "log10_power_ratio": 0.0414}
                    for n in (50, 2500) for k in range(8)]
            TT.write_csv(T.run_dir(root, t) / "figure2_spectral_power_long.csv", rows)
        for t in T.test_tasks(root):
            if t["method"] == "MAS-SNR":
                p = T.run_dir(root, t) / "plan.json"
                d = json.loads(p.read_text()); d.update({"mass_r_max": 100.0 + t["seed"], "mass_r_min": 1e-4}); p.write_text(json.dumps(d))
        try:
            S.export(root, root / "selected_setup")
        except SystemExit as err:
            assert "final table" in str(err)
        else:
            raise AssertionError("export must wait for the sweep's final table")
        T.final(root)
        out = S.export(root, root / "selected_setup")
        for name in ("mass.json", "cosine.json", "isotropic_vp_cosine.json", "uniform_parameters.json", "MASS_best_check.csv",
                     "README.md", "metrics/TABLE5_TUNED.md", "metrics/selection.json", "figure2/figure2_spectral_power_long.csv",
                     "figure2/figure2_spectral_power_by_mode.csv"):
            assert (out / name).is_file(), name
        assert (root / "selected_setup_bundle.zip").is_file()
        u = S.load_uniform(out / "uniform_parameters.json")
        sel = T.load_selection(root)
        assert u["selection_sha256"] == sel["selection_sha256"]
        by = {(e["dataset"], e["method"], e["budget"]): e for e in sel["entries"]}
        for ds in S.DATASETS:
            for m in S.METHODS:
                want = by[(ds, m, 50)] if by[(ds, m, 50)].get("config_id") else by[(ds, m, 2500)]
                assert u["datasets"][ds][m]["config_id"] == want["config_id"], (ds, m)
            assert u["datasets"][ds]["Cosine"]["selected_at_budget"] == 2500      # no stable Cosine at 50 steps
            assert u["datasets"][ds]["FCPS"]["gamma"] == u["datasets"][ds]["Cosine"]["gamma"]
        mass = json.loads((out / "mass.json").read_text())
        assert len(mass["datasets"]["stationary_gaussian"]["mass_endpoints_per_seed_table5"]) == 5
        by_mode = S.read_csv(out / "figure2" / "figure2_spectral_power_by_mode.csv")
        assert by_mode and all(r["num_seeds"] == "5" for r in by_mode)
        check = S.read_csv(out / "MASS_best_check.csv")
        assert any(r["mass_best"] == "True" for r in check) and any(r["mass_best"] == "False" for r in check)

        assert len(S.table3_tasks()) == 30
        assert [t["schedule"] for t in S.table3_tasks()[:3]] == ["FCPS", "Cosine", "MAS-SNR"]
        for i in range(40):
            spec = S.table4_args(u, i, "/p.npz")
            a = spec["driver_args"]
            prm = u["datasets"][spec["dataset"]][spec["schedule"]]
            if spec["schedule"] == "MAS-SNR":
                assert a[a.index("--gamma") + 1] == f"{prm['gamma']:g}" and a[a.index("--mass_start_a2_error") + 1] == f"{prm['delta0']:g}"
            if spec["schedule"] == "Cosine":
                assert a[a.index("--cosine_terminal_epsilon") + 1] == f"{prm['cosine_terminal_epsilon']:g}"
            if spec["schedule"] == "Isotropic-VP-Cosine":
                assert a[a.index("--iso_cosine_s") + 1] == f"{prm['iso_cosine_s']:g}"
            assert a[a.index("--model_seed") + 1] == str(3000 + spec["rep"])
        assert S.table4_task(0)["schedule"] == "Isotropic-VP-Cosine" and S.table4_task(39)["schedule"] == "MAS-SNR"
        u2 = json.loads(json.dumps(u)); u2["datasets"]["block_covariance"]["Cosine"] = None
        assert S.table4_args(u2, 30, "/p.npz")["action"] == "not_selected"
        import importlib.util
        spec = importlib.util.spec_from_file_location("asb", HERE / "scripts" / "add_scalar_baselines.py")
        asb = importlib.util.module_from_spec(spec); spec.loader.exec_module(asb)
        u34 = json.loads(json.dumps(u)); u34["datasets"]["block_covariance"]["Cosine"] = None; u34["datasets"]["block_covariance"]["FCPS"] = None
        (root / "u.json").write_text(json.dumps(u34))
        sys.argv = ["asb", str(root / "u.json"), str(root / "u34.json")]; asb.main()
        u34 = S.load_uniform(root / "u34.json")
        cos = u34["datasets"]["block_covariance"]["Cosine"]
        assert cos["gamma"] == u34["datasets"]["block_covariance"]["MAS-SNR"]["gamma"] and cos["cosine_terminal_epsilon"] == 1e-3
        assert abs(cos["cbar_T"] - __import__("math").log(1e3) / (2 * cos["gamma"])) < 1e-9
        assert [S.table5dpm_task(i)["schedule"] for i in (0, 5, 10, 15)] == S.TABLE5DPM_SCHEDULES
        for i in (20, 25, 30):     # Isotropic, FCPS, Cosine (no torch needed)
            pl = S.table5dpm_plan(u34, i, HERE)
            a = pl["driver_args"]
            assert pl["action"] == "run" and a[a.index("--evaluation_split") + 1] == "test"
            assert a[a.index("--methods") + 1] == "em,dpmpp2m" and a[a.index("--reverse_steps") + 1] == "10,25,50,2500"
        assert "--fcps_c0" in S.table5dpm_plan(u34, 25, HERE)["driver_args"]
        assert "--cosine_cbar_T" in S.table5dpm_plan(u34, 30, HERE)["driver_args"]
        for i in range(40):
            t = S.table5dpm_task(i); rd = S.table5dpm_run_dir(root, t)
            rows = []
            for meth in ("em", "dpmpp2m"):
                for n in (10, 25, 50, 2500):
                    bad = meth == "em" and t["schedule"] == "Cosine"
                    q = {"MAS-SNR": 0.07, "FCPS": 0.08, "Isotropic-VP-Cosine": 0.09, "Cosine": 0.25}[t["schedule"]] + 0.001 * t["seed"]
                    rows.append({"schedule": t["schedule"], "method": meth, "steps": n, "evaluation_split": "test", "stable": not bad,
                                 **({m: q for m in T.METRICS} if not bad else {})})
            TT.write_csv(rd / "results_summary.csv", rows); (rd / "DONE.json").write_text("{}")
        S.table5dpm_aggregate(root, u34)
        md = (root / "table5_dpm" / "TABLE5_DPM.md").read_text()
        assert "| block_covariance | Euler-Maruyama | 50 | Cosine (0/5 seeds finite) | diverged" in md, md[:400]
        pr = S.read_csv(root / "table5_dpm" / "table5_dpm_paired.csv")
        assert all(r["mass_better"] == "True" for r in pr if r["sampler"] == "DPM-Solver++2M")
    print("All selected_setup checks passed.")


if __name__ == "__main__":
    main()
