from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import table4_tuning as T


def fake_run(root: Path, t: dict, swd: dict) -> None:
    rd = T.run_dir(root, t); rd.mkdir(parents=True, exist_ok=True)
    split = "val" if t["phase"] == "tune" else "test"
    rows = [{"schedule": t["schedule"], "method": m, "nfe": n, "stable": swd.get((m, n)) is not None, "evaluation_split": split,
             "global_swd_mean": swd.get((m, n), "")} for m, n in T.POINTS]
    with (rd / "results_raw.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (rd / "DONE.json").write_text("{}")


def quality(cfg_id: str, seed: int, split: str):
    c = T.GRID_BY_ID[cfg_id]; p = c["params"]
    if c["schedule"] == "MAS-SNR":
        q = 0.06 + abs(p["delta0"] - 1e-2) + abs(p["gamma"] - 3e-3)
    elif c["schedule"] == "FCPS":
        q = 0.08 + (0.02 if p["fcps_cmin"] > 1e-3 else 0) + 0.003 * abs(p["fcps_alpha"] - 2)
    elif c["schedule"] == "Cosine":
        if p["cosine_terminal_epsilon"] < 1e-2 and seed == 3:
            return None                                  
        q = 0.2 + p["cosine_terminal_epsilon"]
    else:
        q = 0.09 + abs(p["iso_cosine_s"] - 0.02)
    return q + 0.001 * seed + (0.0005 if split == "test" else 0)


def main() -> None:
    assert len(T.GRID) == 48 and all(sum(c["schedule"] == s for c in T.GRID) == 12 for s in T.SCHEDULES)
    assert len(T.tune_tasks()) == 480
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); T.write_protocol(root); T.write_protocol(root)
        for t in T.tune_tasks():
            q = quality(t["config_id"], t["seed"], "val")
            fake_run(root, t, {} if q is None else {pt: q * (1.3 if pt[0] == "ddim" else 1) for pt in T.POINTS})
        sel = json.loads(T.select(root).read_text())
        by = {(e["dataset"], e["schedule"]): e for e in sel["entries"]}
        assert by[("stationary_gaussian", "MAS-SNR")]["config_id"] == "mass_gamma0.003_delta00.01"
        assert by[("block_covariance", "FCPS")]["config_id"].endswith("alpha2_cmin0.0001")
        assert by[("stationary_gaussian", "Cosine")]["config_id"].endswith("eps0.01")
        assert by[("stationary_gaussian", "Isotropic-VP-Cosine")]["config_id"].startswith("iso_s0.02_")
        t0 = T.tune_tasks()[0]; p = T.run_dir(root, t0) / "results_raw.csv"
        txt = p.read_text(); p.write_text(txt.replace(",val,", ",test,"))
        try:
            T.select(root)
        except SystemExit as err:
            assert "expected 'val'" in str(err)
        else:
            raise AssertionError("selection must read validation rows only")
        p.write_text(txt)
        tt = T.test_tasks(root); assert len(tt) == 40
        for t in tt:
            q = quality(t["config_id"], t["seed"], "test")
            fake_run(root, t, {pt: q * (1.3 if pt[0] == "ddim" else 1) for pt in T.POINTS})
        T.final(root)
        md = (root / "final" / "TABLE4_TUNED.md").read_text()
        assert "stationary_gaussian: MASS < FCPS" in md and "(b) FCPS vs MASS" in md
        a = T.driver_args({"phase": "test", "dataset": "block_covariance", "config_id": "fcps_gamma0.001_alpha2_cmin0.0001",
                           "schedule": "FCPS", "seed": 2}, "/p.npz")
        assert a[a.index("--evaluation_split") + 1] == "test" and a[a.index("--model_seed") + 1] == "3002"
        assert a[a.index("--fcps_cmin") + 1] == "0.0001" and abs(float(a[a.index("--scalar_cbar_T") + 1]) - 3453.877639491) < 1e-6
        dd = T.tasks_for("ddim", root)
        assert len(dd) == 40 and all(t["phase"] == "ddim" for t in dd)
        a = T.driver_args(dd[35], "/p.npz")
        assert a[a.index("--ddim_steps") + 1] == "10,25" and a[a.index("--evaluation_split") + 1] == "test"
        assert T.run_dir(root, dd[0]).parts[-4] == "ddim"
        for t in dd:
            q = quality(t["config_id"], t["seed"], "test")
            rd = T.run_dir(root, t); rd.mkdir(parents=True, exist_ok=True)
            rows = [{"schedule": t["schedule"], "method": m, "nfe": n, "stable": True, "evaluation_split": "test",
                     "global_swd_mean": q * (1.3 if m == "ddim" else 1)} for m, n in T.DDIM_POINTS]
            with (rd / "results_raw.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            (rd / "DONE.json").write_text("{}")
        T.final_ddim(root)
        md = (root / "ddim" / "TABLE4_DDIM.md").read_text()
        assert "| DDIM | 10 | MASS |" in md and "Reproduction check" in md
        assert T.protocol_sha() == json.loads((root / "protocol.json").read_text())["protocol_sha256"]
    print("All table4_tuning checks passed.")


if __name__ == "__main__":
    main()