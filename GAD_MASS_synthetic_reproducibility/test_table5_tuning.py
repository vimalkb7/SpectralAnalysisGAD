from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import table5_tuning as T  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def quality(ds: str, cid: str, steps: int, seed: int, split: str) -> float | None:
    method = T.GRID_BY_ID[cid]["method"]
    idx = [c["id"] for c in T.GRID if c["method"] == method].index(cid)
    if method == "Cosine":
        if cid != "cosine_gamma0.01_eps0.5" or steps < 2500:
            return None
        base = 0.30
    elif method == "MAS-SNR":
        base = 0.10 + 0.01 * abs(idx - 5)          
    else:
        if steps == 50 and not cid.endswith("eps0.01"):
            return None                              
        base = 0.12 + 0.01 * abs(idx - 2)           
    if ds == "block_covariance" and method == "MAS-SNR":
        base += 0.5                                  
    if split == "test":
        base += 0.001                                
    return base + 0.001 * seed


def fabricate_run(root: Path, phase: str, ds: str, cid: str, seed: int) -> None:
    task = {"phase": phase, "dataset": ds, "config_id": cid, "method": T.GRID_BY_ID[cid]["method"], "seed": seed}
    run = T.run_dir(root, task)
    run.mkdir(parents=True, exist_ok=True)
    delta_complete = not (cid == "iso_s0.05_eps0.01")          
    (run / "plan.json").write_text(json.dumps({"task": task, "delta_complete": delta_complete,
                                               "protocol_sha256": T.protocol_sha()}), encoding="utf-8")
    split = "val" if phase == "tune" else "test"
    rows = []
    for steps in T.EM_STEPS:
        q = quality(ds, cid, steps, seed, split)
        base = {"schedule": task["method"], "method": "em", "steps": steps, "evaluation_split": split}
        if q is None:
            rows.append({**base, "stable": False})
        else:
            rows.append({**base, "stable": True, **{m: q * (1 + 0.1 * i) for i, m in enumerate(T.METRICS)}})
    write_csv(run / "results_summary.csv", rows)
    (run / "DONE.json").write_text("{}", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        T.write_protocol(root)
        T.write_protocol(root)                       
        assert (root / "PROTOCOL.md").is_file()
        assert len(T.tune_tasks()) == 2 * len(T.GRID) * len(T.SEEDS) == 320

        try:
            T.select(root)
        except SystemExit as err:
            assert "not finished" in str(err)
        else:
            raise AssertionError("selection must wait for all tuning runs")

        for t in T.tune_tasks():
            fabricate_run(root, "tune", t["dataset"], t["config_id"], t["seed"])

        bad = T.run_dir(root, {"phase": "tune", "dataset": "stationary_gaussian", "config_id": T.GRID[0]["id"], "seed": 0})
        rows = T.read_csv(bad / "results_summary.csv")
        rows[0]["evaluation_split"] = "test"
        write_csv(bad / "results_summary.csv", rows)
        try:
            T.select(root)
        except SystemExit as err:
            assert "expected 'val'" in str(err), str(err)
        else:
            raise AssertionError("selection must only read validation rows")
        fabricate_run(root, "tune", "stationary_gaussian", T.GRID[0]["id"], 0)

        sel = json.loads(T.select(root).read_text(encoding="utf-8"))
        by = {(e["dataset"], e["method"], e["budget"]): e for e in sel["entries"]}
        mass_ids = [c["id"] for c in T.GRID if c["method"] == "MAS-SNR"]
        iso_ids = [c["id"] for c in T.GRID if c["method"] == "Isotropic-VP-Cosine"]
        assert by[("stationary_gaussian", "MAS-SNR", 50)]["config_id"] == mass_ids[5]
        assert by[("stationary_gaussian", "Isotropic-VP-Cosine", 2500)]["config_id"] == iso_ids[2]
        
        e50 = by[("stationary_gaussian", "Isotropic-VP-Cosine", 50)]
        assert e50["config_id"].endswith("eps0.01") and e50["config_id"] != "iso_s0.05_eps0.01"
        assert e50["num_eligible"] == 3, e50
        assert by[("block_covariance", "Cosine", 50)]["config_id"] is None
        assert by[("block_covariance", "Cosine", 2500)]["config_id"] == "cosine_gamma0.01_eps0.5"

        p = root / "selection" / "selection.json"
        original = p.read_text(encoding="utf-8")
        tampered = json.loads(original)
        tampered["entries"][0]["config_id"] = mass_ids[0]
        p.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            T.load_selection(root)
        except SystemExit as err:
            assert "modified" in str(err)
        else:
            raise AssertionError("an edited selection must be refused")
        p.write_text(original, encoding="utf-8")

        tt = T.test_tasks(root)
        assert len(tt) % len(T.SEEDS) == 0 and len(tt) > 0
        for t in tt:
            fabricate_run(root, "test", t["dataset"], t["config_id"], t["seed"])
        bundle = T.final(root)
        assert bundle.is_file()
        paired = T.read_csv(root / "final" / "table5_tuned_paired.csv")
        st = [r for r in paired if r["dataset"] == "stationary_gaussian" and r["baseline"] == "Isotropic-VP-Cosine"
              and r["mass_budget"] == "50" and r["baseline_budget"] == "50" and r["metric"] == "global_swd_mean"]
        assert st and st[0]["mass_better"] == "True" and st[0]["seeds_mass_better"] == "5", st
        bl = [r for r in paired if r["dataset"] == "block_covariance" and r["baseline"] == "Isotropic-VP-Cosine"
              and r["mass_budget"] == "50" and r["baseline_budget"] == "50" and r["metric"] == "global_swd_mean"]
        assert bl and bl[0]["mass_better"] == "False", bl
        md = (root / "final" / "TABLE5_TUNED.md").read_text(encoding="utf-8")
        assert "no delta-complete configuration" in md and "| block_covariance | Cosine" in md

    print("All table5_tuning checks passed.")


if __name__ == "__main__":
    main()
