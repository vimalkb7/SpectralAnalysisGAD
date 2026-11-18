from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts")); sys.path.insert(0, str(HERE))
import table4_tuning as T
import table5_from_table4 as V
import test_table4_tuning as TT


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); T.write_protocol(root)
        for t in T.tune_tasks():
            q = TT.quality(t["config_id"], t["seed"], "val")
            TT.fake_run(root, t, {} if q is None else {pt: q for pt in T.POINTS})
        T.select(root)
        level = {"MAS-SNR": 1.0, "FCPS": 1.2, "Isotropic-VP-Cosine": 1.4, "Cosine": 3.0}
        for t in T.test_tasks(root):
            rd = T.run_dir(root, t); q = level[t["schedule"]] * (1 + 0.01 * t["seed"])
            spec_rows = [{"mode": k, "lam_gamma": 0.5 * k, "band": "low" if k < 2 else ("high" if k >= 6 else "middle"),
                          "reference_power": 1 / (k + 1), "generated_power": q / (k + 1), "log10_power_ratio": 0.0} for k in range(8)]
            spec = rd / "generation_diagnostics" / f"{t['schedule']}_dpmpp2m_10_{6000 + t['seed']}_spectrum.csv"
            write(spec, spec_rows)
            rows = []
            for m, n in T.POINTS:
                rows.append({"schedule": t["schedule"], "method": m, "nfe": n, "stable": True, "evaluation_split": "test",
                             **{k: q * (i + 1) for i, k in enumerate(V.METRICS)},
                             "generation_spectrum_csv": "/elsewhere/" + spec.name})     # moved: exercises the fallback
            write(rd / "results_raw.csv", rows); (rd / "DONE.json").write_text("{}")
        bundle = V.build(root, "test", ("dpmpp2m", 10), ["MAS-SNR", "Isotropic-VP-Cosine", "Cosine"], root / "t5")
        assert bundle.is_file()
        md = (root / "t5" / "TABLE5_FROM_TABLE4.md").read_text()
        assert "| Stationary (broadband WSS) | MASS | **" in md and "FCPS" not in md.split("Paired")[0]
        paired = T.read_csv(root / "t5" / "table5_from_table4_paired.csv")
        assert len(paired) == 2 * 2 * 5 and all(r["mass_better"] == "True" for r in paired)
        by_mode = T.read_csv(root / "t5" / "figure2_spectral_power_by_mode.csv")
        assert len(by_mode) == 2 * 3 * 8 and all(r["num_seeds"] == "5" for r in by_mode)
        # a row scored on validation is refused
        p = T.run_dir(root, T.test_tasks(root)[0]) / "results_raw.csv"
        p.write_text(p.read_text().replace(",test,", ",val,"))
        try:
            V.build(root, "test", ("dpmpp2m", 10), ["MAS-SNR", "Isotropic-VP-Cosine"], root / "t5b")
        except SystemExit as err:
            assert "test split" in str(err)
        else:
            raise AssertionError("validation rows must be refused")
    print("All table5_from_table4 checks passed.")


if __name__ == "__main__":
    main()
