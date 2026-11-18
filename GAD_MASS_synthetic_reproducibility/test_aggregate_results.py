from __future__ import annotations

import csv
import json
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from aggregate_results import (  # noqa: E402
    METRICS,
    aggregate_figure2,
    aggregate_table5,
    parse_operating_points,
)

DATASETS = ("stationary_gaussian", "block_covariance")
NUM_SEEDS = 5
POINTS = (("Cosine", "em", 2500), ("Cosine", "em", 50), ("MAS-SNR", "em", 50), ("Isotropic-VP-Cosine", "em", 50))
N_MODES = 16


def write_csv(path: Path, rows: list[dict]) -> None:
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


def fabricate(root: Path, num_seeds: int = NUM_SEEDS) -> None:
    rng = random.Random(0)
    for dataset in DATASETS:
        for seed in range(num_seeds):
            run = root / "table5_figure2" / dataset / f"seed_{seed}"
            summary: list[dict] = []
            fig2: list[dict] = []
            for schedule, method, steps in POINTS:
                base = {"dataset": dataset, "seed": seed, "schedule": schedule, "method": method, "steps": steps, "nfe": steps}
                unstable = schedule == "Cosine" and steps == 50  # the launcher records this instead of aborting
                if unstable:
                    row = {**base, "stable": False, "failure_reason": "Non-finite reverse state"}
                    row.update({m: float("nan") for m in METRICS})
                    summary.append(row)
                    continue
                row = {**base, "stable": True, "failure_reason": ""}
                row.update({m: rng.uniform(0.1, 1.0) for m in METRICS})
                summary.append(row)
                for mode in range(N_MODES):
                    lam = 1e-3 + 7.0 * mode / (N_MODES - 1) + 0.01 * rng.random()
                    ref = 1.0 / lam
                    fig2.append(
                        {
                            **base,
                            "mode": mode,
                            "lam_gamma": lam,
                            "band": "low" if mode < 4 else ("high" if mode >= 12 else "middle"),
                            "reference_power": ref,
                            "generated_power": ref * rng.uniform(0.8, 1.2),
                            "log10_power_ratio": 0.0,
                        }
                    )
            write_csv(run / "results_summary.csv", summary)
            write_csv(run / "figure2_spectral_power_long.csv", fig2)


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    assert parse_operating_points("Cosine:2500, MAS-SNR:em:50") == [("Cosine", "em", 2500), ("MAS-SNR", "em", 50)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fabricate(root)

        failed = False
        try:
            aggregate_table5(root, operating_points="Cosine:2500,Cosine:50,MAS-SNR:50")
        except RuntimeError as err:
            failed = True
            assert "Cosine/em/50" in str(err) and "unstable" in str(err), str(err)
        assert failed, "expected the unstable paper row to be rejected"
        sweep = read(root / "table5_figure2" / "table5_all_operating_points.csv")
        by_key = {(r["dataset"], r["schedule"], r["method"], int(r["nfe"])): r for r in sweep}
        assert len(by_key) == len(DATASETS) * len(POINTS)
        assert by_key[("stationary_gaussian", "Cosine", "em", 50)]["num_seeds_stable"] == "0"
        assert by_key[("stationary_gaussian", "Cosine", "em", 50)]["global_swd_mean_mean"] == ""
        assert by_key[("block_covariance", "MAS-SNR", "em", 50)]["num_seeds_stable"] == str(NUM_SEEDS)

        try:
            aggregate_table5(root, operating_points="Cosine:2500,FCPS:2500")
        except RuntimeError as err:
            assert "FCPS/em/2500" in str(err), str(err)
        else:
            raise AssertionError("expected the missing FCPS point to be rejected")

        out = aggregate_table5(root, operating_points="Cosine:2500,MAS-SNR:50,Isotropic-VP-Cosine:50")
        rows = read(out)
        assert len(rows) == len(DATASETS) * 3, len(rows)
        for row in rows:
            assert row["num_seeds"] == str(NUM_SEEDS) and row["num_seeds_stable"] == str(NUM_SEEDS)
            for metric in METRICS:
                assert math.isfinite(float(row[f"{metric}_mean"])) and math.isfinite(float(row[f"{metric}_std"]))

        out = aggregate_table5(root, operating_points="Cosine:50", allow_unstable=True)
        rows = read(out)
        assert len(rows) == len(DATASETS) and all(r["num_seeds_stable"] == "0" for r in rows)

        long_out = aggregate_figure2(root, operating_points="Cosine:2500,MAS-SNR:50,Isotropic-VP-Cosine:50")
        long_rows = read(long_out)
        assert len(long_rows) == len(DATASETS) * NUM_SEEDS * 3 * N_MODES, len(long_rows)
        assert set(long_rows[0].keys()) >= {"dataset", "seed", "schedule", "method", "steps", "mode", "lam_gamma", "band", "reference_power", "generated_power"}
        by_mode = read(root / "table5_figure2" / "figure2_spectral_power_by_mode.csv")
        assert len(by_mode) == len(DATASETS) * 3 * N_MODES, len(by_mode)
        assert all(r["num_seeds"] == str(NUM_SEEDS) for r in by_mode)
        manifest = json.loads((root / "table5_figure2" / "figure2_manifest.json").read_text(encoding="utf-8"))
        assert {tuple(p.values()) for p in manifest["operating_points"]} == {("Cosine", "em", 2500), ("MAS-SNR", "em", 50), ("Isotropic-VP-Cosine", "em", 50)}
        assert "band" in manifest["columns"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fabricate(root, num_seeds=3)
        try:
            aggregate_table5(root, operating_points="MAS-SNR:50")
        except RuntimeError as err:
            assert "5 seeds" in str(err), str(err)
        else:
            raise AssertionError("a 3-seed tree must fail the default 5-seed check")
        assert read(aggregate_table5(root, operating_points="MAS-SNR:50", num_seeds=3))[0]["num_seeds"] == "3"
        fabricate(root, num_seeds=5)
        (root / "table5_figure2" / "block_covariance" / "seed_4" / "results_summary.csv").rename(
            root / "table5_figure2" / "block_covariance" / "seed_4" / "moved.csv")
        extra = root / "table5_figure2" / "stationary_gaussian" / "seed_7"
        extra.mkdir()
        (extra / "results_summary.csv").write_bytes(
            (root / "table5_figure2" / "stationary_gaussian" / "seed_0" / "results_summary.csv").read_bytes())
        try:
            aggregate_table5(root, operating_points="MAS-SNR:50")
        except RuntimeError as err:
            assert "expected seeds [0, 1, 2, 3, 4]" in str(err), str(err)
        else:
            raise AssertionError("a seed gap must be reported")

    print("All aggregate_results checks passed.")


if __name__ == "__main__":
    main()
