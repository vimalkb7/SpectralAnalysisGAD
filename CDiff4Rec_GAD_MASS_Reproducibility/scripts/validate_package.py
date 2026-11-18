from __future__ import annotations

import csv
import hashlib
import py_compile
import shlex
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS_ZIP_SHA256 = "1507416860035e076a1ab53b8640d62caee7b5a4d9c057de7886a76ce0ece066"


def require(path: str) -> Path:
    target = ROOT / path
    if not target.exists():
        raise AssertionError(f"Missing required path: {path}")
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest() -> None:
    path = require("configs/published_runs.tsv")
    with path.open() as stream:
        rows = list(csv.DictReader((line for line in stream if not line.startswith("#dataset")), delimiter="\t", fieldnames=[
            "dataset", "seed", "graph_id", "scheduler", "objective", "r_min", "r_max", "steps",
            "sampling_steps", "inference_mode", "t_sampling", "low_time_prob", "low_time_max_step",
            "sigma", "alpha", "beta", "gamma", "topk", "topk_pseudo_user", "lr", "weight_decay",
            "r_agg", "f_agg", "training_pseudo", "batch_size", "epochs", "patience", "eval_interval",
        ]))
    assert len(rows) == 15, f"Expected 15 published runs, found {len(rows)}"
    by_dataset: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_dataset[row["dataset"]].add(int(row["seed"]))
        assert row["scheduler"] == "mas_snr"
        assert row["objective"] == "x0_mse"
        assert row["inference_mode"] == "clean_direct"
        assert row["training_pseudo"] == "0"
    assert {name: seeds for name, seeds in by_dataset.items()} == {
        "amazon_game": {1, 2, 3, 4, 5},
        "yelp": {1, 2, 3, 4, 5},
        "citeulike_t": {1, 2, 3, 4, 5},
    }


def validate_sources() -> None:
    python_files = [
        "main_gad_mas.py", "config_gad_mas.py", "data_utils.py", "evaluate_utils.py", "utils.py",
        "models/DNN.py", "models/Attention.py", "models/gad_mas_diffusion.py", "models/fcps_schedule.py",
        "tools/build_item_graph.py", "tools/fcps_calibration.py",
        "scripts/aggregate_results.py", "scripts/validate_package.py",
    ]
    for item in python_files:
        py_compile.compile(str(require(item)), doraise=True)
    for item in sorted((ROOT / "slurm").glob("*.sbatch")) + sorted((ROOT / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(item)], check=True)


def validate_sbatch_literals() -> None:
    expected = {
        "slurm/01_amazon_game_5seeds.sbatch": ["--array=0-4%3", "--r_min 1e-2", "--alpha 0.70", "--beta 0.15", "--gamma 0.15", "--topk 20", "--lr 5e-5", "--epochs 1000", "--patience 80"],
        "slurm/02_yelp_5seeds.sbatch": ["--array=0-4%5", "--r_min 1e-1", "--alpha 0.30", "--beta 0.10", "--gamma 0.60", "--topk 50", "--lr 1e-4", "--epochs 300", "--patience 40"],
        "slurm/03_citeulike_t_5seeds.sbatch": ["--array=0-4%2", "--r_min 3e-2", "--alpha 0.70", "--beta 0.10", "--gamma 0.20", "--topk 20", "--topk_pseudo_user 10000", "--epochs 1000", "--patience 100"],
    }
    common = ["--scheduler mas_snr", "--objective x0_mse", "--r_max 1e4", "--steps 50", "--sampling_steps 5", "--no-sampling_noise", "--inference_mode clean_direct", "--t_sampling low_time_mix", "--low_time_prob 0.70", "--low_time_max_step 3", "--batch_size 400", "--weight_decay 0.0", "--ur", "--uf", "--no-tf", "--tst_w_val"]
    for filename, fragments in expected.items():
        text = require(filename).read_text()
        for fragment in common + fragments:
            assert fragment in text, f"{filename}: missing exact parameter fragment {fragment!r}"


def _python_flags(text: str) -> dict[str, str]:
    """Flags of the main_gad_mas.py call in an SBATCH file, as {flag: value}."""
    start = text.index("python -u main_gad_mas.py")
    end = text.index("2>&1", start)
    tokens = shlex.split(text[start:end].replace("\\\n", " "))[3:]
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        assert key.startswith("--"), f"unexpected token {key!r}"
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            flags[key], i = tokens[i + 1], i + 2
        else:
            flags[key], i = "", i + 1
    return flags


def validate_fcps_launchers() -> None:
    twins = {
        "slurm/11_amazon_game_fcps_5seeds.sbatch": "slurm/01_amazon_game_5seeds.sbatch",
        "slurm/12_yelp_fcps_5seeds.sbatch": "slurm/02_yelp_5seeds.sbatch",
        "slurm/13_citeulike_t_fcps_5seeds.sbatch": "slurm/03_citeulike_t_5seeds.sbatch",
    }
    schedule_flags = {"--scheduler", "--fcps_alpha", "--fcps_calibration", "--fcps_cbar_T", "--fcps_cmin"}
    for fcps_file, mas_file in twins.items():
        fcps = _python_flags(require(fcps_file).read_text())
        mas = _python_flags(require(mas_file).read_text())
        assert fcps.get("--scheduler") == "fcps", f"{fcps_file}: scheduler is not fcps"
        assert fcps.get("--fcps_calibration") == "snr_matched", f"{fcps_file}: calibration is not snr_matched"
        shared = lambda flags: {k: v for k, v in flags.items() if k not in schedule_flags}
        differ = sorted(k for k in set(shared(fcps)) | set(shared(mas)) if shared(fcps).get(k) != shared(mas).get(k))
        assert not differ, f"{fcps_file} differs from {mas_file} in non-schedule flags: {differ}"
        sbatch = lambda path: [l for l in require(path).read_text().splitlines()
                               if l.startswith("#SBATCH") and "--job-name" not in l]
        assert sbatch(fcps_file) == sbatch(mas_file), f"{fcps_file}: SBATCH resources differ from {mas_file}"


def validate_data() -> None:
    archive = require("datasets.zip")
    assert sha256(archive) == DATASETS_ZIP_SHA256, "datasets.zip checksum mismatch"
    with zipfile.ZipFile(archive) as bundle:
        names = Counter(bundle.namelist())
    expected = {
        f"datasets/{dataset}/{name}"
        for dataset in ("amazon_game", "yelp", "citeulike_t")
        for name in ("train_list.npy", "valid_list.npy", "test_list.npy", "pseudo_data.pickle")
    }
    assert expected.issubset(names), f"datasets.zip is missing: {sorted(expected - names.keys())}"
    if (ROOT / "datasets").is_dir():
        subprocess.run(["sha256sum", "-c", "DATA_CHECKSUMS.sha256"], cwd=ROOT, check=True)


def main() -> None:
    for item in ("README.md", "NOTICE.md", "environment.yml", "configs/graph_assets.tsv", "configs/expected_results.json"):
        require(item)
    validate_manifest()
    validate_sources()
    validate_sbatch_literals()
    validate_fcps_launchers()
    validate_data()
    print("Package validation passed: 15 published runs, exact SBATCH parameters, FCPS launchers matching "
          "their MASS twins, source syntax, and data integrity.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.CalledProcessError, OSError, zipfile.BadZipFile) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)

