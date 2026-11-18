from __future__ import annotations

import argparse
import configparser
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch.optim import Adam

from fence_gad_mas.build import build_model_and_data
from fence_gad_mas.data import build_exp34_clean_loaders
from fence_gad_mas.metrics import evaluate_sampling_metrics
from fence_gad_mas.graph import load_graph_for_fence
from fence_gad_mas.schedules import (
    FCPS_GRIDS,
    FCPS_HORIZON_MODES,
    SCHEDULE_KINDS,
    build_schedule,
    schedule_stiffness_report,
)
from fence_gad_mas.spectral_power import estimate_effective_spectral_power, load_or_estimate_spectral_power


DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "PEMS04": {
        "phase1_epoch": 150,
        "phase2_epoch": 150,
        "batch_size": 64,
        "snapshot_epochs": "50,60,70,80,90,100,110,120,130,140,150",
        "gamma": 0.03,
        "r_min": 5e-5,
        "n_clusters": 15,
        "pi": 0.60,
        "max_guidance": 1.20,
        "graph_relpath": "exp1_k10_alpha025/PEMS04_hybrid_k10_alpha0p25_union/PEMS04_hybrid_k10_alpha0p25_union_adj.npy",
        "metric_note": "Table row was obtained from an Exp4-style checkpoint; validate snapshots before locked test.",
    },
    "PEMS08": {
        "phase1_epoch": 150,
        "phase2_epoch": 120,
        "batch_size": 64,
        "snapshot_epochs": "50,60,70,80,90,100,110,120",
        "gamma": 0.05,
        "r_min": 3e-5,
        "n_clusters": 8,
        "pi": 0.60,
        "max_guidance": 1.20,
        "graph_relpath": "exp1_k10_alpha025/PEMS08_hybrid_k10_alpha0p25_union/PEMS08_hybrid_k10_alpha0p25_union_adj.npy",
        "metric_note": "Exp35 clean end-to-end SC-TC configuration.",
    },
}


BASELINE_TARGETS = {
    "PEMS04": {"MAE": 27.31, "RMSE": 44.28, "MAPE": 0.180},
    "PEMS08": {"MAE": 27.29, "RMSE": 47.78, "MAPE": 0.175},
}
TARGET_NUMERICAL_MARGIN = 0.03

def baseline_ratio_fields(dataset: str, row: dict[str, Any]) -> dict[str, Any]:
    b = BASELINE_TARGETS[dataset.upper()]
    ratios = {m: float(row[m]) / float(b[m]) for m in ("MAE", "RMSE", "MAPE")}
    return {
        "Ratio_MAE": ratios["MAE"],
        "Ratio_RMSE": ratios["RMSE"],
        "Ratio_MAPE": ratios["MAPE"],
        "Baseline_Max_Ratio": max(ratios.values()),
        "Baseline_Mean_Ratio": sum(ratios.values()) / 3.0,
        "Beats_FENCE_All": int(max(ratios.values()) < 1.0),
        "Clears_3pct_Margin_All": int(max(ratios.values()) <= 1.0 - TARGET_NUMERICAL_MARGIN),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(path: str | Path | None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=handlers,
        force=True,
    )


def resolve_dataset_root(data_root: str | Path, dataset: str) -> Path:
    root = Path(data_root).expanduser().resolve()
    if root.name == dataset:
        dataset_root = root
    else:
        dataset_root = root / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}. "
            f"Pass --data-root as either the parent containing {dataset}/ or the {dataset}/ directory itself."
        )
    return dataset_root


def comma_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def make_config(
    dataset: str,
    data_root: str | Path,
    out_root: str | Path,
    *,
    phase1_epoch: int | None = None,
    phase2_epoch: int | None = None,
    batch_size: int | None = None,
    pi: float | None = None,
    max_guidance: float | None = None,
    n_clusters: int | None = None,
    gamma: float | None = None,
    r_min: float | None = None,
    train_levels: int = 50,
    sample_nfe: int = 12,
    nsample: int = 10,
    snapshot_epochs: str | None = None,
    graph_path: str | None = None,
    spectral_power_path: str | None = None,
    guidance: str = "cov_fbg",
    power_mode: str = "clean_residual_blend",
    power_blend: float = 0.5,
    residual_alpha: float = 0.75,
    low_nfe_epoch: int = 40,
    aggregation: str = "median",
    posterior_cluster_average: bool = False,
    final_x0_projection: bool = True,
    schedule: str = "mass",
    fcps_horizon: str = "match_mass_power",
    fcps_cbar_T: float | None = None,
    fcps_delta: float = 1e-3,
    fcps_alpha: float = 4.0,
    fcps_c_min: float = 0.1,
    fcps_grid: str = "time",
) -> configparser.ConfigParser:
    dataset = dataset.upper()
    schedule = str(schedule).lower()
    if schedule not in SCHEDULE_KINDS:
        raise ValueError(f"Unsupported schedule={schedule}; expected one of {SCHEDULE_KINDS}")
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset={dataset}; expected one of {sorted(DATASET_DEFAULTS)}")
    d = dict(DATASET_DEFAULTS[dataset])
    dataset_root = resolve_dataset_root(data_root, dataset)
    out_root = Path(out_root).expanduser().resolve()

    graph = Path(graph_path).expanduser().resolve() if graph_path else dataset_root / d["graph_relpath"]
    data_prefix = dataset_root / "miss_data"
    if spectral_power_path:
        spectral_path = Path(spectral_power_path).expanduser().resolve()
    else:
        gamma_tag = str(gamma if gamma is not None else d["gamma"]).replace("0.", "").replace(".", "p")
        power_tag = str(power_mode).lower().replace(".", "p")
        if power_tag == "blend":
            power_tag += f"{float(power_blend):.2f}".replace(".", "p")
        if power_tag == "clean_residual_blend":
            power_tag += "_alpha" + f"{float(residual_alpha):.2f}".replace(".", "p")
        spectral_path = out_root / dataset.lower() / "shared" / f"spectral_power_{dataset}_{power_tag}_TRAINONLY_SC_TC_G1_normlap_gamma{gamma_tag}_gad_mass_fence_v3.npz"

    cfg = configparser.ConfigParser()
    cfg["file"] = {
        "data_prefix": str(data_prefix),
    }
    cfg["train"] = {
        "miss_rate": "0.8",
        "type": "SC-TC",
        "phase1_epoch": str(phase1_epoch if phase1_epoch is not None else d["phase1_epoch"]),
        "phase2_epoch": str(phase2_epoch if phase2_epoch is not None else d["phase2_epoch"]),
        "phase1_lr": "2e-3",
        "phase2_lr": "1e-3",
        "batch_size": str(batch_size if batch_size is not None else d["batch_size"]),
        "use_nni": "0",
        "test_ratio": "0.2",
        "val_ratio": "0.2",
        "sample_len": "12",
        "valid_epoch_interval": "5",
        "lr_scheduler": "multistep",
        "lr_milestone1": "0.75",
        "lr_milestone2": "0.90",
        "lr_gamma": "0.1",
        "lr_factor": "0.5",
        "lr_patience": "5",
        "lr_min": "1e-7",
        "phase1_patience": "10",
        "phase2_patience": "10",
        "low_nfe_epoch": str(int(low_nfe_epoch)),
        "low_nfe_lr": "2e-4",
        "low_nfe_patience": "7",
        "low_nfe_valid_interval": "5",
        "low_nfe_snapshot_interval": "5",
        "phase1_grad_clip": "5.0",
        "phase2_grad_clip": "1.0",
        "phase1_weight_decay": "1e-6",
        "phase2_weight_decay": "1e-5",
        "train_node_embeddings": "0",
        "phase1_training_loader": "clean_full_train",
        "phase2_training_loader": "clean_target_train_only",
        "phase2_mask_strategy": "exact_stored_sc_tc",
        "clean_target_protocol": "train_split_only_no_val_or_test_targets_sc_tc",
        "disable_early_stopping": "0",
        "snapshot_epochs": snapshot_epochs if snapshot_epochs is not None else d["snapshot_epochs"],
        "sample_valid_enabled": "1",
        "sample_valid_interval": "10",
        "sample_valid_batches": "10",
        "sample_valid_nsample": "5",
        "sample_valid_start_epoch": "50",
        "sample_valid_mape_eps": "1e-4",
        "sample_valid_save_metrics": "RMSE,MAE,MAPE",
        "sample_valid_aggregation": str(aggregation),
        "full_validation_aggregation": str(aggregation),
        "full_validation_nsample": "10",
        "save_sampling_snapshots": "1",
    }
    cfg["diffusion"] = {
        "layers": "4",
        "channels": "64",
        "nheads": "8",
        "diffusion_embedding_dim": "64",
        # Legacy num_steps is kept equal to train_levels for old tooling, but
        # the model uses train_levels and sample_nfe independently.
        "num_steps": str(train_levels),
        "train_levels": str(train_levels),
        "sample_nfe": str(sample_nfe),
        "validation_levels": "16",
        "continuous_time_training": "1",
        "sample_grid_power": "1.0",
        "continuous_embedding_max_frequency": "10000",
        "nsample": str(nsample),
        "beta_start": "0.0001",
        "beta_end": "0.5",
        "schedule": "quad",
    }
    cfg["model"] = {
        "timeemb": "128",
        "featureemb": "16",
        "target_strategy": "mix",
        "p_drop": "0.1",
        "cfg_scale": "1.0",
        "guidance": str(guidance),
    }
    cfg["fbg"] = {
        "mode": "cluster",
        "pi": str(pi if pi is not None else d["pi"]),
        "max_guidance": str(max_guidance if max_guidance is not None else d["max_guidance"]),
        "constant_guidance": "1.0",
        "t0": "0.8",
        "t1": "0.5",
        "n_clusters": str(n_clusters if n_clusters is not None else d["n_clusters"]),
        "temp_scale": "1.0",
        "logp_delta_clip": "0.0",
    }
    cfg["gad"] = {
        "graph_path": str(graph),
        "adj_sigma2": "0.1",
        "adj_threshold": "0.0",
        "normalized_laplacian": "1",
        "gamma": str(gamma if gamma is not None else d["gamma"]),
        "sigma": "1.0",
        "T_diff": "1.0",
        "t_eps": "1e-3",
        "r_max": "1e3",
        "r_min": str(r_min if r_min is not None else d["r_min"]),
        "spectral_power_path": str(spectral_path),
        "spectral_power_min": "1e-6",
        "spectral_power_max_batches": "0",
        "spectral_power_source": "train_only_clean_and_sc_tc_context_residual",
        "spectral_power_mode": str(power_mode),
        "spectral_power_clean_mode": "centered",
        "spectral_power_blend": str(float(power_blend)),
        "spectral_power_residual_alpha": str(float(residual_alpha)),
        "residual_proxy_spatial_weight": "0.65",
        "residual_proxy_temporal_weight": "0.35",
        "spectral_power_force_precompute": "1",
        "objective": "x0",
        "x0_loss_weight": "0.0",
        "loss_mse_weight": "1.0",
        "loss_huber_weight": "0.25",
        "loss_relative_weight": "0.05",
        "loss_spectral_weight": "0.05",
        "huber_delta_physical": "10.0",
        "relative_eps_physical": "1e-4",
        "low_nfe_budgets": "6,8,12,20,50",
        "low_nfe_probs": "0.15,0.30,0.30,0.15,0.10",
        "sampler": "em",
        "heun_stochastic": "0",
        "clamp_observed": "1",
        "score_clip": "20.0",
        "posterior_cluster_average": "1" if posterior_cluster_average else "0",
        "final_x0_projection": "1" if final_x0_projection else "0",
        # Forward schedule. The spectral power above is estimated identically
        # for both schedules; FCPS uses it only for its horizon calibration and
        # the shared spectral loss weights, never in its channel.
        "schedule": schedule,
    }
    if schedule == "fcps":
        cfg["gad"].update({
            "fcps_horizon": str(fcps_horizon),
            "fcps_cbar_T": "" if fcps_cbar_T is None else str(float(fcps_cbar_T)),
            "fcps_delta": str(float(fcps_delta)),
            "fcps_alpha": str(float(fcps_alpha)),
            "fcps_c_min": str(float(fcps_c_min)),
            "fcps_grid": str(fcps_grid),
        })
    return cfg


def write_config(cfg: configparser.ConfigParser, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        cfg.write(f)
    return path


def validate_config_files(cfg: configparser.ConfigParser) -> None:
    data_prefix = Path(cfg["file"]["data_prefix"])
    miss_type = cfg["train"]["type"]
    miss_rate = cfg["train"]["miss_rate"]
    needed = [
        data_prefix / f"true_data_{miss_type}_{miss_rate}.npz",
        data_prefix / f"miss_data_{miss_type}_{miss_rate}.npz",
        Path(cfg["gad"]["graph_path"]),
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: configparser.SectionProxy, epochs: int):
    if cfg.get("lr_scheduler", "multistep").lower() != "multistep":
        raise ValueError("This clean reproduction expects lr_scheduler=multistep.")
    milestones = [
        int(float(cfg.get("lr_milestone1", "0.75")) * epochs),
        int(float(cfg.get("lr_milestone2", "0.90")) * epochs),
    ]
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(cfg.get("lr_gamma", "0.1")),
    )


def install_exact_clean_training_forward(model) -> None:
    def clean_forward(model_self, batch, is_train=1, is_phase1=True):
        observed_data, observed_mask, observed_tp, gt_mask, _, _ = model_self.process_data(batch, is_train)
        if not torch.allclose(observed_mask, torch.ones_like(observed_mask)):
            raise RuntimeError(
                "Clean SC-TC protocol requires observed_mask=1 for clean full-signal training; "
                f"found mean={float(observed_mask.mean().item())}."
            )
        cond_mask = torch.zeros_like(observed_mask) if is_phase1 else gt_mask
        if not is_phase1 and (observed_mask - cond_mask).abs().sum().item() <= 0:
            raise RuntimeError("Conditional SC-TC batch has zero reconstruction targets.")
        side_info = model_self.get_side_info(observed_tp, cond_mask)
        loss_fn = model_self.calc_loss if is_train == 1 else model_self.calc_loss_valid
        return loss_fn(observed_data, cond_mask, observed_mask, side_info, is_train, is_phase1)

    model.forward = MethodType(clean_forward, model)


def write_history_header(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "phase", "epoch", "train_loss", "val_loss", "screening_rmse_mean",
            "screening_mae_mean", "screening_mape_mean", "screening_batches",
            "screening_nsample", "lr", "epoch_time_sec", "elapsed_time_sec",
        ])


def append_history(path: str | Path, row: list[Any]) -> None:
    with Path(path).open("a", newline="") as f:
        csv.writer(f).writerow(row)


def save_combined(model, path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": metadata,
        "diffmodel_uncond": model.diffmodel_uncond.state_dict(),
        "diffmodel_cond": model.diffmodel_cond.state_dict(),
        "embed_layer": model.embed_layer.state_dict(),
        "s_k": model.s_k.detach().cpu(),
        "lam_gamma": model.lam_gamma.detach().cpu(),
        "schedule": model.schedule_state(),
    }, path)


def check_checkpoint_schedule(model, checkpoint: dict[str, Any], checkpoint_path: str | Path) -> None:
    """Refuse to evaluate a checkpoint under a different forward schedule.

    Checkpoints written before the FCPS option existed (including every
    published Table VII checkpoint) carry no schedule record and are MASS.
    """
    mine = model.schedule_state()
    theirs = checkpoint.get("schedule")
    if theirs is None:
        if mine["kind"] != "mass":
            raise ValueError(
                f"Checkpoint {checkpoint_path} predates the schedule option and is MASS, "
                f"but the evaluation config selects schedule={mine['kind']}."
            )
        return
    if theirs.get("kind") != mine.get("kind"):
        raise ValueError(
            f"Checkpoint schedule={theirs.get('kind')} but config schedule={mine.get('kind')}: {checkpoint_path}"
        )
    for key, value in mine.items():
        if key == "calibration":
            continue
        other = theirs.get(key)
        if isinstance(value, float):
            if other is None or not np.isclose(float(other), value, rtol=1e-6, atol=1e-12):
                raise ValueError(f"Checkpoint schedule {key}={other} differs from config {key}={value}: {checkpoint_path}")
        elif other != value:
            raise ValueError(f"Checkpoint schedule {key}={other!r} differs from config {key}={value!r}: {checkpoint_path}")


def full_validation_loss(model, loader, is_phase1: bool) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values.append(float(model(batch, is_train=0, is_phase1=is_phase1).item()))
    return float(np.mean(values))


def ensure_spectral_power(cfg: configparser.ConfigParser, device: str, force: bool = False) -> Path:
    spectral_path = Path(cfg["gad"]["spectral_power_path"])
    if spectral_path.exists() and not force:
        logging.info("Using existing train-only effective spectral-power cache: %s", spectral_path)
        return spectral_path
    logging.info("Preparing train-only clean/context-residual spectral-power cache: %s", spectral_path)
    loaders = build_exp34_clean_loaders(cfg)
    graph = load_graph_for_fence(cfg, target_dim=loaders["target_dim"], device=device, dtype=torch.float32)
    gad = cfg["gad"]
    spectral_power, parts = estimate_effective_spectral_power(
        graph,
        loaders["spectral_train"],
        loaders["phase2_train"],
        device,
        mode=gad.get("spectral_power_mode", "clean_residual_blend"),
        min_power=float(gad.get("spectral_power_min", "1e-6")),
        max_batches=int(gad.get("spectral_power_max_batches", "0")),
        clean_mode=gad.get("spectral_power_clean_mode", "centered"),
        clean_blend=float(gad.get("spectral_power_blend", "0.5")),
        residual_alpha=float(gad.get("spectral_power_residual_alpha", "0.75")),
        residual_spatial_weight=float(gad.get("residual_proxy_spatial_weight", "0.65")),
        residual_temporal_weight=float(gad.get("residual_proxy_temporal_weight", "0.35")),
    )
    spectral_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = spectral_path.with_suffix(spectral_path.suffix + ".tmp.npz")
    payload = {
        "s_k": spectral_power,
        "source": np.asarray("train_only_clean_and_sc_tc_context_residual"),
        "mode": np.asarray(gad.get("spectral_power_mode", "clean_residual_blend")),
        "residual_alpha": np.asarray(float(gad.get("spectral_power_residual_alpha", "0.75"))),
    }
    for name, arr in parts.items():
        payload[f"{name}_power"] = arr
    np.savez(temp_path, **payload)
    os.replace(temp_path, spectral_path)
    logging.info(
        "Saved effective spectral cache modes=%d range=[%.6e, %.6e] -> %s",
        len(spectral_power), float(spectral_power.min()), float(spectral_power.max()), spectral_path,
    )
    return spectral_path


def load_checkpoint_into_model(model, checkpoint_path: str | Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key in ("diffmodel_uncond", "diffmodel_cond", "embed_layer"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint missing key={key}: {checkpoint_path}")
    if "s_k" in checkpoint:
        ck_s = torch.as_tensor(checkpoint["s_k"], device=model.s_k.device, dtype=model.s_k.dtype)
        if ck_s.shape != model.s_k.shape or not torch.allclose(ck_s, model.s_k, rtol=1e-5, atol=1e-7):
            raise ValueError(
                "Checkpoint spectral power does not match the evaluation config/cache. "
                "Use the checkpoint's original config and spectral-power file; do not silently mix schedules."
            )
    if "lam_gamma" in checkpoint:
        ck_lam = torch.as_tensor(checkpoint["lam_gamma"], device=model.lam_gamma.device, dtype=model.lam_gamma.dtype)
        if ck_lam.shape != model.lam_gamma.shape or not torch.allclose(ck_lam, model.lam_gamma, rtol=1e-5, atol=1e-7):
            raise ValueError(
                "Checkpoint shifted spectrum does not match the evaluation graph/gamma. "
                "Use the original graph and gamma."
            )
    check_checkpoint_schedule(model, checkpoint, checkpoint_path)
    model.diffmodel_uncond.load_state_dict(checkpoint["diffmodel_uncond"])
    model.diffmodel_cond.load_state_dict(checkpoint["diffmodel_cond"])
    model.embed_layer.load_state_dict(checkpoint["embed_layer"])
    return checkpoint


def evaluate_checkpoint(
    cfg: configparser.ConfigParser,
    checkpoint_path: str | Path,
    *,
    split: str,
    device: str,
    eval_seed: int,
    trial_name: str,
    results_file: str | Path,
    nsample: int = 10,
    max_batches: int | None = None,
) -> dict[str, Any]:
    set_seed(eval_seed)
    model, loaders, _ = build_model_and_data(cfg, device)
    checkpoint = load_checkpoint_into_model(model, checkpoint_path, device)
    loader = loaders["valid_eval"] if split == "validation" else loaders["test_eval"]
    metadata = loaders["metadata"]
    aggregation = cfg["train"].get("full_validation_aggregation", "median")
    metrics = evaluate_sampling_metrics(
        model,
        loader,
        metadata.train_std,
        metadata.train_mean,
        nsample=nsample,
        max_batches=max_batches,
        aggregation=aggregation,
        fence_compatible_mape=True,
        desc=f"{split}-{aggregation}{nsample}-{trial_name}",
    )
    dataset_name = "PEMS04" if "PEMS04" in cfg["file"]["data_prefix"].upper() else "PEMS08"
    row = {
        "Trial": trial_name,
        "Dataset": dataset_name,
        "Split": split,
        "Sample_NFE": int(cfg["diffusion"].get("sample_nfe", cfg["diffusion"].get("num_steps", 50))),
        "Train_Levels": int(cfg["diffusion"].get("train_levels", cfg["diffusion"].get("num_steps", 50))),
        "Aggregation": metrics["aggregation"],
        "NSample": nsample,
        "Normalization": "train_only",
        "Eval_Seed": eval_seed,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "Batches": metrics["batches"],
        "Eval_Points": metrics["eval_points"],
        "Checkpoint": str(checkpoint_path),
        "Checkpoint_Metadata": repr(checkpoint.get("metadata", {})),
        "Guidance": cfg["model"].get("guidance", "none"),
        "FBG_Mode": cfg["fbg"].get("mode", ""),
        "Pi": cfg["fbg"].get("pi", ""),
        "Max_Guidance": cfg["fbg"].get("max_guidance", ""),
        "N_Clusters": cfg["fbg"].get("n_clusters", ""),
        "T0": cfg["fbg"].get("t0", ""),
        "T1": cfg["fbg"].get("t1", ""),
        "Temp_Scale": cfg["fbg"].get("temp_scale", "1.0"),
        "Cluster_Average": cfg["gad"].get("posterior_cluster_average", "0"),
        "Final_X0_Projection": cfg["gad"].get("final_x0_projection", "1"),
        "Power_Mode": cfg["gad"].get("spectral_power_mode", "clean_residual_blend"),
        "Residual_Alpha": cfg["gad"].get("spectral_power_residual_alpha", ""),
    }
    row.update(baseline_ratio_fields(dataset_name, row))
    # Appended after the historical columns so existing readers are unaffected.
    sched = model.schedule_state()
    row["Schedule"] = sched["kind"]
    row["FCPS_cbar_T"] = sched.get("cbar_T", "")
    row["FCPS_Alpha"] = sched.get("alpha", "")
    row["FCPS_C_Min"] = sched.get("c_min", "")
    row["FCPS_Grid"] = sched.get("grid", "")
    row["Nonfinite_Predictions"] = int(metrics.get("nonfinite_predictions", 0))
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    logging.info("Saved %s result: %s", split, results_file)
    if split == "test":
        logging.warning("HELD-OUT TEST WAS EVALUATED. Do this only after validation-based checkpoint selection.")
    return row


def train(args: argparse.Namespace) -> None:
    dataset = args.dataset.upper()
    schedule_tag = str(args.schedule).upper()
    run_name = args.run_name or f"{dataset}_SC_TC_GAD_{schedule_tag}_V3_{args.power_mode}_{args.guidance}_seed{args.seed}"
    run_dir = Path(args.out_root).expanduser().resolve() / dataset.lower() / run_name
    setup_logging(run_dir / "logs" / f"{run_name}.train.log")
    set_seed(args.seed)

    cfg = make_config(
        dataset,
        args.data_root,
        args.out_root,
        phase1_epoch=args.phase1_epoch,
        phase2_epoch=args.phase2_epoch,
        batch_size=args.batch_size,
        pi=args.pi,
        max_guidance=args.max_guidance,
        n_clusters=args.n_clusters,
        gamma=args.gamma,
        r_min=args.r_min,
        train_levels=args.train_levels,
        sample_nfe=args.sample_nfe,
        nsample=args.nsample,
        snapshot_epochs=args.snapshot_epochs,
        graph_path=args.graph_path,
        spectral_power_path=args.spectral_power_path,
        guidance=args.guidance,
        power_mode=args.power_mode,
        power_blend=args.power_blend,
        residual_alpha=args.residual_alpha,
        low_nfe_epoch=args.low_nfe_epoch,
        aggregation=args.aggregation,
        posterior_cluster_average=args.posterior_cluster_average,
        final_x0_projection=not args.no_final_x0_projection,
        schedule=args.schedule,
        fcps_horizon=args.fcps_horizon,
        fcps_cbar_T=args.fcps_cbar_T,
        fcps_delta=args.fcps_delta,
        fcps_alpha=args.fcps_alpha,
        fcps_c_min=args.fcps_c_min,
        fcps_grid=args.fcps_grid,
    )
    validate_config_files(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_config(cfg, run_dir / "config.ini")
    logging.info("Saved v3 run config: %s", config_path)
    ensure_spectral_power(cfg, args.device, force=args.force_spectral_power)

    tcfg = cfg["train"]
    required = {
        "phase1_training_loader": "clean_full_train",
        "phase2_training_loader": "clean_target_train_only",
        "phase2_mask_strategy": "exact_stored_sc_tc",
        "disable_early_stopping": "0",
        "train_node_embeddings": "0",
    }
    for key, expected in required.items():
        if tcfg.get(key) != expected:
            raise ValueError(f"Config mismatch: {key}={tcfg.get(key)!r}; expected {expected!r}")
    if cfg["gad"].get("spectral_power_source") != "train_only_clean_and_sc_tc_context_residual":
        raise ValueError("v3 spectral power must use train-only clean/context-residual statistics.")

    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "results" / f"{run_name}_history.csv"
    summary_path = run_dir / "results" / f"{run_name}_training_summary.json"
    write_history_header(history_path)

    model, loaders, spectral_power = build_model_and_data(cfg, args.device)
    install_exact_clean_training_forward(model)
    metadata = loaders["metadata"]
    logging.info("dataset=%s seed=%d metadata=%s", dataset, args.seed, metadata)
    logging.info("spectral_power range=[%.6e, %.6e]", float(spectral_power.min()), float(spectral_power.max()))
    logging.info(
        "continuous_logsnr=1 train_levels=%d initial_sample_nfe=%d low_nfe_budgets=%s",
        model.train_levels, model.sample_nfe, model.low_nfe_budgets,
    )
    logging.info("Held-out test loader exists but is never used in train().")
    schedule_state = model.schedule_state()
    stiffness = model.stiffness_report(sorted(set([model.sample_nfe] + list(model.low_nfe_budgets))))
    logging.info("forward schedule: %s", schedule_state)
    logging.info("schedule-only stiffness L_S0=%.4e (EM needs max zeta0*dt < 2): %s",
                 stiffness["L_S0"], stiffness["per_nfe"])

    phase1_epochs = int(tcfg["phase1_epoch"])
    phase2_epochs = int(tcfg["phase2_epoch"])
    low_nfe_epochs = int(tcfg.get("low_nfe_epoch", "40"))
    valid_interval = int(tcfg.get("valid_epoch_interval", "5"))
    screen_interval = int(tcfg.get("sample_valid_interval", "10"))
    screen_start = int(tcfg.get("sample_valid_start_epoch", "30"))
    screen_batches = int(tcfg.get("sample_valid_batches", "10"))
    screen_nsample = int(tcfg.get("sample_valid_nsample", "5"))
    snapshot_epochs = set(comma_ints(tcfg.get("snapshot_epochs", "")))
    start_time = time.time()

    # ---------------- Phase 1 ----------------
    model.set_low_nfe_curriculum(False)
    opt1 = Adam(
        model.diffmodel_uncond.parameters(),
        lr=float(tcfg["phase1_lr"]),
        weight_decay=float(tcfg.get("phase1_weight_decay", "1e-6")),
    )
    sch1 = build_scheduler(opt1, tcfg, phase1_epochs)
    best_phase1 = float("inf")
    phase1_bad = 0
    phase1_patience = int(tcfg.get("phase1_patience", "10"))
    best_phase1_path = run_dir / "params" / f"{run_name}_best_clean_phase1_uncond.pt"

    logging.info("--- Phase 1: continuous-time unconditional clean prior, max_epochs=%d ---", phase1_epochs)
    for epoch in range(1, phase1_epochs + 1):
        epoch_start = time.time()
        model.train(); model.diffmodel_uncond.train(); model.diffmodel_cond.eval()
        losses = []
        for batch in loaders["phase1_train"]:
            opt1.zero_grad(set_to_none=True)
            loss = model(batch, is_train=1, is_phase1=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffmodel_uncond.parameters(), float(tcfg.get("phase1_grad_clip", "5.0")))
            opt1.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss = None
        stop = False
        if epoch % valid_interval == 0:
            val_loss = full_validation_loss(model, loaders["phase1_valid"], True)
            if val_loss < best_phase1 - 1e-8:
                best_phase1 = val_loss
                phase1_bad = 0
                torch.save({
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": args.seed,
                    "diffmodel_uncond": model.diffmodel_uncond.state_dict(),
                    "embed_layer": model.embed_layer.state_dict(),
                    "spectral_power_path": cfg["gad"]["spectral_power_path"],
                    "protocol": "continuous_logsnr_clean_full_train_unconditional_prior",
                    "schedule": schedule_state,
                }, best_phase1_path)
                logging.info("Saved best phase-1 prior epoch=%d val_loss=%.6f", epoch, val_loss)
            else:
                phase1_bad += 1
                stop = phase1_bad >= phase1_patience
        sch1.step()
        lr = opt1.param_groups[0]["lr"]
        append_history(history_path, [
            "uncond", epoch, train_loss, "" if val_loss is None else val_loss,
            "", "", "", "", "", lr, time.time() - epoch_start, time.time() - start_time,
        ])
        logging.info("phase=1 epoch=%03d train=%.6f val=%s lr=%.3e", epoch, train_loss,
                     "" if val_loss is None else f"{val_loss:.6f}", lr)
        if stop:
            logging.info("Phase-1 early stopping after %d non-improving validations.", phase1_bad)
            break

    if not best_phase1_path.exists():
        raise RuntimeError("No phase-1 checkpoint saved; check validation interval/epoch settings.")
    best_prior = torch.load(best_phase1_path, map_location=args.device)
    model.diffmodel_uncond.load_state_dict(best_prior["diffmodel_uncond"])
    model.diffmodel_cond.load_state_dict(best_prior["diffmodel_uncond"])
    model.embed_layer.load_state_dict(best_prior["embed_layer"])
    for p0 in model.diffmodel_uncond.parameters():
        p0.requires_grad_(False)
    for p0 in model.embed_layer.parameters():
        p0.requires_grad_(False)

    # ---------------- Phase 2 ----------------
    model.set_low_nfe_curriculum(False)
    opt2 = Adam(
        model.diffmodel_cond.parameters(),
        lr=float(tcfg["phase2_lr"]),
        weight_decay=float(tcfg.get("phase2_weight_decay", "1e-5")),
    )
    sch2 = build_scheduler(opt2, tcfg, phase2_epochs)
    snapshots: list[str] = []
    best_phase2_val = float("inf")
    best_phase2_screen = float("inf")
    phase2_bad = 0
    phase2_patience = int(tcfg.get("phase2_patience", "10"))
    best_phase2_val_path = run_dir / "params" / f"{run_name}_best_phase2_valloss_combined.pt"
    best_phase2_screen_path = run_dir / "params" / f"{run_name}_best_phase2_screen_combined.pt"

    logging.info("--- Phase 2: task-aligned exact SC-TC conditional training, max_epochs=%d ---", phase2_epochs)
    for epoch in range(1, phase2_epochs + 1):
        epoch_start = time.time()
        model.train(); model.diffmodel_uncond.eval(); model.diffmodel_cond.train()
        losses = []
        for batch in loaders["phase2_train"]:
            opt2.zero_grad(set_to_none=True)
            loss = model(batch, is_train=1, is_phase1=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.diffmodel_cond.parameters(), float(tcfg.get("phase2_grad_clip", "1.0")))
            opt2.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses))
        val_loss = None
        stop = False
        if epoch % valid_interval == 0:
            val_loss = full_validation_loss(model, loaders["phase2_valid"], False)
            if val_loss < best_phase2_val - 1e-8:
                best_phase2_val = val_loss
                phase2_bad = 0
                save_combined(model, best_phase2_val_path, {
                    "protocol": "v3_phase2_best_validation_loss",
                    "dataset": dataset, "seed": args.seed, "phase2_epoch": epoch,
                    "phase2_val_loss": val_loss, "config_path": str(config_path),
                })
            else:
                phase2_bad += 1
                stop = phase2_bad >= phase2_patience

        screening = None
        if epoch >= screen_start and epoch % screen_interval == 0:
            screen_aggregation = tcfg.get("sample_valid_aggregation", "median")
            screening = evaluate_sampling_metrics(
                model, loaders["valid_eval"], metadata.train_std, metadata.train_mean,
                nsample=screen_nsample, max_batches=screen_batches,
                aggregation=screen_aggregation, fence_compatible_mape=True,
                desc=f"phase2-screen-nfe{model.sample_nfe}-{screen_aggregation}-seed{args.seed}-epoch{epoch}",
            )
            screen_row = {"MAE": screening["MAE"], "RMSE": screening["RMSE"], "MAPE": screening["MAPE"]}
            max_ratio = float(baseline_ratio_fields(dataset, screen_row)["Baseline_Max_Ratio"])
            if max_ratio < best_phase2_screen:
                best_phase2_screen = max_ratio
                save_combined(model, best_phase2_screen_path, {
                    "protocol": "v3_phase2_best_partial_validation_sampling",
                    "dataset": dataset, "seed": args.seed, "phase2_epoch": epoch,
                    "sample_nfe": model.sample_nfe, "screening_metrics": screening,
                    "baseline_max_ratio": max_ratio, "config_path": str(config_path),
                })

        if epoch in snapshot_epochs:
            checkpoint = run_dir / "params" / f"{run_name}_phase2_epoch{epoch:03d}_combined.pt"
            save_combined(model, checkpoint, {
                "protocol": "v3_continuous_logsnr_task_aligned_phase2",
                "dataset": dataset, "seed": args.seed, "phase2_epoch": epoch,
                "phase1_selected_epoch": int(best_prior["epoch"]),
                "phase1_selected_val_loss": float(best_prior["val_loss"]),
                "phase2_train_loss": train_loss, "phase2_val_loss": val_loss,
                "screening_mean_metrics": screening, "screening_nsample": screen_nsample,
                "screening_batches": screen_batches, "sample_nfe": model.sample_nfe,
                "config_path": str(config_path), "spectral_power_path": cfg["gad"]["spectral_power_path"],
                "spectral_power_source": cfg["gad"]["spectral_power_source"],
            })
            snapshots.append(str(checkpoint))
            logging.info("Saved phase-2 candidate snapshot: %s", checkpoint)

        sch2.step()
        lr = opt2.param_groups[0]["lr"]
        append_history(history_path, [
            "cond", epoch, train_loss, "" if val_loss is None else val_loss,
            "" if screening is None else screening["RMSE"],
            "" if screening is None else screening["MAE"],
            "" if screening is None else screening["MAPE"],
            "" if screening is None else screening["batches"],
            "" if screening is None else screening["nsample"],
            lr, time.time() - epoch_start, time.time() - start_time,
        ])
        logging.info("phase=2 epoch=%03d train=%.6f val=%s screen=%s lr=%.3e", epoch, train_loss,
                     "" if val_loss is None else f"{val_loss:.6f}",
                     "" if screening is None else {k: screening[k] for k in ("RMSE", "MAE", "MAPE")}, lr)
        if stop:
            logging.info("Phase-2 early stopping after %d non-improving validations.", phase2_bad)
            break

    # Start sparse-trajectory fine-tuning from the best Phase-2 validation state.
    phase2_start = best_phase2_screen_path if best_phase2_screen_path.exists() else best_phase2_val_path
    if not phase2_start.exists():
        # Ensure a usable state even when short debugging runs never hit validation.
        phase2_start = run_dir / "params" / f"{run_name}_phase2_last_combined.pt"
        save_combined(model, phase2_start, {"protocol": "v3_phase2_last_fallback", "dataset": dataset, "seed": args.seed})
    load_checkpoint_into_model(model, phase2_start, args.device)

    # ---------------- Phase 3: low-NFE curriculum ----------------
    phase3_snapshots: list[str] = []
    best_phase3_path = None
    if low_nfe_epochs > 0:
        model.set_low_nfe_curriculum(True)
        opt3 = Adam(
            model.diffmodel_cond.parameters(),
            lr=float(tcfg.get("low_nfe_lr", "2e-4")),
            weight_decay=float(tcfg.get("phase2_weight_decay", "1e-5")),
        )
        low_valid_interval = int(tcfg.get("low_nfe_valid_interval", "5"))
        low_snapshot_interval = int(tcfg.get("low_nfe_snapshot_interval", "5"))
        low_patience = int(tcfg.get("low_nfe_patience", "7"))
        best_low_ratio = float("inf")
        low_bad = 0
        best_phase3_path = run_dir / "params" / f"{run_name}_best_low_nfe_combined.pt"
        logging.info(
            "--- Phase 3: sparse %s trajectory fine-tuning, epochs=%d budgets=%s probs=%s ---",
            schedule_tag, low_nfe_epochs, model.low_nfe_budgets, model.low_nfe_probs,
        )
        for epoch in range(1, low_nfe_epochs + 1):
            epoch_start = time.time()
            model.train(); model.diffmodel_uncond.eval(); model.diffmodel_cond.train()
            losses = []
            for batch in loaders["phase2_train"]:
                opt3.zero_grad(set_to_none=True)
                loss = model(batch, is_train=1, is_phase1=False)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.diffmodel_cond.parameters(), float(tcfg.get("phase2_grad_clip", "1.0")))
                opt3.step()
                losses.append(float(loss.item()))
            train_loss = float(np.mean(losses))
            val_loss = None
            screening = None
            stop = False
            if epoch % low_valid_interval == 0:
                val_loss = full_validation_loss(model, loaders["phase2_valid"], False)
                screening = evaluate_sampling_metrics(
                    model, loaders["valid_eval"], metadata.train_std, metadata.train_mean,
                    nsample=screen_nsample, max_batches=screen_batches,
                    aggregation=tcfg.get("sample_valid_aggregation", "median"), fence_compatible_mape=True,
                    desc=f"phase3-screen-nfe{model.sample_nfe}-seed{args.seed}-epoch{epoch}",
                )
                rfields = baseline_ratio_fields(dataset, screening)
                ratio = float(rfields["Baseline_Max_Ratio"])
                if ratio < best_low_ratio - 1e-6:
                    best_low_ratio = ratio
                    low_bad = 0
                    save_combined(model, best_phase3_path, {
                        "protocol": "v3_best_low_nfe_partial_validation",
                        "dataset": dataset, "seed": args.seed, "phase3_epoch": epoch,
                        "sample_nfe": model.sample_nfe, "screening_metrics": screening,
                        "baseline_ratios": rfields, "config_path": str(config_path),
                    })
                else:
                    low_bad += 1
                    stop = low_bad >= low_patience

            if epoch % low_snapshot_interval == 0 or epoch == low_nfe_epochs:
                checkpoint = run_dir / "params" / f"{run_name}_phase3_low_nfe_epoch{epoch:03d}_combined.pt"
                save_combined(model, checkpoint, {
                    "protocol": "v3_sparse_trajectory_finetune",
                    "dataset": dataset, "seed": args.seed, "phase3_epoch": epoch,
                    "sample_nfe": model.sample_nfe, "low_nfe_budgets": model.low_nfe_budgets,
                    "low_nfe_probs": model.low_nfe_probs, "phase3_train_loss": train_loss,
                    "phase3_val_loss": val_loss, "screening_mean_metrics": screening,
                    "config_path": str(config_path),
                })
                phase3_snapshots.append(str(checkpoint))
            append_history(history_path, [
                "low_nfe", epoch, train_loss, "" if val_loss is None else val_loss,
                "" if screening is None else screening["RMSE"],
                "" if screening is None else screening["MAE"],
                "" if screening is None else screening["MAPE"],
                "" if screening is None else screening["batches"],
                "" if screening is None else screening["nsample"],
                opt3.param_groups[0]["lr"], time.time() - epoch_start, time.time() - start_time,
            ])
            logging.info("phase=3 epoch=%03d train=%.6f val=%s screen=%s", epoch, train_loss,
                         "" if val_loss is None else f"{val_loss:.6f}",
                         "" if screening is None else {k: screening[k] for k in ("RMSE", "MAE", "MAPE")})
            if stop:
                logging.info("Phase-3 early stopping after %d non-improving screens.", low_bad)
                break
        model.set_low_nfe_curriculum(False)

    all_snapshots = snapshots + phase3_snapshots
    if best_phase3_path is not None and best_phase3_path.exists():
        all_snapshots.append(str(best_phase3_path))
    if best_phase2_screen_path.exists():
        all_snapshots.append(str(best_phase2_screen_path))
    if best_phase2_val_path.exists():
        all_snapshots.append(str(best_phase2_val_path))
    # Stable de-duplication.
    all_snapshots = list(dict.fromkeys(all_snapshots))

    summary = {
        "protocol": (
            "FENCE + GAD-MASS v3: continuous log-SNR + conditional residual power + task-aligned loss + low-NFE curriculum"
            if schedule_tag == "MASS" else
            f"FENCE + GAD-{schedule_tag} v3: same pipeline as GAD-MASS v3 with a scalar {schedule_tag} forward channel"
        ),
        "dataset": dataset,
        "seed": args.seed,
        "schedule": schedule_state,
        "schedule_stiffness": stiffness,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "best_phase1_checkpoint": str(best_phase1_path),
        "best_phase1_epoch": int(best_prior["epoch"]),
        "best_phase1_val_loss": float(best_prior["val_loss"]),
        "phase2_warmstart_for_low_nfe": str(phase2_start),
        "best_phase3_checkpoint": None if best_phase3_path is None or not best_phase3_path.exists() else str(best_phase3_path),
        "saved_phase2_snapshots": snapshots,
        "saved_phase3_snapshots": phase3_snapshots,
        "validation_candidate_snapshots": all_snapshots,
        "train_levels": model.train_levels,
        "initial_sample_nfe": int(cfg["diffusion"].get("sample_nfe", "12")),
        "low_nfe_budgets": model.low_nfe_budgets,
        "note": "No held-out test evaluation was performed. Select checkpoint/NFE/guidance on validation only, freeze them, then run locked-test once per seed.",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info("Training complete. Summary=%s", summary_path)


def read_config(path: str | Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(path)
    return cfg


def apply_eval_overrides(cfg: configparser.ConfigParser, args: argparse.Namespace) -> configparser.ConfigParser:
    mapping = {
        "pi": ("fbg", "pi"),
        "max_guidance": ("fbg", "max_guidance"),
        "n_clusters": ("fbg", "n_clusters"),
        "t0": ("fbg", "t0"),
        "t1": ("fbg", "t1"),
        "temp_scale": ("fbg", "temp_scale"),
        "logp_delta_clip": ("fbg", "logp_delta_clip"),
        "sample_nfe": ("diffusion", "sample_nfe"),
        "nsample": ("diffusion", "nsample"),
        "guidance": ("model", "guidance"),
        "cfg_scale": ("model", "cfg_scale"),
    }
    for attr, (section, key) in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            cfg[section][key] = str(value)
    aggregation = getattr(args, "aggregation", None)
    if aggregation is not None:
        cfg["train"]["full_validation_aggregation"] = str(aggregation)
    pca = getattr(args, "posterior_cluster_average", None)
    if pca is not None:
        cfg["gad"]["posterior_cluster_average"] = "1" if bool(pca) else "0"
    final_proj = getattr(args, "final_x0_projection", None)
    if final_proj is not None:
        cfg["gad"]["final_x0_projection"] = "1" if bool(final_proj) else "0"
    return cfg


def evaluate_cmd(args: argparse.Namespace) -> None:
    setup_logging(args.logfile)
    cfg = read_config(args.config)
    apply_eval_overrides(cfg, args)
    validate_config_files(cfg)
    trial_name = args.trial_name or Path(args.checkpoint).stem
    agg = cfg["train"].get("full_validation_aggregation", "median")
    results_file = args.results_file or Path(args.checkpoint).with_suffix(f".{args.split}.{agg}{args.nsample}.csv")
    evaluate_checkpoint(
        cfg,
        args.checkpoint,
        split=args.split,
        device=args.device,
        eval_seed=args.eval_seed,
        trial_name=trial_name,
        results_file=results_file,
        nsample=args.nsample,
        max_batches=args.max_batches,
    )


def rank_rows(rows: list[dict[str, Any]], metric: str, dataset: str | None = None) -> list[dict[str, Any]]:
    metric = metric.upper()
    out = [dict(r) for r in rows]
    if not out:
        return out
    if dataset is None:
        dataset = str(out[0].get("Dataset", "")).upper() or None
    if dataset in BASELINE_TARGETS:
        for row in out:
            row.update(baseline_ratio_fields(dataset, row))
    if metric == "BASELINE_CONSTRAINED":
        if dataset not in BASELINE_TARGETS:
            raise ValueError("BASELINE_CONSTRAINED ranking requires PEMS04/PEMS08 dataset")
        def key(row):
            clears = bool(int(row.get("Clears_3pct_Margin_All", 0)))
            beats = bool(int(row["Beats_FENCE_All"]))
            nfe = int(row.get("Sample_NFE", 10**9))
            # First target a >=3% numerical margin on all three validation
            # metrics; among those, minimize NFE. This is a numerical target,
            # not a formal statistical-significance claim.
            if clears:
                return (0, nfe, float(row["Baseline_Max_Ratio"]), float(row["Baseline_Mean_Ratio"]))
            if beats:
                return (1, float(row["Baseline_Max_Ratio"]), nfe, float(row["Baseline_Mean_Ratio"]))
            return (2, float(row["Baseline_Max_Ratio"]), float(row["Baseline_Mean_Ratio"]), nfe)
        return sorted(out, key=key)
    if metric == "RANK_MEAN":
        metrics = ("MAE", "RMSE", "MAPE")
        n = len(out)
        rank_maps: dict[str, dict[int, float]] = {}
        for m in metrics:
            order = sorted(range(n), key=lambda i: _finite_or_inf(out[i][m]))
            rank_maps[m] = {idx: float(rank + 1) for rank, idx in enumerate(order)}
        for i, row in enumerate(out):
            row["Rank_MAE"] = rank_maps["MAE"][i]
            row["Rank_RMSE"] = rank_maps["RMSE"][i]
            row["Rank_MAPE"] = rank_maps["MAPE"][i]
            row["RANK_MEAN"] = (row["Rank_MAE"] + row["Rank_RMSE"] + row["Rank_MAPE"]) / 3.0
        return sorted(out, key=lambda r: (float(r["RANK_MEAN"]), _finite_or_inf(r["RMSE"])))
    return sorted(out, key=lambda r: _finite_or_inf(r[metric]))


def _finite_or_inf(value: Any) -> float:
    """Rank a diverged (NaN) evaluation last instead of scrambling the sort."""
    v = float(value)
    return v if np.isfinite(v) else float("inf")


def clone_config(cfg: configparser.ConfigParser) -> configparser.ConfigParser:
    out = configparser.ConfigParser()
    out.read_dict({section: dict(cfg[section]) for section in cfg.sections()})
    return out


def apply_candidate(cfg: configparser.ConfigParser, candidate: dict[str, Any]) -> None:
    if "guidance" in candidate:
        cfg["model"]["guidance"] = str(candidate["guidance"])
    if "cfg_scale" in candidate:
        cfg["model"]["cfg_scale"] = str(candidate["cfg_scale"])
    if "fbg_mode" in candidate:
        cfg["fbg"]["mode"] = str(candidate["fbg_mode"])
    for key in ("pi", "max_guidance", "n_clusters", "t0", "t1", "temp_scale", "logp_delta_clip"):
        if key in candidate:
            cfg["fbg"][key] = str(candidate[key])
    if "sample_nfe" in candidate:
        cfg["diffusion"]["sample_nfe"] = str(int(candidate["sample_nfe"]))
    if "aggregation" in candidate:
        cfg["train"]["full_validation_aggregation"] = str(candidate["aggregation"])
    if "posterior_cluster_average" in candidate:
        cfg["gad"]["posterior_cluster_average"] = "1" if candidate["posterior_cluster_average"] else "0"
    if "final_x0_projection" in candidate:
        cfg["gad"]["final_x0_projection"] = "1" if candidate["final_x0_projection"] else "0"


def strong_sampling_candidates(dataset: str, aggregation: str = "median") -> list[dict[str, Any]]:
    """Compact validation-only guidance grid for the low-NFE v3 model.

    cov_fbg/cluster is the primary path because v3 already models modal
    marginals with MASS and uses FBG for the concentrated cross-modal residual
    geometry. The grid is intentionally compact so it can be crossed with the
    NFE sweep without turning the held-out validation set into a large search.
    """
    dataset = dataset.upper()
    k0 = int(DATASET_DEFAULTS[dataset]["n_clusters"])
    k_alt = max(4, int(round(k0 * 0.75)))
    official_pi = 0.50 if dataset == "PEMS04" else 0.95
    profiles = [
        {"pi": 0.50, "max_guidance": 1.10, "n_clusters": k0, "temp_scale": 1.0},
        {"pi": 0.60, "max_guidance": 1.20, "n_clusters": k0, "temp_scale": 1.0},
        {"pi": 0.70, "max_guidance": 1.35, "n_clusters": k0, "temp_scale": 1.0},
        {"pi": official_pi, "max_guidance": 1.50, "n_clusters": k0, "temp_scale": 1.0},
        {"pi": 0.60, "max_guidance": 1.20, "n_clusters": k_alt, "temp_scale": 1.0},
        {"pi": 0.60, "max_guidance": 1.20, "n_clusters": k0, "temp_scale": 0.5},
        {"pi": 0.60, "max_guidance": 1.20, "n_clusters": k0, "temp_scale": 2.0},
    ]
    out = [
        {
            "guidance": "cov_fbg", "fbg_mode": "cluster", **prof,
            "t0": 0.80, "t1": 0.50, "logp_delta_clip": 0.0,
            "posterior_cluster_average": False, "final_x0_projection": True,
            "aggregation": aggregation,
        }
        for prof in profiles
    ]
    out.extend([
        {"guidance": "cfg", "cfg_scale": 1.10, "aggregation": aggregation,
         "posterior_cluster_average": False, "final_x0_projection": True},
        {"guidance": "cfg", "cfg_scale": 1.25, "aggregation": aggregation,
         "posterior_cluster_average": False, "final_x0_projection": True},
        {"guidance": "none", "aggregation": aggregation,
         "posterior_cluster_average": False, "final_x0_projection": True},
    ])
    unique, seen = [], set()
    for cand in out:
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key); unique.append(cand)
    return unique


def validation_search(args: argparse.Namespace) -> None:
    setup_logging(args.logfile)
    if args.screen_batches is not None and args.screen_batches <= 0:
        raise ValueError("--screen-batches must be positive or omitted")
    cfg_base = read_config(args.config)
    validate_config_files(cfg_base)
    dataset = args.dataset.upper()
    base_candidates = strong_sampling_candidates(dataset, aggregation=args.aggregation)
    nfe_grid = comma_ints(args.nfe_grid)
    if not nfe_grid or any(n < 2 for n in nfe_grid):
        raise ValueError("--nfe-grid must contain integers >=2")
    # Cross a compact guidance grid with all requested budgets. NFE is part of
    # the frozen validation-selected candidate, never chosen on the test set.
    candidates = []
    for nfe in sorted(set(nfe_grid)):
        for base in base_candidates:
            cand = dict(base); cand["sample_nfe"] = int(nfe); candidates.append(cand)
    out_dir = Path(args.out_dir).expanduser().resolve()
    screen_dir = out_dir / "screen"
    confirm_dir = out_dir / "confirm"
    screen_dir.mkdir(parents=True, exist_ok=True)
    confirm_dir.mkdir(parents=True, exist_ok=True)

    screen_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        cfg = clone_config(cfg_base)
        apply_candidate(cfg, cand)
        trial = f"cand{i:02d}_{cand['guidance']}"
        row = evaluate_checkpoint(
            cfg, args.checkpoint, split="validation", device=args.device, eval_seed=args.eval_seed,
            trial_name=trial, results_file=screen_dir / f"{trial}.csv",
            nsample=args.screen_nsample, max_batches=args.screen_batches,
        )
        row["Candidate_Index"] = i
        row["Candidate_JSON"] = json.dumps(cand, sort_keys=True)
        screen_rows.append(row)

    screen_ranked = rank_rows(screen_rows, args.selection_metric, dataset=dataset)
    screen_ranked_path = out_dir / "screen_ranked.csv"
    with screen_ranked_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(screen_ranked[0].keys()))
        writer.writeheader(); writer.writerows(screen_ranked)

    top = screen_ranked[: min(args.top_k, len(screen_ranked))]
    confirm_eval_seeds = comma_ints(args.confirm_eval_seeds)
    if not confirm_eval_seeds:
        raise ValueError("--confirm-eval-seeds must contain at least one integer seed")
    confirm_rows: list[dict[str, Any]] = []
    for rank, screened in enumerate(top, start=1):
        i = int(screened["Candidate_Index"])
        cand = candidates[i]
        seed_rows = []
        for eval_seed in confirm_eval_seeds:
            cfg = clone_config(cfg_base)
            apply_candidate(cfg, cand)
            trial = f"top{rank:02d}_cand{i:02d}_{cand['guidance']}_eseed{eval_seed}"
            row = evaluate_checkpoint(
                cfg, args.checkpoint, split="validation", device=args.device, eval_seed=eval_seed,
                trial_name=trial, results_file=confirm_dir / f"{trial}.csv",
                nsample=args.confirm_nsample, max_batches=None,
            )
            seed_rows.append(row)
        aggregate = dict(seed_rows[0])
        aggregate["Trial"] = f"cand{i:02d}_{cand['guidance']}_confirm_mean"
        aggregate["Eval_Seed"] = ",".join(map(str, confirm_eval_seeds))
        for m in ("MAE", "RMSE", "MAPE"):
            vals = np.asarray([float(r[m]) for r in seed_rows], dtype=np.float64)
            aggregate[m] = float(vals.mean())
            aggregate[f"{m}_EvalSeedStd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        aggregate["Candidate_Index"] = i
        aggregate["Candidate_JSON"] = json.dumps(cand, sort_keys=True)
        confirm_rows.append(aggregate)

    confirm_ranked = rank_rows(confirm_rows, args.selection_metric, dataset=dataset)
    confirm_path = out_dir / "confirm_ranked.csv"
    with confirm_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(confirm_ranked[0].keys()))
        writer.writeheader(); writer.writerows(confirm_ranked)

    best = confirm_ranked[0]
    best_idx = int(best["Candidate_Index"])
    selected = {
        "selection_protocol": "validation_only_two_stage_common_random_numbers",
        "selection_metric": args.selection_metric,
        "dataset": dataset,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "candidate_index": best_idx,
        "candidate": candidates[best_idx],
        "validation_metrics": {k: best[k] for k in ("MAE", "RMSE", "MAPE", "Baseline_Max_Ratio", "Baseline_Mean_Ratio", "Beats_FENCE_All", "Clears_3pct_Margin_All", "Sample_NFE")},
        "screen_eval_seed": args.eval_seed,
        "confirm_eval_seeds": confirm_eval_seeds,
        "screen_nsample": args.screen_nsample,
        "confirm_nsample": args.confirm_nsample,
        "baseline_targets": BASELINE_TARGETS[dataset],
        "note": "NFE, guidance, and aggregation were selected on validation only. Do not rerun this search on the test set; apply this frozen candidate in locked-test.",
    }
    selected_path = out_dir / "selected_sampling_config.json"
    selected_path.write_text(json.dumps(selected, indent=2))
    logging.info("Selected validation-only sampling config: %s", selected_path)


def locked_test(args: argparse.Namespace) -> None:
    setup_logging(args.logfile)
    selected = json.loads(Path(args.selected).read_text())
    cfg = read_config(selected["config"] if args.config is None else args.config)
    validate_config_files(cfg)
    selected_dataset = str(selected.get("dataset", "")).upper()
    if selected_dataset and selected_dataset not in cfg["file"]["data_prefix"].upper():
        raise ValueError(
            f"Frozen candidate is for {selected_dataset}, but config data_prefix is {cfg['file']['data_prefix']}."
        )
    apply_candidate(cfg, selected["candidate"])
    checkpoint = args.checkpoint or selected["checkpoint"]
    out = Path(args.results_file) if args.results_file else Path(args.selected).with_name("LOCKED_TEST.csv")
    row = evaluate_checkpoint(
        cfg, checkpoint, split="test", device=args.device, eval_seed=args.eval_seed,
        trial_name=Path(checkpoint).stem + "_LOCKED", results_file=out,
        nsample=args.nsample, max_batches=None,
    )
    sidecar = out.with_suffix(out.suffix + ".metadata.json")
    sidecar.write_text(json.dumps({
        "frozen_selection_file": str(Path(args.selected).expanduser().resolve()),
        "frozen_candidate": selected["candidate"],
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "config": str(Path(args.config or selected["config"]).expanduser().resolve()),
        "locked_test_row": row,
    }, indent=2))
    logging.info("Saved locked-test provenance sidecar: %s", sidecar)

def validate_snapshots(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    setup_logging(run_dir / "logs" / f"validate_snapshots_{int(time.time())}.log")
    summary_candidates = sorted((run_dir / "results").glob("*_training_summary.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"No training summary found under {run_dir / 'results'}")
    summary = json.loads(summary_candidates[-1].read_text())
    cfg = read_config(summary["config_path"])
    apply_eval_overrides(cfg, args)
    validate_config_files(cfg)
    out_dir = run_dir / "full_validation"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_list = summary.get("validation_candidate_snapshots", summary.get("saved_phase2_snapshots", []))
    for ckpt in candidate_list:
        ckpt_path = Path(ckpt)
        if args.epochs:
            if "epoch" not in ckpt_path.stem:
                continue
            epoch = int(ckpt_path.stem.split("epoch")[-1].split("_")[0])
            if epoch not in set(comma_ints(args.epochs)):
                continue
        trial = ckpt_path.stem
        row = evaluate_checkpoint(
            cfg,
            ckpt_path,
            split="validation",
            device=args.device,
            eval_seed=args.eval_seed,
            trial_name=trial,
            results_file=out_dir / "results" / f"{trial}_VALIDATION_{cfg['train'].get('full_validation_aggregation', 'median')}{args.nsample}.csv",
            nsample=args.nsample,
            max_batches=args.max_batches,
        )
        rows.append(row)
    if not rows:
        raise RuntimeError("No snapshots were evaluated.")
    metric = args.selection_metric.upper()
    ranked = rank_rows(rows, metric, dataset=summary["dataset"])
    ranked_path = out_dir / f"ranked_by_{metric.lower()}.csv"
    with ranked_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        writer.writeheader(); writer.writerows(ranked)
    logging.info("Saved ranked validation results: %s", ranked_path)
    best = ranked[0]
    best_path = out_dir / "selected_checkpoint.json"
    best_path.write_text(json.dumps(best, indent=2))
    logging.info("Selected by validation %s: %s", metric, best_path)


def sweep_fbg(args: argparse.Namespace) -> None:
    setup_logging(args.logfile)
    cfg_base = read_config(args.config)
    validate_config_files(cfg_base)
    pis = [float(x) for x in args.pis.split(",") if x.strip()]
    maxgs = [float(x) for x in args.max_guidances.split(",") if x.strip()]
    clusters = [int(x) for x in args.n_clusters_grid.split(",") if x.strip()]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pi in pis:
        for g in maxgs:
            for k in clusters:
                cfg = configparser.ConfigParser()
                cfg.read_dict({section: dict(cfg_base[section]) for section in cfg_base.sections()})
                cfg["fbg"]["pi"] = str(pi)
                cfg["fbg"]["max_guidance"] = str(g)
                cfg["fbg"]["n_clusters"] = str(k)
                cfg["diffusion"]["sample_nfe"] = str(args.sample_nfe)
                trial = f"{Path(args.checkpoint).stem}_pi{str(pi).replace('.', 'p')}_g{str(g).replace('.', 'p')}_k{k}_nfe{args.sample_nfe}"
                row = evaluate_checkpoint(
                    cfg,
                    args.checkpoint,
                    split="validation",
                    device=args.device,
                    eval_seed=args.eval_seed,
                    trial_name=trial,
                    results_file=out_dir / f"{trial}_validation.csv",
                    nsample=args.nsample,
                    max_batches=args.max_batches,
                )
                row["pi"] = pi; row["max_guidance"] = g; row["n_clusters"] = k
                rows.append(row)
    metric = args.selection_metric.upper()
    sweep_dataset = "PEMS04" if "PEMS04" in cfg_base["file"]["data_prefix"].upper() else "PEMS08"
    ranked = rank_rows(rows, metric, dataset=sweep_dataset)
    ranked_path = out_dir / f"fbg_sweep_ranked_by_{metric.lower()}.csv"
    with ranked_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
        writer.writeheader(); writer.writerows(ranked)
    logging.info("Saved FBG sweep ranking: %s", ranked_path)


def schedule_report(args: argparse.Namespace) -> None:
    """Print schedule-only stiffness for MASS and FCPS on the real graph/power.

    Cheap (no training, no denoiser): use it to choose the FCPS NFE sweep
    before spending GPU hours.  MASS uses the config's r_max/r_min; FCPS uses
    the config's fcps_* keys when present and the defaults otherwise.
    """
    setup_logging(args.logfile)
    cfg = read_config(args.config)
    validate_config_files(cfg)
    nfes = comma_ints(args.nfes)
    loaders = build_exp34_clean_loaders(cfg)
    graph = load_graph_for_fence(cfg, target_dim=loaders["target_dim"], device=args.device, dtype=torch.float32)
    s_np = load_or_estimate_spectral_power(
        cfg, graph, loaders["spectral_train"], device=args.device, residual_loader=loaders["phase2_train"]
    )
    s_k = torch.as_tensor(np.asarray(s_np, dtype=np.float32))
    lam = graph.lam_gamma.detach().cpu()
    sigma = float(cfg["gad"].get("sigma", "1.0"))
    variants = [("MASS", {"schedule": "mass"})]
    for grid in FCPS_GRIDS:
        variants.append((f"FCPS[{grid}]", {"schedule": "fcps", "fcps_grid": grid}))
    report: dict[str, Any] = {"config": str(Path(args.config).resolve()), "nfes": nfes, "schedules": {}}
    for name, override in variants:
        gad = dict(cfg["gad"])
        gad.update(override)
        sched, info = build_schedule(gad, lam_gamma=lam, s_k=s_k)
        rep = schedule_stiffness_report(sched, lam, s_k, sigma, nfes)
        rep["calibration"] = info
        report["schedules"][name] = rep
    header = f"{'schedule':<14}{'L_S0':>12}" + "".join(f"{'NFE'+str(n):>12}" for n in nfes)
    print(header)
    for name, rep in report["schedules"].items():
        cells = "".join(f"{rep['per_nfe'][n]['max_zeta0_dt']:>12.3g}" for n in nfes)
        print(f"{name:<14}{rep['L_S0']:>12.4g}{cells}")
    print("cells: max zeta0*dt over reverse steps and modes; explicit EM on the linear part needs < 2.")
    fcps_info = report["schedules"]["FCPS[time]"]["calibration"]
    print(f"FCPS horizon cbar_T={fcps_info['cbar_T']:.6g} ({fcps_info['horizon_mode']}); "
          f"terminal retained power MASS={fcps_info.get('mass_terminal_power_retained', float('nan')):.3e} "
          f"FCPS={fcps_info['fcps_terminal_power_retained']:.3e}; cond(L_gamma)={fcps_info['cond_L_gamma']:.4g}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(args.out)


def add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS))
    p.add_argument("--data-root", required=True, help="Parent directory containing PEMS04/ and PEMS08/, or the dataset directory itself.")
    p.add_argument("--out-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=126)
    p.add_argument("--run-name", default=None)
    p.add_argument("--phase1-epoch", type=int, default=None)
    p.add_argument("--phase2-epoch", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--snapshot-epochs", default=None)
    p.add_argument("--pi", type=float, default=None)
    p.add_argument("--max-guidance", type=float, default=None)
    p.add_argument("--n-clusters", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--r-min", type=float, default=None)
    p.add_argument("--train-levels", type=int, default=50, help="Reference training grid only; continuous training is independent of this count.")
    p.add_argument("--sample-nfe", type=int, default=12, help="Reverse denoiser evaluations used for validation/sampling.")
    p.add_argument("--nsample", type=int, default=10)
    p.add_argument("--graph-path", default=None)
    p.add_argument("--spectral-power-path", default=None)
    p.add_argument("--force-spectral-power", action="store_true")
    p.add_argument("--guidance", choices=["cov_fbg", "fbg", "orig_fbg", "cfg", "none"], default="cov_fbg")
    p.add_argument("--power-mode", choices=["second_moment", "centered", "blend", "conditional_residual", "clean_residual_blend"], default="clean_residual_blend")
    p.add_argument("--power-blend", type=float, default=0.5, help="Clean centered/second-moment blend, when relevant.")
    p.add_argument("--residual-alpha", type=float, default=0.75, help="Weight on train-only SC-TC conditional-residual power in clean_residual_blend.")
    p.add_argument("--low-nfe-epoch", type=int, default=40, help="Phase-3 sparse-trajectory fine-tuning epochs; set 0 to disable.")
    p.add_argument("--aggregation", choices=["median", "mean"], default="median")
    p.add_argument("--posterior-cluster-average", action="store_true", help="Ablation only; official FENCE sums cluster evidence.")
    p.add_argument("--no-final-x0-projection", action="store_true", help="Ablation only; returns the residual t_eps state.")
    p.add_argument("--schedule", choices=list(SCHEDULE_KINDS), default="mass",
                   help="Forward channel: mass (published Table VII) or fcps (scalar GAD baseline).")
    p.add_argument("--fcps-horizon", choices=list(FCPS_HORIZON_MODES), default="match_mass_power",
                   help="FCPS cbar_T: match MASS terminal retained power (default), Definition-1 delta, or explicit.")
    p.add_argument("--fcps-cbar-T", type=float, default=None, help="Explicit FCPS horizon; requires --fcps-horizon explicit.")
    p.add_argument("--fcps-delta", type=float, default=1e-3, help="Tolerance for --fcps-horizon delta.")
    p.add_argument("--fcps-alpha", type=float, default=4.0, help="FCPS polynomial exponent (preliminary GAD default 4).")
    p.add_argument("--fcps-c-min", type=float, default=0.1, help="FCPS rate floor (preliminary GAD default 0.1).")
    p.add_argument("--fcps-grid", choices=list(FCPS_GRIDS), default="time",
                   help="FCPS discretization: uniform in t (same as MASS) or uniform in cbar (preliminary GAD sampler).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-safe SC-TC runner for FENCE + GAD-MASS on PEMS04 and PEMS08."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train clean end-to-end GAD-MAS-FENCE for one SC-TC dataset.")
    add_common_train_args(p_train)
    p_train.set_defaults(func=train)

    p_eval = sub.add_parser("evaluate", help="Evaluate one combined checkpoint on validation or held-out test.")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--split", choices=["validation", "test"], default="validation")
    p_eval.add_argument("--device", default="cuda:0")
    p_eval.add_argument("--eval-seed", type=int, default=42)
    p_eval.add_argument("--nsample", type=int, default=10)
    p_eval.add_argument("--max-batches", type=int, default=None)
    p_eval.add_argument("--trial-name", default=None)
    p_eval.add_argument("--results-file", default=None)
    p_eval.add_argument("--logfile", default=None)
    p_eval.add_argument("--pi", type=float, default=None)
    p_eval.add_argument("--max-guidance", type=float, default=None)
    p_eval.add_argument("--n-clusters", type=int, default=None)
    p_eval.add_argument("--sample-nfe", type=int, default=None)
    p_eval.add_argument("--guidance", choices=["cov_fbg", "fbg", "orig_fbg", "cfg", "none"], default=None)
    p_eval.add_argument("--cfg-scale", type=float, default=None)
    p_eval.add_argument("--t0", type=float, default=None)
    p_eval.add_argument("--t1", type=float, default=None)
    p_eval.add_argument("--temp-scale", type=float, default=None)
    p_eval.add_argument("--logp-delta-clip", type=float, default=None)
    p_eval.add_argument("--aggregation", choices=["median", "mean"], default=None)
    p_eval.set_defaults(func=evaluate_cmd)

    p_val = sub.add_parser("validate-snapshots", help="Full-validation paper-compatible evaluation for all saved snapshots in a run directory.")
    p_val.add_argument("--run-dir", required=True)
    p_val.add_argument("--device", default="cuda:0")
    p_val.add_argument("--eval-seed", type=int, default=42)
    p_val.add_argument("--nsample", type=int, default=10)
    p_val.add_argument("--max-batches", type=int, default=None)
    p_val.add_argument("--epochs", default=None, help="Optional comma-list, e.g. 100,110,120.")
    p_val.add_argument("--selection-metric", default="BASELINE_CONSTRAINED", choices=["BASELINE_CONSTRAINED", "RANK_MEAN", "RMSE", "MAE", "MAPE"])
    p_val.add_argument("--pi", type=float, default=None)
    p_val.add_argument("--max-guidance", type=float, default=None)
    p_val.add_argument("--n-clusters", type=int, default=None)
    p_val.add_argument("--sample-nfe", type=int, default=None)
    p_val.add_argument("--guidance", choices=["cov_fbg", "fbg", "orig_fbg", "cfg", "none"], default=None)
    p_val.add_argument("--aggregation", choices=["median", "mean"], default=None)
    p_val.set_defaults(func=validate_snapshots)

    p_sweep = sub.add_parser("sweep-fbg", help="Legacy validation-only FBG grid for a fixed checkpoint.")
    p_sweep.add_argument("--config", required=True)
    p_sweep.add_argument("--checkpoint", required=True)
    p_sweep.add_argument("--out-dir", required=True)
    p_sweep.add_argument("--device", default="cuda:0")
    p_sweep.add_argument("--eval-seed", type=int, default=42)
    p_sweep.add_argument("--nsample", type=int, default=10)
    p_sweep.add_argument("--max-batches", type=int, default=None)
    p_sweep.add_argument("--sample-nfe", type=int, default=12)
    p_sweep.add_argument("--pis", default="0.45,0.50,0.55")
    p_sweep.add_argument("--max-guidances", default="1.05,1.10,1.20")
    p_sweep.add_argument("--n-clusters-grid", default="12,15,18")
    p_sweep.add_argument("--selection-metric", default="BASELINE_CONSTRAINED", choices=["BASELINE_CONSTRAINED", "RANK_MEAN", "RMSE", "MAE", "MAPE"])
    p_sweep.add_argument("--logfile", default=None)
    p_sweep.set_defaults(func=sweep_fbg)

    p_search = sub.add_parser("validation-search", help="Two-stage validation-only search over guidance geometry; never touches test.")
    p_search.add_argument("--dataset", required=True, choices=sorted(DATASET_DEFAULTS))
    p_search.add_argument("--config", required=True)
    p_search.add_argument("--checkpoint", required=True)
    p_search.add_argument("--out-dir", required=True)
    p_search.add_argument("--device", default="cuda:0")
    p_search.add_argument("--eval-seed", type=int, default=42)
    p_search.add_argument("--screen-nsample", type=int, default=3)
    p_search.add_argument("--screen-batches", type=int, default=10)
    p_search.add_argument("--confirm-nsample", type=int, default=10)
    p_search.add_argument("--confirm-eval-seeds", default="42,43,44")
    p_search.add_argument("--top-k", type=int, default=7)
    p_search.add_argument("--nfe-grid", default="6,8,10,12,16,25,50", help="Validation-only reverse-evaluation budgets.")
    p_search.add_argument("--aggregation", choices=["median", "mean"], default="median")
    p_search.add_argument("--selection-metric", default="BASELINE_CONSTRAINED", choices=["BASELINE_CONSTRAINED", "RANK_MEAN", "RMSE", "MAE", "MAPE"])
    p_search.add_argument("--logfile", default=None)
    p_search.set_defaults(func=validation_search)

    p_lock = sub.add_parser("locked-test", help="Apply one frozen validation-selected sampling config to the held-out test exactly once.")
    p_lock.add_argument("--selected", required=True, help="selected_sampling_config.json from validation-search")
    p_lock.add_argument("--config", default=None, help="Optional config override; normally use the path stored in selected JSON.")
    p_lock.add_argument("--checkpoint", default=None, help="Optional checkpoint override, e.g. another training seed with the same frozen hyperparameters.")
    p_lock.add_argument("--device", default="cuda:0")
    p_lock.add_argument("--eval-seed", type=int, default=42)
    p_lock.add_argument("--nsample", type=int, default=10)
    p_lock.add_argument("--results-file", default=None)
    p_lock.add_argument("--logfile", default=None)
    p_lock.set_defaults(func=locked_test)

    p_rep = sub.add_parser("schedule-report", help="Schedule-only stiffness of MASS vs FCPS on the real graph (no training).")
    p_rep.add_argument("--config", required=True, help="Any run config.ini; its graph, gamma and spectral power are used.")
    p_rep.add_argument("--nfes", default="16,25,50,100,200,500")
    p_rep.add_argument("--device", default="cpu")
    p_rep.add_argument("--out", default=None, help="Optional JSON output path.")
    p_rep.add_argument("--logfile", default=None)
    p_rep.set_defaults(func=schedule_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
