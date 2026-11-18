from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from datasets import make_synthetic_dataset_numpy, to_torch
from graphs import eigh_sym, rgg_knn_graph, shifted_laplacian
from metrics import summarize_generation_metrics
from diagnostics import generation_diagnostics, run_forward_diagnostics, spectral_power
from model import TimeConditionedGNN
from samplers import reverse_sample
from schedules import build_schedules, channel_params_for_times_torch, schedule_metadata


LOWER_IS_BETTER_METRICS = {
    "val_loss",
    "val_selection_score",
    "global_swd_mean",
    "mmd_rbf_z",
    "node_w1_median",
    "node_w1_q90",
    "cov_rel_err",
    "spec_cov_rel_err",
    "smoothness_w1",
    "band_ratio_l1",
    "graph_stationarity_gap_abs",
    "ipr_w1",
    "gini_w1",
}



def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def split_dataset(X: np.ndarray, num_train: int, num_val: int, num_test: int):
    total = num_train + num_val + num_test
    if X.shape[0] < total:
        raise ValueError("Dataset smaller than requested train/val/test split.")
    return X[:num_train], X[num_train:num_train + num_val], X[num_train + num_val:num_train + num_val + num_test]



def sample_forward_pairwise_torch(
    X0: torch.Tensor,
    V: torch.Tensor,
    a: torch.Tensor,
    nu: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    x0_tilde = X0 @ V
    z = torch.randn(x0_tilde.shape, generator=generator, device=X0.device, dtype=X0.dtype)
    xt_tilde = a * x0_tilde + torch.sqrt(torch.clamp(nu, min=0.0)) * z
    return xt_tilde @ V.T


def spectral_x0_loss(
    model: torch.nn.Module,
    xt: torch.Tensor,
    x0: torch.Tensor,
    V: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    x0_hat = model(xt, V, t)
    x0_hat_tilde = x0_hat @ V
    x0_tilde = x0 @ V
    return torch.mean((x0_hat_tilde - x0_tilde) ** 2)


def node_x0_loss(
    model: torch.nn.Module,
    xt: torch.Tensor,
    x0: torch.Tensor,
    V: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    x0_hat = model(xt, V, t)
    return torch.mean((x0_hat - x0) ** 2)


def spectral_dsm_loss(
    model: torch.nn.Module,
    xt: torch.Tensor,
    x0: torch.Tensor,
    V: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    xt_tilde = xt @ V
    x0_tilde = x0 @ V
    x0_hat = model(xt, V, t)
    x0_hat_tilde = x0_hat @ V
    denom = torch.clamp(nu, min=1e-4)
    s_theta = (a * x0_hat_tilde - xt_tilde) / denom
    s_star = (a * x0_tilde - xt_tilde) / denom
    return torch.mean((s_theta - s_star) ** 2)



def save_csv(path: str | Path, rows: list[dict]) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def write_json(path: str | Path, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))



def sanitize_metric_dict(metrics: dict[str, float], penalty: float = 1e9) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        fv = float(value)
        out[key] = fv if np.isfinite(fv) else float(penalty)
    return out



def compute_validation_selection_score(metrics: dict[str, float], args) -> float:
    return float(
        args.val_score_swd_weight * metrics["global_swd_mean"]
        + args.val_score_mmd_weight * metrics["mmd_rbf_z"]
        + args.val_score_cov_weight * metrics["cov_rel_err"]
        + args.val_score_spec_cov_weight * metrics["spec_cov_rel_err"]
        + args.val_score_band_weight * metrics["band_ratio_l1"]
        + args.val_score_nodew1_weight * metrics["node_w1_median"]
        + args.val_score_nodew1q90_weight * metrics["node_w1_q90"]
        + args.val_score_smoothness_weight * metrics["smoothness_w1"]
        + args.val_score_stationarity_weight * metrics["graph_stationarity_gap_abs"]
        + args.val_score_localization_weight * metrics["ipr_w1"]
    )



def get_metric_for_selection(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Validation metric '{metric_name}' not available. Available: {sorted(metrics.keys())}")
    return float(metrics[metric_name])



def is_better_metric(candidate: float, incumbent: float | None, metric_name: str) -> bool:
    if incumbent is None:
        return True
    if metric_name in LOWER_IS_BETTER_METRICS:
        return candidate < incumbent
    return candidate > incumbent



def export_training_plot_data(schedule_dir: str | Path, history: list[dict], best_model_metric: str) -> None:
    if not history:
        return
    rows: list[dict] = []
    for item in history:
        rows.append(
            {
                "epoch": int(item["epoch"]),
                "train_loss": float(item["train_loss"]),
                "validation_loss": (
                    float(item["val_loss"])
                    if item.get("did_validation", False) and item.get("val_loss") is not None
                    else ""
                ),
                "learning_rate": float(item.get("lr", np.nan)),
                "did_validation": bool(item.get("did_validation", False)),
                "is_best": bool(item.get("is_best", False)),
                "selection_metric": best_model_metric,
            }
        )
    csv_path = Path(schedule_dir) / "training_plot_data.csv"
    save_csv(csv_path, rows)

    val_mask = np.asarray([bool(r["did_validation"]) for r in rows], dtype=bool)
    arrays = {
        "epoch": np.asarray([r["epoch"] for r in rows], dtype=np.int64),
        "train_loss": np.asarray([r["train_loss"] for r in rows], dtype=np.float64),
        "learning_rate": np.asarray([r["learning_rate"] for r in rows], dtype=np.float64),
        "validation_epoch": np.asarray([r["epoch"] for r in rows if r["did_validation"]], dtype=np.int64),
        "validation_loss": np.asarray([r["validation_loss"] for r in rows if r["did_validation"]], dtype=np.float64),
        "is_best": np.asarray([bool(r["is_best"]) for r in rows], dtype=bool),
        "validation_mask": val_mask,
    }
    np.savez_compressed(Path(schedule_dir) / "training_plot_data.npz", **arrays)


UNSTABLE_ROW_EMPTY_METRICS = (
    "global_swd_mean",
    "global_swd_std",
    "mmd_rbf_z",
    "cov_rel_err",
    "spec_cov_rel_err",
    "full_spectrum_power_rel_l1",
    "band_ratio_l1",
    "smoothness_w1",
    "low_band_power_rel_l1",
    "high_band_power_rel_l1",
)


def figure2_spectral_power_rows(
    base_row: dict,
    X_gen: np.ndarray,
    X_ref: np.ndarray,
    V: np.ndarray,
    lam_gamma: np.ndarray,
    low_frac: float,
    high_frac: float,
) -> list[dict]:
    V = np.asarray(V, dtype=np.float64)
    lam = np.asarray(lam_gamma, dtype=np.float64)
    n = V.shape[0]
    low_k = max(1, int(round(float(low_frac) * n)))
    high_k = max(1, int(round(float(high_frac) * n)))
    Pg = spectral_power(np.asarray(X_gen, dtype=np.float64), V)
    Pr = spectral_power(np.asarray(X_ref, dtype=np.float64), V)
    eps = 1e-12
    rows: list[dict] = []
    for k in range(n):
        band = "low" if k < low_k else ("high" if k >= n - high_k else "middle")
        rows.append(
            {
                **base_row,
                "mode": int(k),
                "lam_gamma": float(lam[k]),
                "band": band,
                "reference_power": float(Pr[k]),
                "generated_power": float(Pg[k]),
                "log10_power_ratio": float(np.log10(max(Pg[k], eps) / max(Pr[k], eps))),
            }
        )
    return rows


def figure2_manifest(args, lam_gamma_np: np.ndarray, scheduler_names: list[str], methods: list[str], steps_list: list[int]) -> dict:
    return {
        "file": "figure2_spectral_power_long.csv",
        "what_figure_2_shows": (
            "Mode-wise spectral power E[x_tilde_k^2] of generated signals against the shifted Laplacian "
            "eigenvalue lambda_gamma_k, compared with the held-out test signals (reference_power). "
            "Log-scale y axis. Shade the modes with band=='low' (lowest quarter) and band=='high' "
            "(highest quarter); these enter the band-ratio metric of Table V."
        ),
        "columns": {
            "dataset": "synthetic data family",
            "seed": "graph/data/training seed of this run",
            "schedule": "forward schedule: Cosine, MAS-SNR, Isotropic-VP-Cosine, ...",
            "method": "reverse integrator (em for Table V / Figure 2)",
            "steps": "number of reverse steps (= denoiser evaluations)",
            "mode": "graph-frequency index k, ascending eigenvalue",
            "lam_gamma": "shifted Laplacian eigenvalue lambda_k + gamma",
            "band": "low / middle / high; low and high are the shaded quarters",
            "reference_power": "mean squared GFT coefficient of the held-out test signals",
            "generated_power": "mean squared GFT coefficient of the generated signals",
            "log10_power_ratio": "log10(generated_power / reference_power)",
        },
        "operating_points_in_paper": {"Cosine": {"method": "em", "steps": 2500}, "MAS-SNR": {"method": "em", "steps": 50}},
        "schedulers_run": scheduler_names,
        "methods_run": methods,
        "steps_run": steps_list,
        "gamma": float(args.gamma),
        "sigma": float(args.sigma),
        "n": int(lam_gamma_np.size),
        "band_low_frac": float(args.band_low_frac),
        "band_high_frac": float(args.band_high_frac),
        "note": (
            "Unstable (schedule, method, steps) combinations have no rows here; see results_summary.csv "
            "(stable column). Eigenvalues differ between seeds because each seed draws its own graph, so "
            "average across seeds by mode index, or plot one seed."
        ),
    }


VALIDATION_METRIC_KEYS = (
    "global_swd_mean", "global_swd_std", "mmd_rbf_z", "node_w1_median", "node_w1_q90", "cov_rel_err",
    "spec_cov_rel_err", "band_ratio_l1", "smoothness_w1", "graph_stationarity_gap_abs", "ipr_w1", "gini_w1",
)


def penalty_validation_metrics(args, num_samples: int, penalty: float = 1e9) -> dict[str, float]:
    """Validation metrics for a checkpoint whose sampler produced non-finite output."""
    metrics = {key: float(penalty) for key in VALIDATION_METRIC_KEYS}
    metrics["val_num_samples"] = int(num_samples)
    metrics["val_steps"] = int(args.val_steps)
    metrics["val_selection_score"] = float(penalty)
    metrics["val_loss"] = float(penalty)
    metrics["validation_unstable"] = 1.0
    return metrics


def run_full_reverse_validation(
    model: torch.nn.Module,
    schedule_name: str,
    schedule_obj,
    X_ref: torch.Tensor,
    X_ref_np: np.ndarray,
    V: torch.Tensor,
    V_np: np.ndarray,
    lam_gamma: torch.Tensor,
    L_np: np.ndarray,
    s_k: torch.Tensor,
    args,
    epoch: int,
) -> dict[str, float]:
    num_samples = min(args.val_num_samples, X_ref.shape[0])
    X_ref_eval_np = X_ref_np[:num_samples]
    current_mode = model.training
    model.eval()
    with torch.no_grad():
        X_gen = reverse_sample(
            model=model,
            V=V,
            lam_gamma=lam_gamma,
            sigma=args.sigma,
            schedule_name=schedule_name,
            schedule_obj=schedule_obj,
            s_k=s_k,
            num_samples=num_samples,
            steps=args.val_steps,
            t0=args.t0,
            T=args.T,
            seed=args.seed + 50000 + epoch,
            method=args.val_method,
            ddim_eta=args.ddim_eta,
            init_mode=args.sampler_init_mode,
        )
    X_gen_np = X_gen.detach().cpu().numpy()
    if not np.isfinite(X_gen_np).all():
        nan_count = int(np.isnan(X_gen_np).sum())
        inf_count = int(np.isinf(X_gen_np).sum())
        max_abs = float(np.nanmax(np.abs(np.nan_to_num(X_gen_np, nan=0.0, posinf=0.0, neginf=0.0))))
        raise FloatingPointError(
            f"Non-finite generated samples at epoch {epoch}: "
            f"nan_count={nan_count}, inf_count={inf_count}, max_abs={max_abs:.6e}"
        )
    metrics = summarize_generation_metrics(
        X_gen_np,
        X_ref_eval_np,
        V_np,
        L_np,
        sw_num_projections=args.metric_num_projections,
        sw_seeds=range(args.metric_num_seeds),
    )
    metrics = sanitize_metric_dict(metrics)
    metrics["val_num_samples"] = int(num_samples)
    metrics["val_steps"] = int(args.val_steps)
    metrics["val_selection_score"] = compute_validation_selection_score(metrics, args)
    metrics["val_loss"] = get_metric_for_selection(metrics, args.best_model_metric)
    if current_mode:
        model.train()
    return metrics



def build_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_decay_factor,
            patience=args.lr_patience_validations,
            threshold=args.lr_threshold,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unknown lr scheduler: {args.lr_scheduler}")



def train_one_scheduler(
    schedule_name: str,
    schedule_obj,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_val_np: np.ndarray,
    X_test_np: np.ndarray,
    V: torch.Tensor,
    V_np: np.ndarray,
    lam_gamma: torch.Tensor,
    W: torch.Tensor,
    L_torch: torch.Tensor,
    L_np: np.ndarray,
    s_k: torch.Tensor,
    args,
    schedule_dir: str | Path,
) -> tuple[torch.nn.Module, list[dict], list[dict], dict | None]:
    set_all_seeds(args.seed)
    perm_gen = torch.Generator(device=X_train.device)
    perm_gen.manual_seed(args.seed + 456)
    model = TimeConditionedGNN(
        W=W,
        L=L_torch,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_embed_dim=args.time_embed_dim,
        cheb_order=args.cheb_order,
        dropout=args.dropout,
        dtype=X_train.dtype,
    ).to(device=X_train.device, dtype=X_train.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_scheduler(optimizer, args)

    gen = torch.Generator(device=X_train.device)
    gen.manual_seed(args.seed + 123)
    history: list[dict] = []
    validation_history: list[dict] = []
    best_metric_value: float | None = None
    best_summary: dict | None = None

    num_batches = max(1, int(np.ceil(X_train.shape[0] / args.batch_size)))
    ensure_dir(schedule_dir)
    checkpoints_dir = Path(schedule_dir) / "checkpoints"
    ensure_dir(checkpoints_dir)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0], generator=perm_gen, device=X_train.device)
        train_loss_sum = 0.0
        count = 0

        for bi in range(num_batches):
            idx = perm[bi * args.batch_size : (bi + 1) * args.batch_size]
            xb = X_train[idx]
            if xb.shape[0] == 0:
                continue
            # Use the same time for corruption and model conditioning.
            # The previous version re-sampled t after constructing x_t, which made
            # the denoiser see an inconsistent time label.
            u = torch.rand((xb.shape[0],), generator=gen, device=X_train.device, dtype=X_train.dtype)
            t = args.train_t_min + (args.T - args.train_t_min) * u
            pars = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, args.sigma, t)
            xt = sample_forward_pairwise_torch(xb, V, pars["a"], pars["nu"], generator=gen)
            if args.train_loss == "x0":
                loss = spectral_x0_loss(
                    model,
                    xt=xt,
                    x0=xb,
                    V=V,
                    t=t,
                )
            elif args.train_loss == "dsm":
                loss = spectral_dsm_loss(
                    model,
                    xt=xt,
                    x0=xb,
                    V=V,
                    t=t,
                    a=pars["a"],
                    nu=pars["nu"],
                )
            else:
                raise ValueError(f"Unknown train_loss: {args.train_loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.item()) * xb.shape[0]
            count += xb.shape[0]

        train_loss = train_loss_sum / max(count, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        did_validation = (epoch % args.val_every_epochs == 0) or (epoch == args.num_epochs)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": current_lr,
            "did_validation": did_validation,
            "val_loss": None,
            "best_model_metric": args.best_model_metric,
        }
        print(f"[{schedule_name}] epoch {epoch:03d} | train_loss={train_loss:.6f} | lr={current_lr:.6e}")

        if did_validation:
            try:
                val_metrics = run_full_reverse_validation(
                    model=model,
                    schedule_name=schedule_name,
                    schedule_obj=schedule_obj,
                    X_ref=X_val,
                    X_ref_np=X_val_np,
                    V=V,
                    V_np=V_np,
                    lam_gamma=lam_gamma,
                    L_np=L_np,
                    s_k=s_k,
                    args=args,
                    epoch=epoch,
                )
            except FloatingPointError as err:
                if args.fail_on_unstable_sampling:
                    raise
                print(f"[{schedule_name}] validation @ epoch {epoch:03d} | UNSTABLE, assigning penalty metrics: {err}")
                val_metrics = penalty_validation_metrics(args, num_samples=min(args.val_num_samples, X_val.shape[0]))
            val_loss = float(val_metrics["val_loss"])
            row["val_loss"] = val_loss
            row.update({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()})
            validation_history.append({"epoch": epoch, **row})
            checkpoint_path = checkpoints_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "schedule_name": schedule_name,
                    "schedule_params": vars(schedule_obj),
                    "train_loss": train_loss,
                    "validation_metrics": val_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            if lr_scheduler is not None and args.lr_scheduler == "plateau":
                lr_scheduler.step(val_loss)
            is_best = is_better_metric(val_loss, best_metric_value, args.best_model_metric)
            row["is_best"] = is_best
            print(
                f"[{schedule_name}] validation @ epoch {epoch:03d} | "
                f"val_loss({args.best_model_metric})={val_loss:.6f} | "
                f"swd={val_metrics['global_swd_mean']:.6f}±{val_metrics['global_swd_std']:.6f} | "
                f"mmd_z={val_metrics['mmd_rbf_z']:.6f} | nodeW1_med={val_metrics['node_w1_median']:.6f} | "
                f"cov={val_metrics['cov_rel_err']:.6f} | stationarity_gap={val_metrics['graph_stationarity_gap_abs']:.6f}"
            )
            if is_best:
                best_metric_value = val_loss
                best_summary = {
                    "epoch": epoch,
                    "metric_name": args.best_model_metric,
                    "metric_value": val_loss,
                    "checkpoint_path": str(checkpoint_path),
                    **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()},
                }
                torch.save(model.state_dict(), Path(schedule_dir) / "best_model.pt")
                write_json(Path(schedule_dir) / "best_model_summary.json", best_summary)
        else:
            row["is_best"] = False

        history.append(row)
        save_csv(Path(schedule_dir) / "train_history.csv", history)
        save_csv(Path(schedule_dir) / "validation_history.csv", validation_history)
        export_training_plot_data(schedule_dir, history, args.best_model_metric)

    torch.save(model.state_dict(), Path(schedule_dir) / "last_model.pt")
    if best_summary is not None and (Path(schedule_dir) / "best_model.pt").exists():
        model.load_state_dict(torch.load(Path(schedule_dir) / "best_model.pt", map_location=X_train.device))
    return model, history, validation_history, best_summary



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_mode",
        type=str,
        default="train_and_sample",
        choices=["forward_diagnostics", "train_only", "train_and_sample", "all"],
        help=(
            "forward_diagnostics: only build graph/data/schedules and export plot-ready arrays/metrics; "
            "train_only: train denoisers but skip final test sampling; "
            "train_and_sample: train and run final reverse generation; "
            "all: run forward diagnostics first, then train and sample."
        ),
    )
    p.add_argument(
        "--diagnostics_schedulers",
        type=str,
        default=None,
        help="Comma-separated schedules for forward diagnostics. Defaults to --schedulers.",
    )
    p.add_argument("--out", type=str, default="outputs_simple_compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--dataset_family", type=str, default="stationary_gaussian", choices=["stationary_gaussian", "smooth_bandlimited", "highfreq_mixture", "block_covariance", "localized_bursts"])
    p.add_argument("--stationary_spectrum", type=str, default="power_law", choices=["power_law", "bandlimited", "highfreq"])
    p.add_argument("--power_law_p", type=float, default=1.0)
    p.add_argument("--k_cut", type=int, default=16)
    p.add_argument("--strength", type=float, default=3.0)
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--num_bursts", type=int, default=2)
    p.add_argument("--burst_amp", type=float, default=3.0)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--schedulers", type=str, default="FCPS,Uniform,Cosine,MAS-SNR")
    p.add_argument("--methods", type=str, default="em,heun,ddpm,ddim")
    p.add_argument("--steps", type=str, default="25,50,100")
    p.add_argument("--t0", type=float, default=1e-3)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--sampler_init_mode", type=str, default="algorithm2", choices=["algorithm2", "stationary", "rho"])
    p.add_argument("--fcps_c0", type=float, default=2300.0)
    p.add_argument("--fcps_cmin", type=float, default=0.1)
    p.add_argument("--fcps_alpha", type=float, default=4.0)
    p.add_argument("--uniform_cbar_T", type=float, default=None, help="Terminal integrated clock for Uniform; defaults to fcps_c0 for endpoint matching.")
    p.add_argument("--cosine_cbar_T", type=float, default=None, help="Terminal integrated clock for Cosine; defaults to fcps_c0 for endpoint matching.")
    p.add_argument("--cosine_s", type=float, default=0.008, help="Nichol--Dhariwal cosine offset.")
    p.add_argument("--cosine_terminal_epsilon", type=float, default=1e-3, help="Finite endpoint truncation for the cosine cumulative attenuation.")
    p.add_argument(
        "--iso_cosine_s",
        type=float,
        default=None,
        help="Cosine offset of the graph-agnostic Isotropic-VP-Cosine control. Defaults to --cosine_s (previous behaviour). Table IV used 0.02.",
    )
    p.add_argument(
        "--iso_cosine_terminal_epsilon",
        type=float,
        default=None,
        help="Terminal truncation of the Isotropic-VP-Cosine control. Defaults to --cosine_terminal_epsilon. Table IV used 0.01.",
    )
    p.add_argument(
        "--fail_on_unstable_sampling",
        action="store_true",
        help=(
            "Abort the run when a reverse sampler produces a non-finite state. By default an unstable "
            "(schedule, method, steps) combination is recorded as a row with stable=False and empty metrics, "
            "and the remaining combinations and schedules still run. The default is what a step sweep for the "
            "coarsest stable grid needs."
        ),
    )
    p.add_argument("--r_max", type=float, default=1e4)
    p.add_argument("--r_min", type=float, default=None)
    p.add_argument("--match_terminal_energy", action="store_true")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--time_embed_dim", type=int, default=64)
    p.add_argument("--cheb_order", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--train_loss", type=str, default="x0", choices=["x0", "dsm"], help="Training objective: plain x0 reconstruction loss or DSM loss through x0 prediction.")
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--val_every_epochs", type=int, default=5)
    p.add_argument("--val_method", type=str, default="em", choices=["em", "heun", "ddpm", "ddim"])
    p.add_argument("--val_steps", type=int, default=50)
    p.add_argument("--val_num_samples", type=int, default=512)
    p.add_argument(
        "--evaluation_split",
        choices=["test", "val"],
        default="test",
        help=(
            "Reference split for the final reverse-sampling evaluation. 'test' (default) is the held-out "
            "test split used for the reported tables; 'val' scores the samples against the validation split "
            "instead, so that a hyperparameter sweep never computes any metric on the test split."
        ),
    )
    p.add_argument("--metric_num_projections", type=int, default=128)
    p.add_argument("--metric_num_seeds", type=int, default=5)
    p.add_argument("--best_model_metric", type=str, default="val_selection_score", choices=["val_selection_score", "global_swd_mean", "mmd_rbf_z", "cov_rel_err", "spec_cov_rel_err", "node_w1_median", "node_w1_q90", "band_ratio_l1", "smoothness_w1", "graph_stationarity_gap_abs", "ipr_w1", "gini_w1"])
    p.add_argument("--val_score_swd_weight", type=float, default=1.0)
    p.add_argument("--val_score_mmd_weight", type=float, default=1.0)
    p.add_argument("--val_score_cov_weight", type=float, default=1.0)
    p.add_argument("--val_score_spec_cov_weight", type=float, default=0.5)
    p.add_argument("--val_score_band_weight", type=float, default=1.0)
    p.add_argument("--val_score_nodew1_weight", type=float, default=0.5)
    p.add_argument("--val_score_nodew1q90_weight", type=float, default=0.5)
    p.add_argument("--val_score_smoothness_weight", type=float, default=0.25)
    p.add_argument("--val_score_stationarity_weight", type=float, default=0.25)
    p.add_argument("--val_score_localization_weight", type=float, default=0.25)
    p.add_argument("--lr_scheduler", type=str, default="plateau", choices=["plateau", "none"])
    p.add_argument("--lr_decay_factor", type=float, default=0.5)
    p.add_argument("--lr_patience_validations", type=int, default=2)
    p.add_argument("--lr_threshold", type=float, default=1e-4)
    p.add_argument("--train_t_min", type=float, default=1e-2)
    p.add_argument("--forward_diag_num_times", type=int, default=256)
    p.add_argument(
        "--forward_diag_num_extreme_modes",
        type=int,
        default=4,
        help="K: export the first K low-frequency and last K high-frequency modes for Figure-1 attenuation curves.",
    )
    p.add_argument("--forward_diag_num_samples", type=int, default=1024)
    p.add_argument("--forward_diag_t_min", type=float, default=1e-4)
    p.add_argument("--forward_diag_pmin", type=float, default=1e-10)
    p.add_argument("--stiffness_h_proxy", type=float, default=0.0)
    p.add_argument("--reverse_steps", type=str, default=None, help="Alias for --steps. Example: '25,50,100' or '100'.")
    p.add_argument("--band_low_frac", type=float, default=0.25, help="Fraction of lowest graph frequencies used for low-band generation diagnostics.")
    p.add_argument("--band_high_frac", type=float, default=0.25, help="Fraction of highest graph frequencies used for high-band generation diagnostics.")
    args = p.parse_args()

    print("\n=== Parsed arguments ===")
    print(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
    print("========================\n")

    set_all_seeds(args.seed)
    ensure_dir(args.out)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    _, W, L = rgg_knn_graph(n=args.n, k=args.k, seed=args.seed)
    Lg = shifted_laplacian(L, gamma=args.gamma)
    lam_gamma_np, V_np = eigh_sym(Lg)

    total = args.num_train + args.num_val + args.num_test
    X_all_np, data_meta = make_synthetic_dataset_numpy(
        family=args.dataset_family,
        V=V_np,
        lam_gamma=lam_gamma_np,
        N=total,
        seed=args.seed,
        spectrum=args.stationary_spectrum,
        power_law_p=args.power_law_p,
        k_cut=args.k_cut,
        strength=args.strength,
        num_blocks=args.num_blocks,
        rho=args.rho,
        num_bursts=args.num_bursts,
        burst_amp=args.burst_amp,
    )
    X_train_np, X_val_np, X_test_np = split_dataset(X_all_np, args.num_train, args.num_val, args.num_test)

    V = torch.tensor(V_np, device=device, dtype=dtype)
    lam_gamma = torch.tensor(lam_gamma_np, device=device, dtype=dtype)
    W_torch = torch.tensor(W, device=device, dtype=dtype)
    L_torch = torch.tensor(L, device=device, dtype=dtype)
    X_train = to_torch(X_train_np, device, dtype)
    X_val = to_torch(X_val_np, device, dtype)
    X_test = to_torch(X_test_np, device, dtype)
    s_k = ((X_train @ V) ** 2).mean(dim=0)

    sched_info = build_schedules(
        lam_gamma=lam_gamma_np,
        s_k=s_k.detach().cpu().numpy(),
        sigma=args.sigma,
        T=args.T,
        fcps_c0=args.fcps_c0,
        fcps_cmin=args.fcps_cmin,
        fcps_alpha=args.fcps_alpha,
        r_max=args.r_max,
        r_min=args.r_min,
        match_terminal_energy=args.match_terminal_energy,
        uniform_cbar_T=args.uniform_cbar_T,
        cosine_cbar_T=args.cosine_cbar_T,
        cosine_s=args.cosine_s,
        cosine_terminal_epsilon=args.cosine_terminal_epsilon,
        iso_cosine_s=args.iso_cosine_s,
        iso_cosine_terminal_epsilon=args.iso_cosine_terminal_epsilon,
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    steps_arg = args.reverse_steps if args.reverse_steps is not None else args.steps
    steps_list = [int(s.strip()) for s in steps_arg.split(",") if s.strip()]
    scheduler_names = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    diagnostics_scheduler_names = (
        [s.strip() for s in args.diagnostics_schedulers.split(",") if s.strip()]
        if args.diagnostics_schedulers is not None
        else scheduler_names
    )
    for name in scheduler_names + diagnostics_scheduler_names:
        if name not in sched_info:
            raise KeyError(f"Unknown scheduler {name!r}; available={sorted(k for k in sched_info if k != 'r_min')}")
    write_json(
        os.path.join(args.out, "run_config.json"),
        {
            "args": vars(args),
            "device_used": str(device),
            "data_meta": data_meta,
            "r_min_used": float(sched_info["r_min"]),
            "lambda_gamma_min": float(lam_gamma_np[0]),
            "lambda_gamma_max": float(lam_gamma_np[-1]),
            "lambda_gamma_condition_number": float(lam_gamma_np[-1] / lam_gamma_np[0]),
            # Every schedule that was built, so the isotropic controls' s and
            # terminal_epsilon are on record even when they differ from GAD Cosine.
            "schedules": {
                name: schedule_metadata(name, sched_info[name])
                for name in sched_info
                if name != "r_min" and name != "Isotropic-VP"  # skip the deprecated alias
            },
            "schedulers_trained": scheduler_names,
            "paired_invariants": {
                "same_graph_and_data_all_schedulers": True,
                "same_model_initialization_seed": args.seed,
                "same_minibatch_permutation_seed": args.seed + 456,
                "same_forward_noise_and_time_seed": args.seed + 123,
                "reseeded_at_start_of_every_schedule": True,
            },
            "unstable_sampling_policy": "abort" if args.fail_on_unstable_sampling else "record_row_with_stable_false",
            "plotting": {
                "generated_in_experiment": False,
                "reader": "plot_synthetic_exports.py",
                "figure2_export": "figure2_spectral_power_long.csv (this directory); aggregate across seeds with scripts/aggregate_results.py --table figure2",
            },
        },
    )

    if args.run_mode in {"forward_diagnostics", "all"}:
        print("\n=== Running forward diagnostics before training ===")
        run_forward_diagnostics(
            X_train_np=X_train_np,
            V_np=V_np,
            lam_gamma_np=lam_gamma_np,
            sigma=args.sigma,
            schedules=sched_info,
            scheduler_names=diagnostics_scheduler_names,
            dataset_name=args.dataset_family,
            out_dir=Path(args.out) / "forward_diagnostics",
            num_times=args.forward_diag_num_times,
            num_extreme_modes=args.forward_diag_num_extreme_modes,
            terminal_num_samples=args.forward_diag_num_samples,
            seed=args.seed,
            pmin=args.forward_diag_pmin,
            t_min=args.forward_diag_t_min,
            stiffness_h_proxy=args.stiffness_h_proxy,
        )
        print("=== Forward diagnostics complete ===\n")
        if args.run_mode == "forward_diagnostics":
            print("run_mode=forward_diagnostics, so training/sampling is skipped.")
            return

    results_rows: list[dict] = []
    figure2_rows: list[dict] = []
    for schedule_name in scheduler_names:
        schedule_obj = sched_info[schedule_name]
        schedule_dir = os.path.join(args.out, schedule_name.replace("/", "_"))
        model, _, _, best_summary = train_one_scheduler(
            schedule_name=schedule_name,
            schedule_obj=schedule_obj,
            X_train=X_train,
            X_val=X_val,
            X_val_np=X_val_np,
            X_test_np=X_test_np,
            V=V,
            V_np=V_np,
            lam_gamma=lam_gamma,
            W=W_torch,
            L_torch=L_torch,
            L_np=L,
            s_k=s_k,
            args=args,
            schedule_dir=schedule_dir,
        )
        if args.run_mode == "train_only":
            print(f"[{schedule_name}] training complete; final reverse generation skipped because run_mode=train_only.")
            continue

        for method in methods:
            for steps in steps_list:
                X_eval_np = X_val_np if args.evaluation_split == "val" else X_test_np
                num_eval = min(len(X_eval_np), args.val_num_samples)
                X_ref_np = X_eval_np[:num_eval]
                base_row = {
                    "dataset": args.dataset_family,
                    "seed": int(args.seed),
                    "schedule": schedule_name,
                    "method": method,
                    "steps": steps,
                    "nfe": steps,
                    "num_samples": int(num_eval),
                    "evaluation_split": args.evaluation_split,
                }
                if best_summary is not None:
                    base_row["best_epoch"] = int(best_summary["epoch"])
                try:
                    with torch.no_grad():
                        X_gen = reverse_sample(
                            model=model,
                            V=V,
                            lam_gamma=lam_gamma,
                            sigma=args.sigma,
                            schedule_name=schedule_name,
                            schedule_obj=schedule_obj,
                            s_k=s_k,
                            num_samples=num_eval,
                            steps=steps,
                            t0=args.t0,
                            T=args.T,
                            seed=args.seed + 80000 + steps,
                            method=method,
                            ddim_eta=args.ddim_eta,
                            init_mode=args.sampler_init_mode,
                        )
                    X_gen_np = X_gen.detach().cpu().numpy()
                    if not np.isfinite(X_gen_np).all():
                        raise FloatingPointError(
                            f"Non-finite generated samples: schedule={schedule_name}, method={method}, steps={steps}."
                        )
                except FloatingPointError as err:
                    if args.fail_on_unstable_sampling:
                        raise
                    print(f"[{schedule_name}] {args.evaluation_split} {method:>4s} steps={steps:4d} | UNSTABLE: {err}")
                    row = {**base_row, "stable": False, "failure_reason": str(err)}
                    for key in UNSTABLE_ROW_EMPTY_METRICS:
                        row[key] = float("nan")
                    results_rows.append(row)
                    save_csv(Path(args.out) / "results_summary.csv", results_rows)
                    continue

                metrics = summarize_generation_metrics(
                    X_gen_np,
                    X_ref_np,
                    V_np,
                    L,
                    sw_num_projections=args.metric_num_projections,
                    sw_seeds=range(args.metric_num_seeds),
                )
                prefix = f"{args.dataset_family}_{schedule_name}_{method}_{steps}steps"
                gen_diag = generation_diagnostics(
                    X_gen=X_gen_np,
                    X_ref=X_ref_np,
                    V=V_np,
                    lam_gamma=lam_gamma_np,
                    out_dir=Path(schedule_dir) / "generation_diagnostics",
                    prefix=prefix,
                    low_frac=args.band_low_frac,
                    high_frac=args.band_high_frac,
                )
                row = {**base_row, "stable": True, "failure_reason": "", **sanitize_metric_dict(metrics), **gen_diag}
                results_rows.append(row)
                figure2_rows.extend(
                    figure2_spectral_power_rows(
                        base_row=base_row,
                        X_gen=X_gen_np,
                        X_ref=X_ref_np,
                        V=V_np,
                        lam_gamma=lam_gamma_np,
                        low_frac=args.band_low_frac,
                        high_frac=args.band_high_frac,
                    )
                )
                print(
                    f"[{schedule_name}] {args.evaluation_split} {method:>4s} steps={steps:4d} | "
                    f"swd={row['global_swd_mean']:.6f} | mmd_z={row['mmd_rbf_z']:.6f} | "
                    f"cov={row['cov_rel_err']:.6f} | low_rel_l1={row['low_band_power_rel_l1']:.6f} | "
                    f"high_rel_l1={row['high_band_power_rel_l1']:.6f}"
                )
                # Write after every combination so a partial run is still usable.
                save_csv(Path(args.out) / "results_summary.csv", results_rows)
                save_csv(Path(args.out) / "figure2_spectral_power_long.csv", figure2_rows)

    if results_rows:
        save_csv(Path(args.out) / "results_summary.csv", results_rows)
    if figure2_rows:
        save_csv(Path(args.out) / "figure2_spectral_power_long.csv", figure2_rows)
        write_json(
            Path(args.out) / "figure2_spectral_power_manifest.json",
            figure2_manifest(args, lam_gamma_np, scheduler_names, methods, steps_list),
        )


if __name__ == "__main__":
    main()
