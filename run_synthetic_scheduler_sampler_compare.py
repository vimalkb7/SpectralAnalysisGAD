from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import make_synthetic_dataset_numpy, to_torch
from graphs import eigh_sym, rgg_knn_graph, shifted_laplacian
from metrics import summarize_generation_metrics
from model import TimeConditionedGNN
from samplers import reverse_sample
from schedules import build_schedules, channel_params_for_times_torch


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



def plot_training_curves(schedule_dir: str | Path, history: list[dict], best_model_metric: str) -> None:
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    train_losses = [float(r["train_loss"]) for r in history]
    val_epochs = [int(r["epoch"]) for r in history if r.get("did_validation", False)]
    val_losses = [float(r["val_loss"]) for r in history if r.get("did_validation", False)]
    lrs = [float(r.get("lr", np.nan)) for r in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", linewidth=1.5, label="train_loss")
    if val_epochs:
        plt.plot(val_epochs, val_losses, marker="s", linewidth=1.5, label=f"val_loss ({best_model_metric})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(schedule_dir) / "train_val_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lrs, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(schedule_dir) / "learning_rate_curve.png", dpi=160)
    plt.close()



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
        perm = torch.randperm(X_train.shape[0], device=X_train.device)
        train_loss_sum = 0.0
        count = 0

        for bi in range(num_batches):
            idx = perm[bi * args.batch_size : (bi + 1) * args.batch_size]
            xb = X_train[idx]
            if xb.shape[0] == 0:
                continue
            t = torch.rand((xb.shape[0],), generator=gen, device=X_train.device, dtype=X_train.dtype) * args.T
            pars = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, args.sigma, t)
            xt = sample_forward_pairwise_torch(xb, V, pars["a"], pars["nu"], generator=gen)
            # loss = spectral_dsm_loss(model, xt=xt, x0=xb, V=V, t=t, a=pars["a"], nu=pars["nu"])
            u = torch.rand((xb.shape[0],), generator=gen, device=X_train.device, dtype=X_train.dtype)
            t = args.train_t_min + (args.T - args.train_t_min) * u
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
        plot_training_curves(schedule_dir, history, args.best_model_metric)

    torch.save(model.state_dict(), Path(schedule_dir) / "last_model.pt")
    if best_summary is not None and (Path(schedule_dir) / "best_model.pt").exists():
        model.load_state_dict(torch.load(Path(schedule_dir) / "best_model.pt", map_location=X_train.device))
    return model, history, validation_history, best_summary



def main() -> None:
    p = argparse.ArgumentParser()
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
    p.add_argument("--schedulers", type=str, default="FCPS,MAS-SNR")
    p.add_argument("--methods", type=str, default="em,heun,ddpm,ddim")
    p.add_argument("--steps", type=str, default="25,50,100")
    p.add_argument("--t0", type=float, default=1e-3)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--sampler_init_mode", type=str, default="algorithm2", choices=["algorithm2", "stationary", "rho"])
    p.add_argument("--fcps_c0", type=float, default=2300.0)
    p.add_argument("--fcps_cmin", type=float, default=0.1)
    p.add_argument("--fcps_alpha", type=float, default=4.0)
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
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    steps_list = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    scheduler_names = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    write_json(os.path.join(args.out, "run_config.json"), {"args": vars(args), "device_used": str(device), "data_meta": data_meta, "r_min_used": float(sched_info["r_min"])})

    results_rows: list[dict] = []
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
        for method in methods:
            for steps in steps_list:
                with torch.no_grad():
                    X_gen = reverse_sample(
                        model=model,
                        V=V,
                        lam_gamma=lam_gamma,
                        sigma=args.sigma,
                        schedule_name=schedule_name,
                        schedule_obj=schedule_obj,
                        s_k=s_k,
                        num_samples=min(len(X_test_np), args.val_num_samples),
                        steps=steps,
                        t0=args.t0,
                        T=args.T,
                        seed=args.seed + 80000 + steps,
                        method=method,
                        ddim_eta=args.ddim_eta,
                        init_mode=args.sampler_init_mode,
                    )
                metrics = summarize_generation_metrics(
                    X_gen.detach().cpu().numpy(),
                    X_test_np[: min(len(X_test_np), args.val_num_samples)],
                    V_np,
                    L,
                    sw_num_projections=args.metric_num_projections,
                    sw_seeds=range(args.metric_num_seeds),
                )
                row = {"schedule": schedule_name, "method": method, "steps": steps, **sanitize_metric_dict(metrics)}
                if best_summary is not None:
                    row["best_epoch"] = int(best_summary["epoch"])
                results_rows.append(row)
                print(f"[{schedule_name}] test {method:>4s} steps={steps:4d} | swd={row['global_swd_mean']:.6f} | mmd_z={row['mmd_rbf_z']:.6f}")
        save_csv(Path(args.out) / "results_summary.csv", results_rows)


if __name__ == "__main__":
    main()
