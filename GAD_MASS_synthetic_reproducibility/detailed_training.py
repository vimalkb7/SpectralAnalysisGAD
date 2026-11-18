from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from diagnostics import generation_diagnostics
from metrics import summarize_generation_metrics
from model import TimeConditionedGNN
from samplers import method_nfe, reverse_sample
from schedules import channel_params_for_times_torch, schedule_metadata


def _ensure(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def write_json(path: str | Path, payload) -> None:
    _ensure(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    _ensure(Path(path).parent)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


@dataclass
class TrainingConfig:
    hidden_dim: int = 64
    num_layers: int = 4
    time_embed_dim: int = 64
    cheb_order: int = 3
    dropout: float = 0.0
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    batch_size: int = 128
    num_epochs: int = 40
    train_t_min: float = 1e-2
    val_every_epochs: int = 2
    val_num_samples: int = 1024
    grad_clip_norm: float = 1.0
    lr_decay_factor: float = 0.5
    lr_patience_validations: int = 3
    lr_threshold: float = 1e-4
    model_seed: int = 3001
    train_seed: int = 4001
    validation_seed: int = 5001
    score_nu_floor: float = 1e-12
    score_clip: float | None = None


@dataclass
class EvaluationConfig:
    methods: tuple[str, ...] = ("em", "ddim", "dpmpp2m")
    steps: tuple[int, ...] = (10, 25, 50, 100, 250)
    method_steps: dict[str, tuple[int, ...]] | None = None
    sample_seeds: tuple[int, ...] = (6001, 6002, 6003, 6004, 6005)
    num_samples: int = 1024
    t0: float = 1e-3
    T: float = 1.0
    ddim_eta: float = 0.0
    init_mode: str = "algorithm2"
    metric_num_projections: int = 128
    metric_num_seeds: int = 5
    plot_export_steps: tuple[int, ...] = (50, 250, 1000, 2500, 3500)
    plot_export_sample_seeds: tuple[int, ...] = (6001,)


def _make_model(L: torch.Tensor, W: torch.Tensor, dtype: torch.dtype, device: torch.device, cfg: TrainingConfig):
    # Reset before every scheduler: paired parameter initialization is a hard invariant.
    torch.manual_seed(int(cfg.model_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.model_seed))
    model = TimeConditionedGNN(
        W=W,
        L=L,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        time_embed_dim=cfg.time_embed_dim,
        cheb_order=cfg.cheb_order,
        dropout=cfg.dropout,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.score_nu_floor = float(cfg.score_nu_floor)
    model.score_clip = cfg.score_clip
    return model


def _corrupt_from_u_z(
    X0: torch.Tensor,
    V: torch.Tensor,
    schedule_name: str,
    schedule_obj,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    u: torch.Tensor,
    z: torch.Tensor,
    t_min: float,
    T: float,
):
    t = float(t_min) + (float(T) - float(t_min)) * u
    pars = channel_params_for_times_torch(schedule_name, schedule_obj, lam_gamma, s_k, sigma, t)
    x0_tilde = X0 @ V
    xt_tilde = pars["a"] * x0_tilde + torch.sqrt(torch.clamp(pars["nu"], min=0.0)) * z
    return xt_tilde @ V.T, t, pars


@torch.no_grad()
def denoising_validation_loss(
    model,
    X_val: torch.Tensor,
    V: torch.Tensor,
    schedule_name: str,
    schedule_obj,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    cfg: TrainingConfig,
    T: float,
) -> float:
    """Sampler-independent, repeatable checkpoint-selection objective."""
    m = min(int(cfg.val_num_samples), X_val.shape[0])
    x0 = X_val[:m]
    g = torch.Generator(device=X_val.device)
    g.manual_seed(int(cfg.validation_seed))
    u = torch.rand((m,), generator=g, device=X_val.device, dtype=X_val.dtype)
    z = torch.randn((m, X_val.shape[1]), generator=g, device=X_val.device, dtype=X_val.dtype)
    xt, t, _ = _corrupt_from_u_z(
        x0, V, schedule_name, schedule_obj, lam_gamma, s_k, sigma, u, z, cfg.train_t_min, T
    )
    was_training = model.training
    model.eval()
    pred = model(xt, V, t)
    loss = torch.mean((pred - x0) ** 2)
    if was_training:
        model.train()
    return float(loss.item())


def train_paired_denoiser(
    schedule_name: str,
    schedule_obj,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    V: torch.Tensor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    W: torch.Tensor,
    L: torch.Tensor,
    sigma: float,
    T: float,
    cfg: TrainingConfig,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    _ensure(out_dir)
    model = _make_model(L, W, X_train.dtype, X_train.device, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_decay_factor,
        patience=cfg.lr_patience_validations,
        threshold=cfg.lr_threshold,
        min_lr=cfg.min_lr,
    )

    # Dedicated generators ensure identical minibatch order, times and Gaussian
    # innovations across schedulers. Model/dropout RNG cannot perturb them.
    g_perm = torch.Generator(device=X_train.device)
    g_perm.manual_seed(int(cfg.train_seed))
    g_corrupt = torch.Generator(device=X_train.device)
    g_corrupt.manual_seed(int(cfg.train_seed) + 1)

    best_val = float("inf")
    best_epoch = -1
    history: list[dict] = []
    n_batches = max(1, int(np.ceil(X_train.shape[0] / cfg.batch_size)))

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0], generator=g_perm, device=X_train.device)
        loss_sum = 0.0
        seen = 0
        for bi in range(n_batches):
            idx = perm[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]
            if idx.numel() == 0:
                continue
            x0 = X_train[idx]
            u = torch.rand((x0.shape[0],), generator=g_corrupt, device=X_train.device, dtype=X_train.dtype)
            z = torch.randn((x0.shape[0], X_train.shape[1]), generator=g_corrupt, device=X_train.device, dtype=X_train.dtype)
            xt, t, _ = _corrupt_from_u_z(
                x0, V, schedule_name, schedule_obj, lam_gamma, s_k, sigma, u, z, cfg.train_t_min, T
            )
            pred = model(xt, V, t)
            loss = torch.mean((pred - x0) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.item()) * x0.shape[0]
            seen += x0.shape[0]

        train_loss = loss_sum / max(seen, 1)
        do_val = epoch % cfg.val_every_epochs == 0 or epoch == cfg.num_epochs
        val_loss = None
        is_best = False
        if do_val:
            val_loss = denoising_validation_loss(
                model, X_val, V, schedule_name, schedule_obj, lam_gamma, s_k, sigma, cfg, T
            )
            lr_sched.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                is_best = True
                torch.save(model.state_dict(), out_dir / "best_model.pt")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "denoising_val_loss": "" if val_loss is None else val_loss,
                "is_best": is_best,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        write_csv(out_dir / "train_history.csv", history)
        if do_val:
            print(
                f"[{schedule_name}] epoch={epoch:03d} train={train_loss:.6g} "
                f"denoise_val={val_loss:.6g} best={best_val:.6g}"
            )

    torch.save(model.state_dict(), out_dir / "last_model.pt")
    if best_epoch > 0:
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=X_train.device))
    summary = {
        "schedule": schedule_name,
        "schedule_metadata": schedule_metadata(schedule_name, schedule_obj),
        "checkpoint_selection": "fixed sampler-independent denoising validation MSE",
        "best_epoch": best_epoch,
        "best_denoising_val_loss": best_val,
        "training_config": asdict(cfg),
        "sigma": float(sigma),
        "T": float(T),
    }
    write_json(out_dir / "training_summary.json", summary)
    return model, summary


def evaluate_model_nfe_sweep(
    model,
    schedule_name: str,
    schedule_obj,
    X_ref_np: np.ndarray,
    V_np: np.ndarray,
    L_np: np.ndarray,
    V: torch.Tensor,
    lam_gamma: torch.Tensor,
    s_k: torch.Tensor,
    sigma: float,
    cfg: EvaluationConfig,
    out_dir: str | Path,
    shared_fields: dict | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    _ensure(out_dir)
    rows: list[dict] = []
    n_eval = min(int(cfg.num_samples), len(X_ref_np))
    X_ref = X_ref_np[:n_eval]
    shared_fields = dict(shared_fields or {})

    model.eval()
    for method in cfg.methods:
        method_steps = cfg.steps
        if cfg.method_steps is not None and method in cfg.method_steps:
            method_steps = cfg.method_steps[method]
        for steps in method_steps:
            for sample_seed in cfg.sample_seeds:
                try:
                    X_gen = reverse_sample(
                        model=model,
                        V=V,
                        lam_gamma=lam_gamma,
                        sigma=sigma,
                        schedule_name=schedule_name,
                        schedule_obj=schedule_obj,
                        s_k=s_k,
                        num_samples=n_eval,
                        steps=int(steps),
                        t0=cfg.t0,
                        T=cfg.T,
                        seed=int(sample_seed),
                        method=method,
                        ddim_eta=cfg.ddim_eta,
                        init_mode=cfg.init_mode,
                    )
                    X_gen_np = X_gen.detach().cpu().numpy()
                    metrics = summarize_generation_metrics(
                        X_gen_np,
                        X_ref,
                        V_np,
                        L_np,
                        sw_num_projections=cfg.metric_num_projections,
                        sw_seeds=range(cfg.metric_num_seeds),
                    )
                    keep_samples = (
                        int(steps) in set(cfg.plot_export_steps)
                        and int(sample_seed) in set(cfg.plot_export_sample_seeds)
                    )
                    diag = generation_diagnostics(
                        X_gen_np,
                        X_ref,
                        V_np,
                        lam_gamma.detach().cpu().numpy(),
                        out_dir / "generation_diagnostics",
                        prefix=f"{schedule_name}_{method}_{steps}_{sample_seed}",
                        include_samples=keep_samples,
                    )
                    stable = True
                    error = ""
                except FloatingPointError as exc:
                    metrics = {}
                    diag = {}
                    stable = False
                    error = str(exc)
                row = {
                    **shared_fields,
                    "schedule": schedule_name,
                    "method": method,
                    "steps": int(steps),
                    "nfe": method_nfe(method, int(steps)),
                    "sample_seed": int(sample_seed),
                    "stable": stable,
                    "error": error,
                    **metrics,
                    **diag,
                }
                rows.append(row)
                print(
                    f"[{schedule_name}] {method} steps={steps} nfe={row['nfe']} seed={sample_seed} "
                    f"stable={stable} swd={row.get('global_swd_mean', float('nan')):.6g}"
                )
                write_csv(out_dir / "results_raw.csv", rows)

    array_rows = [
        {
            "schedule": r.get("schedule", schedule_name),
            "method": r.get("method", ""),
            "steps": r.get("steps", ""),
            "nfe": r.get("nfe", ""),
            "sample_seed": r.get("sample_seed", ""),
            "stable": r.get("stable", False),
            "generation_plot_npz": r.get("generation_plot_npz", ""),
            "full_spectrum_power_matching_csv": r.get("full_spectrum_power_matching_csv", ""),
        }
        for r in rows
        if r.get("generation_plot_npz")
    ]
    write_csv(out_dir / "generation_array_manifest.csv", array_rows)
    write_json(
        out_dir / "generation_array_manifest.json",
        {
            "schema_version": 1,
            "description": "Plot-ready per-mode spectral arrays for every stable sampler/NFE run; no figures generated.",
            "entries": array_rows,
        },
    )
    return rows
