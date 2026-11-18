from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch

PowerMode = Literal[
    "second_moment",
    "centered",
    "blend",
    "conditional_residual",
    "clean_residual_blend",
]


def _collect_first_moment(graph, train_loader, device, max_batches: int = 0):
    sums = None
    count = 0
    with torch.no_grad():
        for b, batch in enumerate(train_loader):
            x = batch["observed_data"].to(device).float().permute(0, 2, 1)  # [B,N,T]
            x_spec = graph.to_spectral(x)
            val = x_spec.sum(dim=(0, 2))
            sums = val if sums is None else sums + val
            count += x_spec.shape[0] * x_spec.shape[2]
            if max_batches and (b + 1) >= max_batches:
                break
    if sums is None or count == 0:
        raise RuntimeError("Could not estimate spectral mean: empty loader")
    return sums / float(count), count


def estimate_spectral_power_from_loader(
    graph,
    train_loader,
    device,
    max_batches: int = 0,
    min_power: float = 1e-6,
    mode: str = "second_moment",
    blend: float = 0.5,
) -> np.ndarray:
    """Estimate train-only clean graph-mode power."""
    mode = str(mode).lower()
    if mode not in {"second_moment", "centered", "blend"}:
        raise ValueError(f"Clean power estimator does not support mode={mode}")
    if not (0.0 <= float(blend) <= 1.0):
        raise ValueError("blend must be in [0,1]")

    mean_spec = None
    if mode in {"centered", "blend"}:
        mean_spec, _ = _collect_first_moment(graph, train_loader, device, max_batches=max_batches)

    sums = None
    count = 0
    with torch.no_grad():
        for b, batch in enumerate(train_loader):
            x = batch["observed_data"].to(device).float().permute(0, 2, 1)
            x_spec = graph.to_spectral(x)
            second = (x_spec ** 2).sum(dim=(0, 2))
            sums = second if sums is None else sums + second
            count += x_spec.shape[0] * x_spec.shape[2]
            if max_batches and (b + 1) >= max_batches:
                break

    if sums is None or count == 0:
        raise RuntimeError("Could not estimate spectral power: empty loader")

    second_moment = sums / float(count)
    if mean_spec is None:
        power = second_moment
    else:
        centered = torch.clamp(second_moment - mean_spec.pow(2), min=0.0)
        power = centered if mode == "centered" else (1.0 - float(blend)) * centered + float(blend) * second_moment

    arr = power.detach().cpu().numpy().astype(np.float64)
    return np.maximum(arr, float(min_power)).astype(np.float32)


def _context_proxy_prediction(
    graph,
    x: torch.Tensor,
    cond_mask: torch.Tensor,
    *,
    spatial_weight: float = 0.65,
    temporal_weight: float = 0.35,
) -> torch.Tensor:
    if x.shape != cond_mask.shape:
        raise ValueError(f"x/mask mismatch: {x.shape} vs {cond_mask.shape}")
    mask = cond_mask.to(x.dtype)
    observed = x * mask

    # Temporal estimate per node from observed time points.
    tden = mask.sum(dim=2, keepdim=True)
    temporal = observed.sum(dim=2, keepdim=True) / tden.clamp(min=1.0)
    temporal = temporal.expand_as(x)
    temporal_ok = (tden > 0).expand_as(x)

    # Spatial estimate at each time from graph neighbors that remain observed.
    A = graph.A.to(device=x.device, dtype=x.dtype)
    spatial_num = torch.einsum("nm,bmt->bnt", A, observed)
    spatial_den = torch.einsum("nm,bmt->bnt", A, mask)
    spatial = spatial_num / spatial_den.clamp(min=1e-6)
    spatial_ok = spatial_den > 1e-6

    # Global timewise fallback, still using only observed values.
    gden = mask.sum(dim=1, keepdim=True)
    global_t = observed.sum(dim=1, keepdim=True) / gden.clamp(min=1.0)
    global_t = global_t.expand_as(x)

    sw = float(spatial_weight)
    tw = float(temporal_weight)
    if sw < 0 or tw < 0 or (sw + tw) <= 0:
        raise ValueError("spatial/temporal proxy weights must be nonnegative with positive sum")
    sw, tw = sw / (sw + tw), tw / (sw + tw)

    both = spatial_ok & temporal_ok
    pred = global_t.clone()
    pred = torch.where(temporal_ok, temporal, pred)
    pred = torch.where(spatial_ok, spatial, pred)
    pred = torch.where(both, sw * spatial + tw * temporal, pred)
    # Preserve observed entries exactly; residual power is measured on hidden entries.
    return mask * x + (1.0 - mask) * pred


def estimate_conditional_residual_power_from_loader(
    graph,
    train_loader,
    device,
    *,
    max_batches: int = 0,
    min_power: float = 1e-6,
    spatial_weight: float = 0.65,
    temporal_weight: float = 0.35,
    rescale_missing: bool = True,
) -> np.ndarray:
    sums = None
    count = 0
    with torch.no_grad():
        for b, batch in enumerate(train_loader):
            x = batch["observed_data"].to(device).float().permute(0, 2, 1)  # [B,N,T]
            cond = batch["gt_mask"].to(device).float().permute(0, 2, 1)
            pred = _context_proxy_prediction(
                graph,
                x,
                cond,
                spatial_weight=spatial_weight,
                temporal_weight=temporal_weight,
            )
            missing = 1.0 - cond
            residual = (x - pred) * missing
            if rescale_missing:
                miss_frac = missing.mean(dim=(1, 2), keepdim=True).clamp(min=1e-3)
                residual = residual / torch.sqrt(miss_frac)
            res_spec = graph.to_spectral(residual)
            second = res_spec.pow(2).sum(dim=(0, 2))
            sums = second if sums is None else sums + second
            count += res_spec.shape[0] * res_spec.shape[2]
            if max_batches and (b + 1) >= max_batches:
                break
    if sums is None or count == 0:
        raise RuntimeError("Could not estimate conditional residual spectral power: empty loader")
    arr = (sums / float(count)).detach().cpu().numpy().astype(np.float64)
    return np.maximum(arr, float(min_power)).astype(np.float32)


def estimate_effective_spectral_power(
    graph,
    clean_loader,
    residual_loader,
    device,
    *,
    mode: PowerMode,
    min_power: float = 1e-6,
    max_batches: int = 0,
    clean_mode: str = "centered",
    clean_blend: float = 0.5,
    residual_alpha: float = 0.75,
    residual_spatial_weight: float = 0.65,
    residual_temporal_weight: float = 0.35,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mode = str(mode).lower()
    if mode in {"second_moment", "centered", "blend"}:
        clean = estimate_spectral_power_from_loader(
            graph, clean_loader, device, max_batches=max_batches, min_power=min_power,
            mode=mode, blend=clean_blend,
        )
        return clean, {"clean": clean}

    if residual_loader is None:
        raise ValueError(f"mode={mode} requires a residual/SC-TC training loader")
    residual = estimate_conditional_residual_power_from_loader(
        graph, residual_loader, device,
        max_batches=max_batches,
        min_power=min_power,
        spatial_weight=residual_spatial_weight,
        temporal_weight=residual_temporal_weight,
    )
    if mode == "conditional_residual":
        return residual, {"residual": residual}
    if mode != "clean_residual_blend":
        raise ValueError(f"Unknown spectral power mode={mode}")
    if not (0.0 <= float(residual_alpha) <= 1.0):
        raise ValueError("spectral_power_residual_alpha must lie in [0,1]")
    clean = estimate_spectral_power_from_loader(
        graph, clean_loader, device, max_batches=max_batches, min_power=min_power,
        mode=clean_mode, blend=clean_blend,
    )
    effective = (1.0 - float(residual_alpha)) * clean + float(residual_alpha) * residual
    effective = np.maximum(effective, float(min_power)).astype(np.float32)
    return effective, {"clean": clean, "residual": residual}


def _read_cache(path: str | Path, expected_mode: str, min_power: float, residual_alpha: float):
    obj = np.load(path, allow_pickle=False)
    if not isinstance(obj, np.lib.npyio.NpzFile):
        if expected_mode != "second_moment":
            raise ValueError(f"Legacy .npy spectral cache cannot certify mode={expected_mode}; recompute it.")
        return np.maximum(np.asarray(obj, dtype=np.float64), min_power).astype(np.float32)
    key = "s_k" if "s_k" in obj else ("spectral_power" if "spectral_power" in obj else obj.files[0])
    cached_mode = str(obj["mode"].item()) if "mode" in obj else "second_moment"
    if cached_mode != expected_mode:
        raise ValueError(f"Spectral cache mode mismatch: cache={cached_mode}, config={expected_mode}. Recompute it.")
    if expected_mode == "clean_residual_blend" and "residual_alpha" in obj:
        cached_alpha = float(obj["residual_alpha"].item())
        if abs(cached_alpha - float(residual_alpha)) > 1e-12:
            raise ValueError(f"Residual-alpha cache mismatch: {cached_alpha} vs {residual_alpha}. Recompute it.")
    return np.maximum(np.asarray(obj[key], dtype=np.float64), min_power).astype(np.float32)


def load_or_estimate_spectral_power(config, graph, clean_loader, device, residual_loader=None) -> np.ndarray:
    gad = config["gad"]
    path = str(gad.get("spectral_power_path", "")).strip()
    min_power = float(gad.get("spectral_power_min", 1e-6))
    mode = str(gad.get("spectral_power_mode", "clean_residual_blend")).lower()
    residual_alpha = float(gad.get("spectral_power_residual_alpha", 0.75))

    if path and os.path.exists(path):
        return _read_cache(path, mode, min_power, residual_alpha)

    s, parts = estimate_effective_spectral_power(
        graph,
        clean_loader,
        residual_loader,
        device,
        mode=mode,
        min_power=min_power,
        max_batches=int(gad.get("spectral_power_max_batches", 0)),
        clean_mode=str(gad.get("spectral_power_clean_mode", "centered")),
        clean_blend=float(gad.get("spectral_power_blend", 0.5)),
        residual_alpha=residual_alpha,
        residual_spatial_weight=float(gad.get("residual_proxy_spatial_weight", 0.65)),
        residual_temporal_weight=float(gad.get("residual_proxy_temporal_weight", 0.35)),
    )
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "s_k": s,
            "mode": np.asarray(mode),
            "residual_alpha": np.asarray(residual_alpha),
            "source": np.asarray("train_only_clean_and_sc_tc_context_residual"),
        }
        for name, arr in parts.items():
            payload[f"{name}_power"] = arr
        np.savez(path, **payload)
    return s
