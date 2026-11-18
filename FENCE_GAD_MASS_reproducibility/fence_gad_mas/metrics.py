from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

Aggregation = Literal["median", "mean"]


def _point_prediction(samples: torch.Tensor, aggregation: Aggregation) -> torch.Tensor:
    if aggregation == "median":
        return samples.median(dim=1).values
    if aggregation == "mean":
        return samples.mean(dim=1)
    raise ValueError(f"Unknown aggregation={aggregation!r}; use 'median' or 'mean'.")


def evaluate_sampling_metrics(
    model,
    data_loader,
    std,
    mean,
    *,
    nsample: int = 10,
    max_batches: int | None = None,
    mape_eps: float = 1e-4,
    aggregation: Aggregation = "median",
    fence_compatible_mape: bool = True,
    desc: str = "sampling-evaluation",
):
    model.eval()
    device = next(model.parameters()).device
    scaler = torch.as_tensor(std, device=device, dtype=torch.float32)
    mean_scaler = torch.as_tensor(mean, device=device, dtype=torch.float32)

    mse_total = 0.0
    mae_total = 0.0
    mape_total = 0.0
    eval_total = 0.0
    valid_mape_total = 0.0
    nonfinite_total = 0
    batches = 0

    with torch.no_grad():
        iterator = tqdm(data_loader, desc=desc, mininterval=5.0, maxinterval=50.0)
        for batch in iterator:
            if max_batches is not None and batches >= max_batches:
                break

            samples, target, eval_points, _, _ = model.evaluate(batch, nsample)
            prediction = _point_prediction(samples, aggregation)

            diff_phys = (prediction - target) * scaler
            mask = eval_points > 0
            mask_f = mask.to(diff_phys.dtype)
            nonfinite_total += int((mask & ~torch.isfinite(prediction)).sum().item())

            mse_total += float(((diff_phys * mask_f) ** 2).sum().item())
            mae_total += float((torch.abs(diff_phys) * mask_f).sum().item())
            eval_count = float(mask_f.sum().item())
            eval_total += eval_count

            target_phys = target * scaler + mean_scaler
            if fence_compatible_mape:
                valid = mask & (target_phys > float(mape_eps))
                ratio = torch.zeros_like(diff_phys)
                ratio[valid] = torch.abs(diff_phys[valid]) / target_phys[valid]
                ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                mape_total += float((ratio * mask_f).sum().item())
            else:
                valid = mask & (torch.abs(target_phys) > float(mape_eps))
                ratio = torch.zeros_like(diff_phys)
                ratio[valid] = torch.abs(diff_phys[valid]) / torch.abs(target_phys[valid])
                mape_total += float(ratio[valid].sum().item())
            valid_mape_total += float(valid.sum().item())

            batches += 1
            iterator.set_postfix(
                rmse=np.sqrt(mse_total / max(eval_total, 1.0)),
                mae=mae_total / max(eval_total, 1.0),
            )

    mape_denom = eval_total if fence_compatible_mape else valid_mape_total
    metrics = {
        "RMSE": float(np.sqrt(mse_total / max(eval_total, 1.0))),
        "MAE": float(mae_total / max(eval_total, 1.0)),
        "MAPE": float(mape_total / max(mape_denom, 1.0)),
        "batches": int(batches),
        "nsample": int(nsample),
        "eval_points": float(eval_total),
        "valid_mape_points": float(valid_mape_total),
        "nonfinite_predictions": int(nonfinite_total),
        "aggregation": aggregation,
        "fence_compatible_mape": bool(fence_compatible_mape),
    }
    if nonfinite_total:
        metrics["MAPE"] = float("nan")
        metrics["MAE"] = float("nan")
        metrics["RMSE"] = float("nan")
        logging.warning("%s: %d evaluation points have non-finite predictions (diverged sampler).", desc, nonfinite_total)
    logging.info("%s metrics: %s", desc, metrics)
    return metrics



def evaluate_sampling_metrics_mean(model, data_loader, std, mean, nsample=10, max_batches=None,
                                   mape_eps=1e-4, desc="mean-sampling-evaluation"):
    return evaluate_sampling_metrics(
        model,
        data_loader,
        std,
        mean,
        nsample=nsample,
        max_batches=max_batches,
        mape_eps=mape_eps,
        aggregation="mean",
        fence_compatible_mape=True,
        desc=desc,
    )
