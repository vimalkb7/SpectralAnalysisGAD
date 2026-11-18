from __future__ import annotations

import configparser
import logging
import torch

from .data import build_exp34_clean_loaders
from .graph import load_graph_for_fence
from .model import GADMASFENCE_Traffic
from .spectral_power import load_or_estimate_spectral_power


def build_model_and_data(cfg: configparser.ConfigParser, device: str):
    loaders = build_exp34_clean_loaders(cfg)
    graph = load_graph_for_fence(cfg, target_dim=loaders["target_dim"], device=device, dtype=torch.float32)
    # Make physical-scale task losses reproducible from the config/model object.
    cfg["train"]["data_train_mean"] = str(loaders["mean"])
    cfg["train"]["data_train_std"] = str(loaders["std"])
    spectral_power = load_or_estimate_spectral_power(
        cfg,
        graph,
        loaders["spectral_train"],
        device=device,
        residual_loader=loaders["phase2_train"],
    )
    logging.info(
        "train-only spectral power mode=%s range=%.6e..%.6e",
        cfg["gad"].get("spectral_power_mode", "clean_residual_blend"),
        float(spectral_power.min()), float(spectral_power.max()),
    )
    model = GADMASFENCE_Traffic(cfg, loaders["target_dim"], device, graph, spectral_power).to(device)
    return model, loaders, spectral_power
