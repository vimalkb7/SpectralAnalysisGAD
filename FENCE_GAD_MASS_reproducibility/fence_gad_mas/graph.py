from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd
import torch


def sanitize_adjacency(
    A: np.ndarray,
    sym_mode: str = "avg",
    remove_self_loops: bool = True,
    clip_negative: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("Adjacency contains NaN or Inf")
    A[np.abs(A) < eps] = 0.0
    if sym_mode == "avg":
        A = 0.5 * (A + A.T)
    elif sym_mode == "max":
        A = np.maximum(A, A.T)
    else:
        raise ValueError(f"Unknown sym_mode={sym_mode}")
    if remove_self_loops:
        np.fill_diagonal(A, 0.0)
    if clip_negative:
        A = np.maximum(A, 0.0)
    return A


def adjacency_from_csv(path: str | Path, num_nodes: int, sigma2: float = 0.1, threshold: float = 0.0) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    src_col = cols_lower.get("from") or cols_lower.get("from_node") or cols_lower.get("src") or df.columns[0]
    dst_col = cols_lower.get("to") or cols_lower.get("to_node") or cols_lower.get("dst") or df.columns[1]
    dist_col = cols_lower.get("distance") or cols_lower.get("cost") or cols_lower.get("dist")
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if dist_col is None:
        for _, row in df.iterrows():
            i, j = int(row[src_col]), int(row[dst_col])
            if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
                A[i, j] = 1.0
                A[j, i] = 1.0
        return sanitize_adjacency(A)

    src = df[src_col].astype(int).to_numpy()
    dst = df[dst_col].astype(int).to_numpy()
    dist = df[dist_col].astype(float).to_numpy()
    valid = np.isfinite(dist) & (dist >= 0)
    if valid.sum() == 0:
        raise ValueError(f"No valid distances in {path}")
    std = float(np.std(dist[valid]))
    std = max(std, 1e-12)
    weights = np.exp(-np.square(dist / std) / max(float(sigma2), 1e-12))
    weights[weights < threshold] = 0.0
    for i, j, w in zip(src, dst, weights):
        if 0 <= i < num_nodes and 0 <= j < num_nodes and i != j:
            A[i, j] = max(A[i, j], float(w))
            A[j, i] = max(A[j, i], float(w))
    return sanitize_adjacency(A)


def adjacency_from_npy(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        for key in ("adj", "A", "adjacency", "data"):
            if key in arr:
                return sanitize_adjacency(arr[key])
        first = arr.files[0]
        return sanitize_adjacency(arr[first])
    return sanitize_adjacency(arr)


def build_laplacian(A: np.ndarray, normalized: bool = False) -> np.ndarray:
    A = sanitize_adjacency(A)
    deg = A.sum(axis=1)
    if normalized:
        inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        L = np.eye(A.shape[0], dtype=np.float64) - np.diag(inv_sqrt) @ A @ np.diag(inv_sqrt)
    else:
        L = np.diag(deg) - A
    return 0.5 * (L + L.T)



def log_laplacian_eigenspectrum(
    lam: np.ndarray,
    lam_gamma: np.ndarray,
    gamma: float,
    neg_tol: float = 1e-10,
) -> None:
    lam = np.asarray(lam, dtype=np.float64)
    lam_gamma = np.asarray(lam_gamma, dtype=np.float64)

    raw_negative_modes = np.where(lam < -neg_tol)[0]
    raw_tiny_negative_modes = np.where((lam < 0.0) & (lam >= -neg_tol))[0]
    shifted_negative_modes = np.where(lam_gamma < -neg_tol)[0]
    shifted_nonpositive_modes = np.where(lam_gamma <= 0.0)[0]

    logging.info("=" * 100)
    logging.info("GRAPH LAPLACIAN EIGENVALUE DIAGNOSTICS")
    logging.info("=" * 100)
    logging.info("gamma = %.16e", gamma)
    logging.info("num modes = %d", len(lam))

    logging.info("raw lambda min = %.16e", float(lam.min()))
    logging.info("raw lambda max = %.16e", float(lam.max()))
    logging.info("lambda_gamma min = %.16e", float(lam_gamma.min()))
    logging.info("lambda_gamma max = %.16e", float(lam_gamma.max()))

    logging.info("num raw significantly negative modes, lambda < -%.1e: %d", neg_tol, len(raw_negative_modes))
    logging.info("num raw tiny negative modes, -%.1e <= lambda < 0: %d", neg_tol, len(raw_tiny_negative_modes))
    logging.info("num shifted negative modes, lambda_gamma < -%.1e: %d", neg_tol, len(shifted_negative_modes))
    logging.info("num shifted non-positive modes, lambda_gamma <= 0: %d", len(shifted_nonpositive_modes))

    if len(raw_negative_modes) > 0:
        logging.warning("Raw Laplacian has significantly negative eigenvalues at modes: %s", raw_negative_modes.tolist())

    if len(raw_tiny_negative_modes) > 0:
        logging.info(
            "Raw Laplacian has tiny numerical negative eigenvalues at modes: %s",
            raw_tiny_negative_modes.tolist(),
        )

    if len(shifted_nonpositive_modes) > 0:
        logging.error(
            "Some gamma-shifted eigenvalues are still non-positive at modes: %s",
            shifted_nonpositive_modes.tolist(),
        )
    else:
        logging.info("All gamma-shifted eigenvalues are strictly positive.")

    # The original experiment logs printed every eigenvalue.  For a clean
    # reproduction package, keep logs concise by default.  Set
    # GAD_LOG_FULL_SPECTRUM=1 to recover the full per-mode table.
    import os
    if os.environ.get("GAD_LOG_FULL_SPECTRUM", "0").lower() in {"1", "true", "yes"}:
        logging.info("-" * 100)
        logging.info("FULL EIGENVALUE TABLE")
        logging.info("-" * 100)
        for k, (l_raw, l_shifted) in enumerate(zip(lam, lam_gamma)):
            if l_raw < -neg_tol:
                raw_status = "RAW_SIGNIFICANT_NEGATIVE"
            elif l_raw < 0.0:
                raw_status = "RAW_TINY_NEGATIVE_NUMERICAL_ZERO"
            elif abs(l_raw) <= neg_tol:
                raw_status = "RAW_NEAR_ZERO"
            else:
                raw_status = "RAW_POSITIVE"
            shifted_status = "SHIFTED_NON_POSITIVE" if l_shifted <= 0.0 else "SHIFTED_POSITIVE"
            logging.info(
                "mode %04d | lambda = %.16e | lambda_gamma = %.16e | %s | %s",
                k, float(l_raw), float(l_shifted), raw_status, shifted_status,
            )
    logging.info("=" * 100)

@dataclass
class SpatialGraph:
    adjacency: np.ndarray
    gamma: float = 1e-3
    normalized_laplacian: bool = False
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma must be strictly positive")
        self.device = torch.device(self.device)
        self.A_np = sanitize_adjacency(self.adjacency)
        self.L_np = build_laplacian(self.A_np, normalized=self.normalized_laplacian)
        # lam, U = np.linalg.eigh(self.L_np)
        # idx = np.argsort(lam)
        # self.lam_np = lam[idx].astype(np.float64)
        # self.U_np = U[:, idx].astype(np.float64)
        # self.lam_gamma_np = self.lam_np + float(self.gamma)

        lam, U = np.linalg.eigh(self.L_np)
        idx = np.argsort(lam)

        lam = lam[idx].astype(np.float64)
        U = U[:, idx].astype(np.float64)

        lam_gamma = lam + float(self.gamma)

        log_laplacian_eigenspectrum(
            lam=lam,
            lam_gamma=lam_gamma,
            gamma=float(self.gamma),
            neg_tol=1e-10,
        )

        if np.any(lam_gamma <= 0.0):
            bad_modes = np.where(lam_gamma <= 0.0)[0]
            raise ValueError(
                f"Some gamma-shifted eigenvalues are non-positive. "
                f"Bad modes: {bad_modes.tolist()}, "
                f"min lambda_gamma = {lam_gamma.min():.16e}"
            )

        self.lam_np = lam
        self.U_np = U
        self.lam_gamma_np = lam_gamma

        self.N = int(self.A_np.shape[0])
        self.U = torch.tensor(self.U_np, device=self.device, dtype=self.dtype)
        self.UT = self.U.T.contiguous()
        self.lam = torch.tensor(self.lam_np, device=self.device, dtype=self.dtype)
        self.lam_gamma = torch.tensor(self.lam_gamma_np, device=self.device, dtype=self.dtype)
        self.L = torch.tensor(self.L_np, device=self.device, dtype=self.dtype)
        self.A = torch.tensor(self.A_np, device=self.device, dtype=self.dtype)

    def to_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Node -> graph Fourier transform for [B,N,T]."""
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,T], got {tuple(x.shape)}")
        return torch.einsum("kn,bnt->bkt", self.UT, x)

    def from_spectral(self, x_spec: torch.Tensor) -> torch.Tensor:
        """Graph Fourier -> node transform for [B,N,T]."""
        if x_spec.ndim != 3:
            raise ValueError(f"Expected [B,N,T], got {tuple(x_spec.shape)}")
        return torch.einsum("nk,bkt->bnt", self.U, x_spec)


def load_graph_for_fence(config, target_dim: int, device: str | torch.device, dtype: torch.dtype = torch.float32) -> SpatialGraph:
    gad_cfg = config["gad"] if config.has_section("gad") else {}
    graph_path = str(gad_cfg.get("graph_path", "")).strip() if hasattr(gad_cfg, "get") else ""
    gamma = float(gad_cfg.get("gamma", 1e-3)) if hasattr(gad_cfg, "get") else 1e-3
    normalized = str(gad_cfg.get("normalized_laplacian", "0")).lower() in {"1", "true", "yes"} if hasattr(gad_cfg, "get") else False
    if graph_path:
        suffix = Path(graph_path).suffix.lower()
        if suffix in {".npy", ".npz"}:
            A = adjacency_from_npy(graph_path)
        else:
            A = adjacency_from_csv(
                graph_path,
                num_nodes=target_dim,
                sigma2=float(gad_cfg.get("adj_sigma2", 0.1)),
                threshold=float(gad_cfg.get("adj_threshold", 0.0)),
            )
    else:
        # Safe fallback for debugging only. Real experiments should provide graph_path.
        A = np.eye(target_dim, dtype=np.float64)
    if A.shape[0] != target_dim:
        raise ValueError(f"Graph has {A.shape[0]} nodes but data has target_dim={target_dim}")
    return SpatialGraph(A, gamma=gamma, normalized_laplacian=normalized, device=device, dtype=dtype)
