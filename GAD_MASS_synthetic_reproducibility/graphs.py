from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

def rgg_knn_graph(
    n: int = 128,
    k: int = 8,
    seed: int = 0,
    weight_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n, 2))

    tree = cKDTree(pos)
    dists, idx = tree.query(pos, k=k + 1)
    dists = dists[:, 1:]
    idx = idx[:, 1:]

    if weight_sigma is None:
        weight_sigma = float(np.median(dists))
        weight_sigma = max(weight_sigma, 1e-3)

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j, dij in zip(idx[i], dists[i]):
            w = np.exp(-(float(dij) ** 2) / (weight_sigma ** 2))
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)

    L = adjacency_to_laplacian(W, normalized=False)
    return pos, W, L



def adjacency_to_laplacian(W: np.ndarray, normalized: bool = False) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    deg = W.sum(axis=1)
    if normalized:
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return np.eye(W.shape[0], dtype=np.float64) - D_inv_sqrt @ W @ D_inv_sqrt
    return np.diag(deg) - W



def shifted_laplacian(L: np.ndarray, gamma: float = 1e-3) -> np.ndarray:
    return np.asarray(L, dtype=np.float64) + gamma * np.eye(L.shape[0], dtype=np.float64)



def sym_normalized_adjacency(W: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if add_self_loops:
        W = W + np.eye(W.shape[0], dtype=np.float64)
    deg = W.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]



def eigh_sym(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, V = np.linalg.eigh(np.asarray(A, dtype=np.float64))
    order = np.argsort(lam)
    return lam[order], V[:, order]
