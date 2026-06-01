from __future__ import annotations

import numpy as np
import torch


# These synthetic families match the definitions used in code.zip.

def stationary_gaussian_in_graph_fourier(
    V: np.ndarray,
    s_k: np.ndarray,
    N: int,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = V.shape[0]
    z = rng.standard_normal(size=(N, n))
    x_tilde = z * np.sqrt(np.asarray(s_k, dtype=np.float64)[None, :])
    return (x_tilde @ V.T).astype(np.float64)


def spectrum_power_law(lam_gamma: np.ndarray, p: float = 1.0, floor: float = 1e-6) -> np.ndarray:
    s = 1.0 / (np.asarray(lam_gamma, dtype=np.float64) ** p)
    s = np.maximum(s, floor)
    return s / np.mean(s)


def spectrum_bandlimited(n: int, k_cut: int = 16, low_power: float = 1.0, high_power: float = 0.0) -> np.ndarray:
    s = np.full((n,), high_power, dtype=np.float64)
    s[:k_cut] = low_power
    denom = np.mean(s[s > 0]) if np.any(s > 0) else 1.0
    return s / denom


def spectrum_highfreq_mixture(lam_gamma: np.ndarray, strength: float = 3.0, floor: float = 1e-3) -> np.ndarray:
    lam = np.asarray(lam_gamma, dtype=np.float64)
    lam_norm = lam / (np.max(lam) + 1e-12)
    s = floor + lam_norm ** strength
    return s / np.mean(s)


def block_covariance_vertex_domain(
    n: int,
    num_blocks: int = 4,
    rho: float = 0.8,
    seed: int = 0,
    jitter: float = 1e-3,
    N: int = 1000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sizes = [n // num_blocks] * num_blocks
    for i in range(n % num_blocks):
        sizes[i] += 1

    L = np.zeros((n, n), dtype=np.float64)
    start = 0
    for b in sizes:
        Cb = (1 - rho) * np.eye(b) + rho * np.ones((b, b))
        Cb = Cb + jitter * np.eye(b)
        Lb = np.linalg.cholesky(Cb)
        L[start : start + b, start : start + b] = Lb
        start += b

    Z = rng.standard_normal(size=(N, n))
    X = Z @ L.T
    return (X / np.std(X)).astype(np.float64)


def localized_bursts(
    n: int,
    N: int,
    num_bursts: int = 2,
    burst_amp: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(N, n)) * 0.3
    for i in range(N):
        nodes = rng.choice(n, size=num_bursts, replace=False)
        X[i, nodes] += burst_amp * rng.standard_normal(size=num_bursts)
    return (X / np.std(X)).astype(np.float64)


def make_synthetic_dataset_numpy(
    family: str,
    V: np.ndarray,
    lam_gamma: np.ndarray,
    N: int,
    seed: int = 0,
    spectrum: str = "power_law",
    power_law_p: float = 1.0,
    k_cut: int = 16,
    strength: float = 3.0,
    num_blocks: int = 4,
    rho: float = 0.8,
    jitter: float = 1e-3,
    num_bursts: int = 2,
    burst_amp: float = 3.0,
) -> tuple[np.ndarray, dict]:
    meta: dict = {"family": family}

    if family == "stationary_gaussian":
        if spectrum == "power_law":
            s_k = spectrum_power_law(lam_gamma, p=power_law_p)
        elif spectrum == "bandlimited":
            s_k = spectrum_bandlimited(len(lam_gamma), k_cut=k_cut)
        elif spectrum == "highfreq":
            s_k = spectrum_highfreq_mixture(lam_gamma, strength=strength)
        else:
            raise ValueError(f"Unknown spectrum: {spectrum}")
        X = stationary_gaussian_in_graph_fourier(V, s_k, N=N, seed=seed)
        meta["s_k_true"] = s_k
        return X, meta

    if family == "smooth_bandlimited":
        s_k = spectrum_bandlimited(len(lam_gamma), k_cut=k_cut, low_power=1.0, high_power=0.02)
        X = stationary_gaussian_in_graph_fourier(V, s_k, N=N, seed=seed)
        meta["s_k_true"] = s_k
        return X, meta

    if family == "highfreq_mixture":
        s_k = spectrum_highfreq_mixture(lam_gamma, strength=strength)
        X = stationary_gaussian_in_graph_fourier(V, s_k, N=N, seed=seed)
        meta["s_k_true"] = s_k
        return X, meta

    if family == "block_covariance":
        X = block_covariance_vertex_domain(n=V.shape[0], num_blocks=num_blocks, rho=rho, seed=seed, jitter=jitter, N=N)
        return X, meta

    if family == "localized_bursts":
        X = localized_bursts(n=V.shape[0], N=N, num_bursts=num_bursts, burst_amp=burst_amp, seed=seed)
        return X, meta

    raise ValueError(f"Unknown family: {family}")


def to_torch(X: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(X, device=device, dtype=dtype)
