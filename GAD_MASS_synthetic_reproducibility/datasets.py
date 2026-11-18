from __future__ import annotations

import numpy as np
import torch


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


def _normalize_mean_power(s: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    s = np.maximum(np.asarray(s, dtype=np.float64), floor)
    return s / max(float(np.mean(s)), floor)


def spectrum_power_law(lam: np.ndarray, p: float = 1.0, floor: float = 1e-6) -> np.ndarray:
    """Legacy spectrum. Prefer spectrum_broadband_wss for the detailed study."""
    lam = np.asarray(lam, dtype=np.float64)
    positive = lam[lam > 1e-12]
    shift = float(np.min(positive)) if positive.size else 1.0
    s = 1.0 / (np.maximum(lam, shift * 1e-3) ** p)
    return _normalize_mean_power(np.maximum(s, floor))


def spectrum_broadband_wss(
    lam_base: np.ndarray,
    exponent: float = 1.5,
    delta: float = 0.03,
    target_high_band_fraction: float = 0.20,
    high_band_fraction: float = 0.25,
    floor: float = 1e-8,
) -> np.ndarray:
    lam = np.asarray(lam_base, dtype=np.float64)
    if exponent <= 0 or delta <= 0:
        raise ValueError("exponent and delta must be positive")
    lam_norm = lam / max(float(np.max(lam)), 1e-12)
    base = (delta + lam_norm) ** (-float(exponent))
    n = len(base)
    m = max(1, int(round(float(high_band_fraction) * n)))
    target = float(target_high_band_fraction)
    if not (0 < target < m / n):
        raise ValueError(
            f"target_high_band_fraction must lie in (0,{m/n:.3f}) for an additive floor; got {target}."
        )
    sum_all = float(np.sum(base))
    sum_high = float(np.sum(base[-m:]))
    denom = float(m - target * n)
    c = max(0.0, (target * sum_all - sum_high) / max(denom, 1e-12))
    s = base + c
    return _normalize_mean_power(np.maximum(s, floor))


def spectrum_bandlimited(n: int, k_cut: int = 16, low_power: float = 1.0, high_power: float = 0.0) -> np.ndarray:
    s = np.full((n,), high_power, dtype=np.float64)
    s[:k_cut] = low_power
    return _normalize_mean_power(s)


def spectrum_highfreq_mixture(lam: np.ndarray, strength: float = 3.0, floor: float = 1e-3) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    lam_norm = lam / (np.max(lam) + 1e-12)
    s = floor + lam_norm ** strength
    return _normalize_mean_power(s)


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

    chol = np.zeros((n, n), dtype=np.float64)
    start = 0
    for b in sizes:
        Cb = (1 - rho) * np.eye(b) + rho * np.ones((b, b)) + jitter * np.eye(b)
        Lb = np.linalg.cholesky(Cb)
        chol[start : start + b, start : start + b] = Lb
        start += b

    Z = rng.standard_normal(size=(N, n))
    X = Z @ chol.T
    return (X / max(float(np.std(X)), 1e-12)).astype(np.float64)


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
    return (X / max(float(np.std(X)), 1e-12)).astype(np.float64)


def graph_localized_nonwss(
    V: np.ndarray,
    lam_base: np.ndarray,
    N: int,
    seed: int = 0,
    base_exponent: float = 1.5,
    base_delta: float = 0.03,
    base_high_band_target: float = 0.20,
    localization_strength: float = 0.8,
    heat_tau: float = 0.25,
    active_node_fraction: float = 0.25,
) -> tuple[np.ndarray, dict]:
    V = np.asarray(V, dtype=np.float64)
    lam = np.asarray(lam_base, dtype=np.float64)
    n = V.shape[0]
    rng = np.random.default_rng(seed)
    p = spectrum_broadband_wss(
        lam,
        exponent=base_exponent,
        delta=base_delta,
        target_high_band_fraction=base_high_band_target,
    )
    X_wss = stationary_gaussian_in_graph_fourier(V, p, N=N, seed=seed + 11)

    heat_gain = np.exp(-float(heat_tau) * lam / max(float(np.max(lam)), 1e-12))
    H = V @ np.diag(heat_gain) @ V.T
    active_count = max(1, int(round(float(active_node_fraction) * n)))
    # Fixed active set for all samples => a genuinely non-stationary covariance.
    active_nodes = np.sort(rng.choice(n, size=active_count, replace=False))
    Hs = H[:, active_nodes]
    Z = rng.standard_normal(size=(N, active_count))
    X_local = Z @ Hs.T
    X_local /= max(float(np.std(X_local)), 1e-12)
    X = X_wss + float(localization_strength) * X_local
    X /= max(float(np.std(X)), 1e-12)

    cov_tilde = np.cov((X @ V), rowvar=False, bias=True)
    offdiag = cov_tilde - np.diag(np.diag(cov_tilde))
    stationarity_gap = float(np.linalg.norm(offdiag, ord="fro") / (np.linalg.norm(cov_tilde, ord="fro") + 1e-12))
    meta = {
        "s_k_base": p,
        "active_nodes": active_nodes,
        "heat_tau": float(heat_tau),
        "localization_strength": float(localization_strength),
        "active_node_fraction": float(active_node_fraction),
        "spectral_cov_offdiag_ratio_empirical": stationarity_gap,
    }
    return X.astype(np.float64), meta


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
    lam_base: np.ndarray | None = None,
    broadband_exponent: float = 1.5,
    broadband_delta: float = 0.03,
    broadband_high_band_target: float = 0.20,
    nonwss_localization_strength: float = 0.8,
    nonwss_heat_tau: float = 0.25,
    nonwss_active_node_fraction: float = 0.25,
) -> tuple[np.ndarray, dict]:
    meta: dict = {"family": family}
    lam_data = np.asarray(lam_base if lam_base is not None else lam_gamma, dtype=np.float64)

    if family == "broadband_wss":
        s_k = spectrum_broadband_wss(
            lam_data,
            exponent=broadband_exponent,
            delta=broadband_delta,
            target_high_band_fraction=broadband_high_band_target,
        )
        X = stationary_gaussian_in_graph_fourier(V, s_k, N=N, seed=seed)
        meta.update(
            {
                "s_k_true": s_k,
                "spectrum_definition": "(delta+lambda/lambda_max)^(-nu)+calibrated_floor",
                "spectrum_uses_unshifted_laplacian": lam_base is not None,
                "broadband_exponent": float(broadband_exponent),
                "broadband_delta": float(broadband_delta),
                "target_high_band_fraction": float(broadband_high_band_target),
            }
        )
        return X, meta

    if family == "graph_localized_nonwss":
        X, extra = graph_localized_nonwss(
            V=V,
            lam_base=lam_data,
            N=N,
            seed=seed,
            base_exponent=broadband_exponent,
            base_delta=broadband_delta,
            base_high_band_target=broadband_high_band_target,
            localization_strength=nonwss_localization_strength,
            heat_tau=nonwss_heat_tau,
            active_node_fraction=nonwss_active_node_fraction,
        )
        meta.update(extra)
        meta["spectrum_uses_unshifted_laplacian"] = lam_base is not None
        return X, meta

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
        meta["legacy_gamma_dependent_spectrum"] = True
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
