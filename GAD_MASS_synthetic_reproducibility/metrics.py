from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import wasserstein_distance


def sample_cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / max(X.shape[0], 1)



def spectral_covariance(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    Xtil = np.asarray(X, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    Xtil_c = Xtil - Xtil.mean(axis=0, keepdims=True)
    return (Xtil_c.T @ Xtil_c) / max(Xtil.shape[0], 1)



def cov_rel_error(emp_cov: np.ndarray, ref_cov: np.ndarray) -> float:
    emp_cov = np.asarray(emp_cov, dtype=np.float64)
    ref_cov = np.asarray(ref_cov, dtype=np.float64)
    return float(np.linalg.norm(emp_cov - ref_cov, ord="fro") / (np.linalg.norm(ref_cov, ord="fro") + 1e-12))


def _safe_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    std = np.where(np.asarray(std, dtype=np.float64) < eps, 1.0, std)
    return (np.asarray(X, dtype=np.float64) - mean[None, :]) / std[None, :]



def standardize_pair_from_reference(X_gen: np.ndarray, X_ref: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    X_ref = np.asarray(X_ref, dtype=np.float64)
    mean = X_ref.mean(axis=0)
    std = X_ref.std(axis=0)
    return _safe_standardize(X_gen, mean, std, eps=eps), _safe_standardize(X_ref, mean, std, eps=eps)



def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    return np.maximum(AA + BB - 2.0 * (A @ B.T), 0.0)



def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    N = X.shape[0]
    M = Y.shape[0]
    if gamma is None:
        Z = np.vstack([X, Y])[: min(len(X) + len(Y), 1000)]
        D = _pairwise_sq_dists(Z, Z)
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        gamma = 1.0 / (2.0 * med + 1e-12)

    Kxx = np.exp(-gamma * _pairwise_sq_dists(X, X))
    Kyy = np.exp(-gamma * _pairwise_sq_dists(Y, Y))
    Kxy = np.exp(-gamma * _pairwise_sq_dists(X, Y))
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = Kxx.sum() / (N * (N - 1) + 1e-12) + Kyy.sum() / (M * (M - 1) + 1e-12) - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))



def mmd_rbf_standardized(X_gen: np.ndarray, X_ref: np.ndarray, gamma: float | None = None) -> float:
    X_gen_z, X_ref_z = standardize_pair_from_reference(X_gen, X_ref)
    return mmd_rbf(X_gen_z, X_ref_z, gamma=gamma)



def nodewise_wasserstein_summary(X_gen: np.ndarray, X_ref: np.ndarray) -> dict[str, float]:
    X_gen = np.asarray(X_gen, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    per_node = np.asarray([wasserstein_distance(X_gen[:, i], X_ref[:, i]) for i in range(X_gen.shape[1])], dtype=np.float64)
    return {
        "node_w1_mean": float(np.mean(per_node)),   
        "node_w1_median": float(np.median(per_node)),
        "node_w1_q90": float(np.quantile(per_node, 0.90)),
        "node_w1_max": float(np.max(per_node)),
    }



def _subsample_equal_size(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(X), len(Y))
    if len(X) > n:
        idx = rng.choice(len(X), size=n, replace=False)
        X = X[idx]
    if len(Y) > n:
        idx = rng.choice(len(Y), size=n, replace=False)
        Y = Y[idx]
    return X, Y



def sliced_wasserstein_distance(
    X: np.ndarray,
    Y: np.ndarray,
    num_projections: int = 128,
    seed: int = 0,
    standardize: bool = False,
) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if standardize:
        X, Y = standardize_pair_from_reference(X, Y)
    rng = np.random.default_rng(seed)
    X, Y = _subsample_equal_size(X, Y, rng)
    dim = X.shape[1]
    dirs = rng.standard_normal(size=(num_projections, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    proj_x = np.sort(X @ dirs.T, axis=0)
    proj_y = np.sort(Y @ dirs.T, axis=0)
    return float(np.mean(np.abs(proj_x - proj_y)))



def repeated_sliced_wasserstein(
    X: np.ndarray,
    Y: np.ndarray,
    num_projections: int = 128,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    standardize: bool = False,
) -> dict[str, float]:
    vals = [sliced_wasserstein_distance(X, Y, num_projections=num_projections, seed=int(s), standardize=standardize) for s in seeds]
    vals_arr = np.asarray(vals, dtype=np.float64)
    return {
        "global_swd_mean": float(vals_arr.mean()),
        "global_swd_std": float(vals_arr.std(ddof=0)),
        "global_swd_min": float(vals_arr.min()),
        "global_swd_max": float(vals_arr.max()),
    }


def graph_smoothness(X: np.ndarray, L: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    return np.einsum("bi,ij,bj->b", X, L, X)



def smoothness_distribution_metrics(X_gen: np.ndarray, X_ref: np.ndarray, L: np.ndarray) -> dict[str, float]:
    s_gen = graph_smoothness(X_gen, L)
    s_ref = graph_smoothness(X_ref, L)
    return {
        "smoothness_w1": float(wasserstein_distance(s_gen, s_ref)),
        "smoothness_mean_gap_abs": float(abs(s_gen.mean() - s_ref.mean())),
        "smoothness_std_gap_abs": float(abs(s_gen.std() - s_ref.std())),
    }



def _band_slices(num_modes: int, num_bands: int = 3) -> list[slice]:
    edges = np.linspace(0, num_modes, num_bands + 1, dtype=int)
    out: list[slice] = []
    for i in range(num_bands):
        start, stop = int(edges[i]), int(edges[i + 1])
        if stop <= start:
            stop = min(num_modes, start + 1)
        out.append(slice(start, stop))
    return out



def band_energy_ratios(X: np.ndarray, V: np.ndarray, num_bands: int = 3) -> np.ndarray:
    Xtil = np.asarray(X, dtype=np.float64) @ np.asarray(V, dtype=np.float64)
    E = Xtil ** 2
    total = E.sum(axis=1, keepdims=True) + 1e-12
    ratios: list[np.ndarray] = []
    for sl in _band_slices(X.shape[1], num_bands=num_bands):
        ratios.append(E[:, sl].sum(axis=1, keepdims=True) / total)
    return np.concatenate(ratios, axis=1)



def band_ratio_metrics(X_gen: np.ndarray, X_ref: np.ndarray, V: np.ndarray, num_bands: int = 3) -> dict[str, float]:
    R_gen = band_energy_ratios(X_gen, V, num_bands=num_bands)
    R_ref = band_energy_ratios(X_ref, V, num_bands=num_bands)
    mean_gen = R_gen.mean(axis=0)
    mean_ref = R_ref.mean(axis=0)
    per_band_w1 = [wasserstein_distance(R_gen[:, i], R_ref[:, i]) for i in range(R_gen.shape[1])]
    out = {
        "band_ratio_l1": float(np.sum(np.abs(mean_gen - mean_ref))),
        "band_ratio_w1_mean": float(np.mean(per_band_w1)),
    }
    for i in range(R_gen.shape[1]):
        out[f"band{i}_mean_gen"] = float(mean_gen[i])
        out[f"band{i}_mean_ref"] = float(mean_ref[i])
        out[f"band{i}_mean_gap_abs"] = float(abs(mean_gen[i] - mean_ref[i]))
        out[f"band{i}_w1"] = float(per_band_w1[i])
    return out



def graph_stationarity_discrepancy(cov: np.ndarray, L: np.ndarray) -> float:
    cov = np.asarray(cov, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    comm = cov @ L - L @ cov
    return float(np.linalg.norm(comm, ord="fro") / ((np.linalg.norm(cov, ord="fro") * np.linalg.norm(L, ord="fro")) + 1e-12))



def _gini_vector(x: np.ndarray) -> float:
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    denom = float(np.sum(x))
    if denom <= 1e-12:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1) * x)) / (n * denom)) - (n + 1) / n)



def inverse_participation_ratio(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = X ** 2
    num = np.sum(sq ** 2, axis=1)
    den = np.sum(sq, axis=1) ** 2 + 1e-12
    return num / den



def gini_over_nodes(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.asarray([_gini_vector(row) for row in np.abs(X)], dtype=np.float64)



def spatial_localization_metrics(X_gen: np.ndarray, X_ref: np.ndarray) -> dict[str, float]:
    ipr_gen = inverse_participation_ratio(X_gen)
    ipr_ref = inverse_participation_ratio(X_ref)
    gini_gen = gini_over_nodes(X_gen)
    gini_ref = gini_over_nodes(X_ref)
    return {
        "ipr_w1": float(wasserstein_distance(ipr_gen, ipr_ref)),
        "ipr_mean_gap_abs": float(abs(ipr_gen.mean() - ipr_ref.mean())),
        "gini_w1": float(wasserstein_distance(gini_gen, gini_ref)),
        "gini_mean_gap_abs": float(abs(gini_gen.mean() - gini_ref.mean())),
    }



def summarize_generation_metrics(
    X_gen: np.ndarray,
    X_ref: np.ndarray,
    V: np.ndarray,
    L: np.ndarray,
    sw_num_projections: int = 128,
    sw_seeds: Iterable[int] = (0, 1, 2, 3, 4),
) -> dict[str, float]:
    X_gen = np.asarray(X_gen, dtype=np.float64)
    X_ref = np.asarray(X_ref, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)

    cov_gen = sample_cov(X_gen)
    cov_ref = sample_cov(X_ref)
    spec_cov_gen = spectral_covariance(X_gen, V)
    spec_cov_ref = spectral_covariance(X_ref, V)

    out: dict[str, float] = {}
    out.update(repeated_sliced_wasserstein(X_gen, X_ref, num_projections=sw_num_projections, seeds=sw_seeds))
    out["mmd_rbf_z"] = mmd_rbf_standardized(X_gen, X_ref)
    out.update(nodewise_wasserstein_summary(X_gen, X_ref))
    out.update(smoothness_distribution_metrics(X_gen, X_ref, L))
    out.update(band_ratio_metrics(X_gen, X_ref, V, num_bands=3))
    out["cov_rel_err"] = cov_rel_error(cov_gen, cov_ref)
    out["spec_cov_rel_err"] = cov_rel_error(spec_cov_gen, spec_cov_ref)
    out["graph_stationarity_gen"] = graph_stationarity_discrepancy(cov_gen, L)
    out["graph_stationarity_ref"] = graph_stationarity_discrepancy(cov_ref, L)
    out["graph_stationarity_gap_abs"] = abs(out["graph_stationarity_gen"] - out["graph_stationarity_ref"])
    out.update(spatial_localization_metrics(X_gen, X_ref))

    # Backward-compatible aliases used in older scripts.
    out["node_w1"] = out["node_w1_mean"]
    out["global_w1"] = out["global_swd_mean"]
    out["mmd_rbf"] = out["mmd_rbf_z"]
    out["smoothness_gap_abs"] = out["smoothness_mean_gap_abs"]
    return out
