from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

from graphs import adjacency_to_laplacian, shifted_laplacian


class GraphSignalSnapshotDataset(Dataset):
    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected [num_samples, num_nodes], got {x.shape}.")
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.x[index]


@dataclass
class LoadedRealDataset:
    name: str
    train: GraphSignalSnapshotDataset
    val: GraphSignalSnapshotDataset
    test: GraphSignalSnapshotDataset
    adjacency: np.ndarray
    laplacian: np.ndarray
    node_names: list[str]
    train_mean: np.ndarray
    train_std: np.ndarray



def contiguous_time_split(n_samples: int, train_frac: float = 0.7, val_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    idx = np.arange(n_samples)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]



def _zscore_from_train(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray, eps: float = 1e-6):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std, (x_test - mean) / std, mean.squeeze(0), std.squeeze(0)



def dense_adj_from_edge_index(edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int, symmetrize: bool = True) -> np.ndarray:
    a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src, dst = edge_index
    a[src, dst] = edge_weight.astype(np.float32)
    if symmetrize:
        a = 0.5 * (a + a.T)
    np.fill_diagonal(a, 0.0)
    return a



def knn_gaussian_adjacency(coords: np.ndarray, k: int = 6, self_loops: bool = False) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    dist = cdist(coords, coords, metric="euclidean")
    nonzero = dist[dist > 0]
    sigma = np.median(nonzero) if nonzero.size else 1.0
    weights = np.exp(-(dist ** 2) / (sigma ** 2 + 1e-12)).astype(np.float32)
    n = dist.shape[0]
    a = np.zeros_like(weights, dtype=np.float32)
    for i in range(n):
        nn_idx = np.argsort(dist[i])[1 : k + 1]
        a[i, nn_idx] = weights[i, nn_idx]
    a = np.maximum(a, a.T)
    if not self_loops:
        np.fill_diagonal(a, 0.0)
    return a



def _finalize_loaded_dataset(
    name: str,
    x: np.ndarray,
    adjacency: np.ndarray,
    node_names: list[str],
    train_frac: float,
    val_frac: float,
    standardize: bool,
    gamma: float,
) -> LoadedRealDataset:
    train_idx, val_idx, test_idx = contiguous_time_split(len(x), train_frac=train_frac, val_frac=val_frac)
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    if standardize:
        x_train, x_val, x_test, mean, std = _zscore_from_train(x_train, x_val, x_test)
    else:
        mean = np.zeros(x.shape[1], dtype=np.float32)
        std = np.ones(x.shape[1], dtype=np.float32)
    laplacian = shifted_laplacian(adjacency_to_laplacian(adjacency, normalized=False), gamma=gamma).astype(np.float32)
    return LoadedRealDataset(
        name=name,
        train=GraphSignalSnapshotDataset(x_train),
        val=GraphSignalSnapshotDataset(x_val),
        test=GraphSignalSnapshotDataset(x_test),
        adjacency=np.asarray(adjacency, dtype=np.float32),
        laplacian=np.asarray(laplacian, dtype=np.float32),
        node_names=node_names,
        train_mean=np.asarray(mean, dtype=np.float32),
        train_std=np.asarray(std, dtype=np.float32),
    )



def load_tsl_snapshot_dataset(
    name: str,
    root: str = "./data",
    stride: int = 1,
    threshold: float = 0.1,
    include_self: bool = False,
    normalize_axis: Optional[int] = 1,
    standardize: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    gamma: float = 1e-3,
    min_observed_frac: float = 0.95,
    small_airquality: bool = False,
) -> LoadedRealDataset:
    try:
        from tsl.datasets import AirQuality, MetrLA, PemsBay
    except ImportError as exc:
        raise ImportError("This loader requires `tsl` / torch-spatiotemporal. Install it in the runtime that will use these scripts.") from exc

    key = name.lower().strip()
    if key == "metr_la":
        dataset = MetrLA(root=root)
    elif key == "pems_bay":
        dataset = PemsBay(root=root)
    elif key == "air_quality":
        dataset = AirQuality(root=root, small=small_airquality)
    else:
        raise ValueError("name must be one of {'metr_la','pems_bay','air_quality'}.")

    df = dataset.dataframe()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected dataset.dataframe() to return a pandas DataFrame.")

    x_df = df.copy()
    if hasattr(dataset, "mask") and dataset.mask is not None:
        mask = np.asarray(dataset.mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape[:2] == x_df.shape:
            observed_frac = mask.mean(axis=1)
            keep = observed_frac >= min_observed_frac
            x_df = x_df.iloc[keep]
    x_df = x_df.ffill().bfill()
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy(dtype=np.float32)[::stride]

    connectivity = dataset.get_connectivity(
        threshold=threshold,
        include_self=include_self,
        normalize_axis=normalize_axis,
        layout="edge_index",
    )
    edge_index_np, edge_weight_np = connectivity
    edge_index_np = np.asarray(edge_index_np)
    edge_weight_np = np.asarray(edge_weight_np, dtype=np.float32)
    adjacency = dense_adj_from_edge_index(edge_index_np, edge_weight_np, n_nodes=x.shape[1], symmetrize=True)
    node_names = [str(c) for c in x_df.columns.tolist()]
    return _finalize_loaded_dataset(
        name=("air_quality_small" if key == "air_quality" and small_airquality else key),
        x=x,
        adjacency=adjacency,
        node_names=node_names,
        train_frac=train_frac,
        val_frac=val_frac,
        standardize=standardize,
        gamma=gamma,
    )



def load_intel_lab_snapshots(
    readings_path: str,
    coords_path: str,
    value_col: str = "temperature",
    stride: int = 1,
    standardize: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    gamma: float = 1e-3,
    knn_k: int = 6,
) -> LoadedRealDataset:
    readings_path = str(readings_path)
    coords_path = str(coords_path)

    cols = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]
    df = pd.read_csv(readings_path, sep=r"\s+", header=None, names=cols, compression="infer")
    if value_col not in df.columns:
        raise ValueError(f"value_col must be one of {cols[4:]}, got {value_col!r}.")
    df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df = df.dropna(subset=["timestamp"])

    piv = df.pivot_table(index="timestamp", columns="moteid", values=value_col, aggfunc="mean").sort_index()
    piv = piv.ffill().bfill().dropna(axis=1, how="all").dropna(axis=0, how="all")
    x = piv.to_numpy(dtype=np.float32)[::stride]

    coords = pd.read_csv(coords_path, sep=r"\s+", header=None)
    if coords.shape[1] < 3:
        raise ValueError("coords_path must contain at least three columns: moteid x y")
    coords = coords.iloc[:, :3].copy()
    coords.columns = ["moteid", "x", "y"]
    coords = coords[coords["moteid"].isin(piv.columns)]
    coords = coords.set_index("moteid").loc[piv.columns]
    adjacency = knn_gaussian_adjacency(coords[["x", "y"]].to_numpy(dtype=np.float32), k=knn_k, self_loops=False)

    return _finalize_loaded_dataset(
        name=f"intel_{value_col}",
        x=x,
        adjacency=adjacency,
        node_names=[str(c) for c in piv.columns.tolist()],
        train_frac=train_frac,
        val_frac=val_frac,
        standardize=standardize,
        gamma=gamma,
    )



def load_real_dataset(
    dataset_name: str,
    root: str = "./data",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    gamma: float = 1e-3,
    stride: int = 1,
    intel_readings_path: Optional[str] = None,
    intel_coords_path: Optional[str] = None,
    intel_value_col: str = "temperature",
) -> LoadedRealDataset:
    key = dataset_name.lower().strip()
    if key == "metr_la":
        return load_tsl_snapshot_dataset("metr_la", root=root, stride=stride, train_frac=train_frac, val_frac=val_frac, gamma=gamma)
    if key == "pems_bay":
        return load_tsl_snapshot_dataset("pems_bay", root=root, stride=stride, train_frac=train_frac, val_frac=val_frac, gamma=gamma)
    if key == "air_quality_small":
        return load_tsl_snapshot_dataset("air_quality", root=root, stride=stride, train_frac=train_frac, val_frac=val_frac, gamma=gamma, small_airquality=True)
    if key == "air_quality_full":
        return load_tsl_snapshot_dataset("air_quality", root=root, stride=stride, train_frac=train_frac, val_frac=val_frac, gamma=gamma, small_airquality=False)
    if key == "intel_lab":
        if intel_readings_path is None or intel_coords_path is None:
            raise ValueError("Intel Lab loading requires --intel_readings_path and --intel_coords_path.")
        return load_intel_lab_snapshots(
            readings_path=intel_readings_path,
            coords_path=intel_coords_path,
            value_col=intel_value_col,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            gamma=gamma,
        )
    raise ValueError(
        "dataset_name must be one of {'metr_la','pems_bay','air_quality_small','air_quality_full','intel_lab'}."
    )
