from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pickle

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


def contiguous_time_split(
    n_samples: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    idx = np.arange(n_samples)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _zscore_from_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    eps: float = 1e-6,
):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (
        (x_train - mean) / std,
        (x_val - mean) / std,
        (x_test - mean) / std,
        mean.squeeze(0),
        std.squeeze(0),
    )


def knn_gaussian_adjacency(
    coords: np.ndarray,
    k: int = 6,
    self_loops: bool = False,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    dist = cdist(coords, coords, metric="euclidean")
    nonzero = dist[dist > 0]
    sigma = np.median(nonzero) if nonzero.size else 1.0
    weights = np.exp(-(dist ** 2) / (sigma ** 2 + 1e-12)).astype(np.float32)
    n = dist.shape[0]
    a = np.zeros_like(weights, dtype=np.float32)
    for i in range(n):
        nn_idx = np.argsort(dist[i])[1:k + 1]
        a[i, nn_idx] = weights[i, nn_idx]
    a = np.maximum(a, a.T)
    if not self_loops:
        np.fill_diagonal(a, 0.0)
    return a


def correlation_threshold_adjacency(
    x: np.ndarray,
    threshold: float = 0.3,
    use_abs: bool = True,
    self_loops: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.abs(corr) if use_abs else np.maximum(corr, 0.0)
    weights = np.where(weights >= float(threshold), weights, 0.0)
    weights = 0.5 * (weights + weights.T)
    if not self_loops:
        np.fill_diagonal(weights, 0.0)
    return weights.astype(np.float32)


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
    train_idx, val_idx, test_idx = contiguous_time_split(
        len(x), train_frac=train_frac, val_frac=val_frac
    )
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    if standardize:
        x_train, x_val, x_test, mean, std = _zscore_from_train(x_train, x_val, x_test)
    else:
        mean = np.zeros(x.shape[1], dtype=np.float32)
        std = np.ones(x.shape[1], dtype=np.float32)

    adjacency = np.asarray(adjacency, dtype=np.float32)
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    laplacian = shifted_laplacian(
        adjacency_to_laplacian(adjacency, normalized=False),
        gamma=gamma,
    ).astype(np.float32)

    return LoadedRealDataset(
        name=name,
        train=GraphSignalSnapshotDataset(x_train),
        val=GraphSignalSnapshotDataset(x_val),
        test=GraphSignalSnapshotDataset(x_test),
        adjacency=adjacency,
        laplacian=laplacian,
        node_names=node_names,
        train_mean=np.asarray(mean, dtype=np.float32),
        train_std=np.asarray(std, dtype=np.float32),
    )


def _find_first_existing_path(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_traffic_paths(
    dataset_key: str,
    root: str,
    data_path: Optional[str] = None,
    adj_path: Optional[str] = None,
) -> tuple[Path, Path]:
    root_path = Path(root)

    if data_path is not None:
        data_file = Path(data_path)
    else:
        if dataset_key == "metr_la":
            data_file = _find_first_existing_path([
                root_path / "METR-LA.h5",
                root_path / "METR_LA" / "METR-LA.h5",
                root_path / "metr_la" / "METR-LA.h5",
            ])
        elif dataset_key == "pems_bay":
            data_file = _find_first_existing_path([
                root_path / "pems-bay.h5",
                root_path / "PEMS_BAY" / "pems-bay.h5",
                root_path / "pems_bay" / "pems-bay.h5",
            ])
        else:
            data_file = None

    if adj_path is not None:
        adj_file = Path(adj_path)
    else:
        if dataset_key == "metr_la":
            adj_file = _find_first_existing_path([
                root_path / "adj_METR-LA.pkl",
                root_path / "METR_LA" / "adj_METR-LA.pkl",
                root_path / "metr_la" / "adj_METR-LA.pkl",
            ])
        elif dataset_key == "pems_bay":
            adj_file = _find_first_existing_path([
                root_path / "adj_mx_bay.pkl",
                root_path / "PEMS_BAY" / "adj_mx_bay.pkl",
                root_path / "pems_bay" / "adj_mx_bay.pkl",
            ])
        else:
            adj_file = None

    if data_file is None or not data_file.exists():
        raise FileNotFoundError(
            f"Could not find the H5 signal file for {dataset_key!r}. "
            f"Looked under root={root!r}. You can also pass data_path explicitly."
        )
    if adj_file is None or not adj_file.exists():
        raise FileNotFoundError(
            f"Could not find the adjacency PKL file for {dataset_key!r}. "
            f"Looked under root={root!r}. You can also pass adj_path explicitly."
        )
    return data_file, adj_file


def _load_dcrnn_style_adj_pkl(adj_path: str | Path) -> tuple[list[str], np.ndarray]:
    with open(adj_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if not isinstance(obj, (list, tuple)) or len(obj) != 3:
        raise ValueError(
            f"Expected adjacency pickle to contain [sensor_ids, sensor_id_to_ind, adj_mx], got type={type(obj)}."
        )

    sensor_ids, _, adj_mx = obj
    sensor_ids = [str(x) for x in sensor_ids]
    adj_mx = np.asarray(adj_mx, dtype=np.float32)
    if adj_mx.ndim != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj_mx.shape}.")
    return sensor_ids, adj_mx


def load_metr_or_pems_from_files(
    name: str,
    root: str = "./data",
    data_path: Optional[str] = None,
    adj_path: Optional[str] = None,
    stride: int = 1,
    standardize: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    gamma: float = 1e-3,
    min_observed_frac: float = 0.95,
) -> LoadedRealDataset:
    key = name.lower().strip()
    if key not in {"metr_la", "pems_bay"}:
        raise ValueError("name must be one of {'metr_la', 'pems_bay'}." )

    data_file, adj_file = _resolve_traffic_paths(
        dataset_key=key,
        root=root,
        data_path=data_path,
        adj_path=adj_path,
    )

    df = pd.read_hdf(data_file)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame inside {data_file}, got {type(df)}.")

    sensor_ids, adjacency = _load_dcrnn_style_adj_pkl(adj_file)

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    missing_in_h5 = [sid for sid in sensor_ids if sid not in df.columns]
    if missing_in_h5:
        raise ValueError(
            f"These nodes exist in adjacency but not in the H5 dataframe columns: {missing_in_h5[:10]}"
        )

    df = df[sensor_ids]

    observed = (~df.isna()).mean(axis=1)
    keep = observed >= float(min_observed_frac)
    df = df.loc[keep]
    df = df.ffill().bfill()
    df = df.fillna(df.median(numeric_only=True))

    x = df.to_numpy(dtype=np.float32)[::stride]
    adjacency = np.asarray(adjacency, dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)

    return _finalize_loaded_dataset(
        name=key,
        x=x,
        adjacency=adjacency,
        node_names=sensor_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        standardize=standardize,
        gamma=gamma,
    )


def _resolve_air_quality_csv_path(root: str, csv_path: Optional[str] = None) -> Path:
    if csv_path is not None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"AirQuality csv_path not found: {csv_path}")
        return p

    root_path = Path(root)
    p = _find_first_existing_path([
        root_path / "AirQuality.csv",
        root_path / "air_quality.csv",
        root_path / "archive" / "AirQuality.csv",
    ])
    if p is None:
        raise FileNotFoundError(
            f"Could not find AirQuality.csv under root={root!r}. You can also pass csv_path explicitly."
        )
    return p


def load_air_quality_csv_dataset(
    root: str = "./data",
    csv_path: Optional[str] = None,
    stride: int = 1,
    standardize: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    gamma: float = 1e-3,
    graph_mode: str = "correlation",
    corr_threshold: float = 0.3,
    knn_k: int = 4,
    small: bool = False,
) -> LoadedRealDataset:
    csv_file = _resolve_air_quality_csv_path(root=root, csv_path=csv_path)

    # This is the UCI-style CSV with ';' separators and decimal commas.
    df = pd.read_csv(csv_file, sep=';', decimal=',')
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]

    if 'Date' in df.columns and 'Time' in df.columns:
        ts = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str),
            format='%d/%m/%Y %H.%M.%S',
            errors='coerce',
        )
        df = df.drop(columns=['Date', 'Time'])
        df.index = ts
        df = df.loc[~df.index.isna()]

    # UCI AirQuality uses -200 for missing values.
    df = df.replace(-200, np.nan)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError(f"No numeric sensor columns found in {csv_file}.")

    if small:
        preferred_small_cols = [
            'CO(GT)',
            'PT08.S1(CO)',
            'NMHC(GT)',
            'C6H6(GT)',
            'PT08.S2(NMHC)',
            'NOx(GT)',
            'PT08.S3(NOx)',
            'NO2(GT)',
        ]
        selected_cols = [c for c in preferred_small_cols if c in numeric_cols]
        if len(selected_cols) < 4:
            selected_cols = numeric_cols[: min(8, len(numeric_cols))]
    else:
        selected_cols = numeric_cols

    x_df = df[selected_cols].copy()
    x_df = x_df.ffill().bfill()
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy(dtype=np.float32)[::stride]

    if graph_mode == 'correlation':
        adjacency = correlation_threshold_adjacency(
            x,
            threshold=corr_threshold,
            use_abs=True,
            self_loops=False,
        )
    elif graph_mode == 'feature_knn':
        features = x_df.to_numpy(dtype=np.float32).T  # [num_features, num_times]
        adjacency = knn_gaussian_adjacency(features, k=knn_k, self_loops=False)
    else:
        raise ValueError("graph_mode must be one of {'correlation', 'feature_knn'}." )

    return _finalize_loaded_dataset(
        name='air_quality_small' if small else 'air_quality_full',
        x=x,
        adjacency=adjacency,
        node_names=[str(c) for c in x_df.columns.tolist()],
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
    adjacency = knn_gaussian_adjacency(
        coords[["x", "y"]].to_numpy(dtype=np.float32),
        k=knn_k,
        self_loops=False,
    )

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
    data_path: Optional[str] = None,
    adj_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    airquality_graph_mode: str = 'correlation',
    airquality_corr_threshold: float = 0.3,
    airquality_knn_k: int = 4,
) -> LoadedRealDataset:
    key = dataset_name.lower().strip()
    if key == 'metr_la':
        return load_metr_or_pems_from_files(
            name='metr_la',
            root=root,
            data_path=data_path,
            adj_path=adj_path,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            gamma=gamma,
        )
    if key == 'pems_bay':
        return load_metr_or_pems_from_files(
            name='pems_bay',
            root=root,
            data_path=data_path,
            adj_path=adj_path,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            gamma=gamma,
        )
    if key == 'air_quality_small':
        return load_air_quality_csv_dataset(
            root=root,
            csv_path=csv_path,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            gamma=gamma,
            graph_mode=airquality_graph_mode,
            corr_threshold=airquality_corr_threshold,
            knn_k=airquality_knn_k,
            small=True,
        )
    if key == 'air_quality_full':
        return load_air_quality_csv_dataset(
            root=root,
            csv_path=csv_path,
            stride=stride,
            train_frac=train_frac,
            val_frac=val_frac,
            gamma=gamma,
            graph_mode=airquality_graph_mode,
            corr_threshold=airquality_corr_threshold,
            knn_k=airquality_knn_k,
            small=False,
        )
    if key == 'intel_lab':
        if intel_readings_path is None or intel_coords_path is None:
            raise ValueError('Intel Lab loading requires --intel_readings_path and --intel_coords_path.')
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
