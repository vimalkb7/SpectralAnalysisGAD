from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset


class CleanTrafficDataset(Dataset):
    def __init__(self, data_windows: np.ndarray, condition_masks: np.ndarray, sample_len: int):
        if data_windows.shape != condition_masks.shape:
            raise ValueError(f"Data/mask shapes differ: {data_windows.shape} vs {condition_masks.shape}")
        self.observed_data = data_windows.astype(np.float32, copy=False)
        self.observed_masks = np.ones_like(condition_masks, dtype=np.float32)
        self.gt_masks = condition_masks.astype(np.float32, copy=False)
        self.sample_len = int(sample_len)

    def __len__(self):
        return len(self.observed_data)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.sample_len, dtype=np.float32),
        }


def _windows(x: np.ndarray, length: int) -> np.ndarray:
    if x.ndim != 2 or x.shape[0] < length:
        raise ValueError(f"Cannot window shape={x.shape} at length={length}")
    return np.asarray([x[i:i + length] for i in range(x.shape[0] - length + 1)], dtype=np.float32)


@dataclass(frozen=True)
class Exp34DataMetadata:
    true_path: str
    mask_path: str
    total_timepoints: int
    nodes: int
    train_timepoints: int
    valid_timepoints: int
    test_timepoints: int
    train_windows: int
    valid_windows: int
    test_windows: int
    train_mean: float
    train_std: float
    train_condition_fraction: float
    validation_condition_fraction: float
    test_condition_fraction: float


def build_exp34_clean_loaders(cfg: configparser.ConfigParser):
    prefix = Path(cfg["file"]["data_prefix"])
    missing_type = cfg["train"]["type"]
    miss_rate = cfg["train"]["miss_rate"]
    true_path = prefix / f"true_data_{missing_type}_{miss_rate}.npz"
    miss_path = prefix / f"miss_data_{missing_type}_{miss_rate}.npz"
    if not true_path.exists() or not miss_path.exists():
        raise FileNotFoundError(f"Required dataset files not found: {true_path}, {miss_path}")

    true_npz = np.load(true_path)
    miss_npz = np.load(miss_path)
    clean_raw = np.nan_to_num(true_npz["data"][:, :, 0].astype(np.float32))
    stored_mask = np.nan_to_num(miss_npz["mask"][:, :, 0].astype(np.float32))
    if clean_raw.shape != stored_mask.shape:
        raise ValueError(f"Data/mask mismatch: {clean_raw.shape} vs {stored_mask.shape}")
    if not np.all(np.isin(np.unique(stored_mask), [0.0, 1.0])):
        raise ValueError("Stored SC-TC mask is not binary.")

    total_t, nodes = clean_raw.shape
    val_len = int(total_t * float(cfg["train"]["val_ratio"]))
    test_len = int(total_t * float(cfg["train"]["test_ratio"]))
    train_len = total_t - val_len - test_len
    sample_len = int(cfg["train"]["sample_len"])
    batch_size = int(cfg["train"]["batch_size"])

    mean = float(clean_raw[:train_len].mean())
    std = float(clean_raw[:train_len].std())
    if std <= 0:
        raise ValueError(f"Invalid training normalization std={std}")
    clean = (clean_raw - mean) / std

    train_clean = _windows(clean[:train_len], sample_len)
    val_clean = _windows(clean[train_len:train_len + val_len], sample_len)
    test_clean = _windows(clean[-test_len:], sample_len)
    train_mask = _windows(stored_mask[:train_len], sample_len)
    val_mask = _windows(stored_mask[train_len:train_len + val_len], sample_len)
    test_mask = _windows(stored_mask[-test_len:], sample_len)

    ones_train = np.ones_like(train_mask, dtype=np.float32)
    ones_val = np.ones_like(val_mask, dtype=np.float32)

    phase1_train_ds = CleanTrafficDataset(train_clean, ones_train, sample_len)
    phase1_valid_ds = CleanTrafficDataset(val_clean, ones_val, sample_len)
    phase2_train_ds = CleanTrafficDataset(train_clean, train_mask, sample_len)
    phase2_valid_ds = CleanTrafficDataset(val_clean, val_mask, sample_len)
    valid_eval_ds = CleanTrafficDataset(val_clean, val_mask, sample_len)
    test_eval_ds = CleanTrafficDataset(test_clean, test_mask, sample_len)

    metadata = Exp34DataMetadata(
        true_path=str(true_path), mask_path=str(miss_path), total_timepoints=total_t, nodes=nodes,
        train_timepoints=train_len, valid_timepoints=val_len, test_timepoints=test_len,
        train_windows=len(phase1_train_ds), valid_windows=len(phase1_valid_ds), test_windows=len(test_eval_ds),
        train_mean=mean, train_std=std,
        train_condition_fraction=float(train_mask.mean()),
        validation_condition_fraction=float(val_mask.mean()),
        test_condition_fraction=float(test_mask.mean()),
    )

    return {
        "spectral_train": DataLoader(phase1_train_ds, batch_size=batch_size, shuffle=False),
        "phase1_train": DataLoader(phase1_train_ds, batch_size=batch_size, shuffle=True),
        "phase1_valid": DataLoader(phase1_valid_ds, batch_size=batch_size, shuffle=False),
        "phase2_train": DataLoader(phase2_train_ds, batch_size=batch_size, shuffle=True),
        "phase2_valid": DataLoader(phase2_valid_ds, batch_size=batch_size, shuffle=False),
        "valid_eval": DataLoader(valid_eval_ds, batch_size=batch_size, shuffle=False),
        "test_eval": DataLoader(test_eval_ds, batch_size=batch_size, shuffle=False),
        "target_dim": nodes, "std": std, "mean": mean, "metadata": metadata,
    }
