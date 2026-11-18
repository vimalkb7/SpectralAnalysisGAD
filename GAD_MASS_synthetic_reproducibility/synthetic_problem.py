from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datasets import make_synthetic_dataset_numpy
from graphs import eigh_sym, rgg_knn_graph


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


@dataclass
class FrozenSyntheticProblem:
    pos: np.ndarray
    W: np.ndarray
    L: np.ndarray
    lam: np.ndarray
    V: np.ndarray
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    train_mean: np.ndarray
    metadata: dict

    @property
    def n(self) -> int:
        return int(self.L.shape[0])

    @property
    def s_k_train(self) -> np.ndarray:
        return np.mean((self.X_train @ self.V) ** 2, axis=0)

    def shifted_eigenvalues(self, gamma: float) -> np.ndarray:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        return self.lam + float(gamma)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            pos=self.pos,
            W=self.W,
            L=self.L,
            lam=self.lam,
            V=self.V,
            X_train=self.X_train,
            X_val=self.X_val,
            X_test=self.X_test,
            train_mean=self.train_mean,
            metadata_json=np.asarray(json.dumps(self.metadata, default=_jsonable)),
        )
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=_jsonable)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenSyntheticProblem":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["metadata_json"].item()))
        return cls(
            pos=d["pos"],
            W=d["W"],
            L=d["L"],
            lam=d["lam"],
            V=d["V"],
            X_train=d["X_train"],
            X_val=d["X_val"],
            X_test=d["X_test"],
            train_mean=d["train_mean"],
            metadata=meta,
        )


def create_frozen_problem(
    family: str = "broadband_wss",
    n: int = 128,
    k: int = 8,
    graph_seed: int = 1001,
    data_seed: int = 2001,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
    center_from_train: bool = True,
    broadband_exponent: float = 1.5,
    broadband_delta: float = 0.03,
    broadband_high_band_target: float = 0.20,
    nonwss_localization_strength: float = 0.8,
    nonwss_heat_tau: float = 0.25,
    nonwss_active_node_fraction: float = 0.25,
) -> FrozenSyntheticProblem:
    pos, W, L = rgg_knn_graph(n=n, k=k, seed=graph_seed)
    lam, V = eigh_sym(L)
    total = int(num_train + num_val + num_test)

    X, data_meta = make_synthetic_dataset_numpy(
        family=family,
        V=V,
        lam_gamma=lam + 1e-3,
        lam_base=lam,
        N=total,
        seed=data_seed,
        broadband_exponent=broadband_exponent,
        broadband_delta=broadband_delta,
        broadband_high_band_target=broadband_high_band_target,
        nonwss_localization_strength=nonwss_localization_strength,
        nonwss_heat_tau=nonwss_heat_tau,
        nonwss_active_node_fraction=nonwss_active_node_fraction,
    )
    X_train = X[:num_train].copy()
    X_val = X[num_train:num_train + num_val].copy()
    X_test = X[num_train + num_val:num_train + num_val + num_test].copy()
    train_mean = X_train.mean(axis=0) if center_from_train else np.zeros(n, dtype=np.float64)
    X_train -= train_mean[None, :]
    X_val -= train_mean[None, :]
    X_test -= train_mean[None, :]

    eigengap = np.diff(lam)
    min_gap = float(np.min(eigengap)) if len(eigengap) else float("inf")
    problem_hash = _sha256_arrays(L, X_train, X_val, X_test)
    meta = {
        "schema_version": 2,
        "family": family,
        "n": int(n),
        "k": int(k),
        "graph_seed": int(graph_seed),
        "data_seed": int(data_seed),
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "centered_using_training_mean": bool(center_from_train),
        "minimum_laplacian_eigengap": min_gap,
        "problem_sha256": problem_hash,
        "data_meta": data_meta,
        "invariant_statement": (
            "This file fixes graph, eigenbasis, train/val/test arrays, and centering. "
            "Changing gamma or sigma changes only the forward/reverse diffusion channel."
        ),
    }
    return FrozenSyntheticProblem(pos, W, L, lam, V, X_train, X_val, X_test, train_mean, meta)


def main() -> None:
    p = argparse.ArgumentParser(description="Create a frozen synthetic graph/data problem reused by all schedulers and gamma/sigma ablations.")
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="broadband_wss", choices=["broadband_wss", "graph_localized_nonwss", "block_covariance"])
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--graph_seed", type=int, default=1001)
    p.add_argument("--data_seed", type=int, default=2001)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--broadband_exponent", type=float, default=1.5)
    p.add_argument("--broadband_delta", type=float, default=0.03)
    p.add_argument("--broadband_high_band_target", type=float, default=0.20)
    p.add_argument("--nonwss_localization_strength", type=float, default=0.8)
    p.add_argument("--nonwss_heat_tau", type=float, default=0.25)
    p.add_argument("--nonwss_active_node_fraction", type=float, default=0.25)
    args = p.parse_args()
    problem = create_frozen_problem(**{k: v for k, v in vars(args).items() if k != "out"})
    problem.save(args.out)
    print(json.dumps(problem.metadata, indent=2, default=_jsonable))
    print(f"Saved frozen problem: {args.out}")


if __name__ == "__main__":
    main()
