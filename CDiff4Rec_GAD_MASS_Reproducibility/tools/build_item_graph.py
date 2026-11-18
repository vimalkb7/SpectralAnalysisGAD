from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, laplacian
from sklearn.metrics.pairwise import cosine_similarity

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import data_utils


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build item graph assets for GAD-MAS-CDiff4Rec")
    p.add_argument("--dataset", type=str, default="amazon_game")
    p.add_argument("--data_path", type=str, default="./datasets")
    p.add_argument("--output", type=str, default=None,
                   help="Output .npz path; default graph_assets/<dataset>_hybrid_k20_gad_mas.npz")
    p.add_argument("--graph_mode", choices=["cf", "content", "hybrid"], default="hybrid")
    p.add_argument("--cf_weight", type=float, default=0.5,
                   help="Hybrid graph weight on collaborative similarity; content weight is 1-cf_weight")
    p.add_argument("--topk_edges", type=int, default=20,
                   help="Number of strongest candidate neighbours retained for each item before symmetrization")
    p.add_argument("--topk_pseudo_user", type=int, default=1000,
                   help="Number of review-derived pseudo-user feature rows used for the content graph")
    p.add_argument("--laplacian", choices=["normalized", "combinatorial"], default="normalized")
    p.add_argument("--lap_gamma", type=float, default=1e-3,
                   help="Shift added to Laplacian eigenvalues so L_gamma is positive definite")
    p.add_argument("--s_min", type=float, default=1e-8,
                   help="Floor for estimated clean spectral powers")
    p.add_argument("--bridge_eps", type=float, default=1e-8,
                   help="Fallback bridge weight only when no positive cross-component similarity exists")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def _load_data(dataset_dir: Path):
    train_data, valid_y, test_y, n_user, n_item = data_utils.data_load(
        str(dataset_dir / "train_list.npy"),
        str(dataset_dir / "valid_list.npy"),
        str(dataset_dir / "test_list.npy"),
    )
    with open(dataset_dir / "pseudo_data.pickle", "rb") as fh:
        pseudo_data = pickle.load(fh)
    return train_data.tocsr(), pseudo_data, n_user, n_item


def _topk_sparsify(sim: np.ndarray, k: int) -> sp.csr_matrix:
    sim = np.asarray(sim, dtype=np.float64).copy()
    n = sim.shape[0]
    np.fill_diagonal(sim, 0.0)
    k = max(1, min(k, n - 1))
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    vals = sim[rows, cols]
    valid = vals > 0.0
    W = sp.csr_matrix((vals[valid], (rows[valid], cols[valid])), shape=(n, n))
    W = W.maximum(W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def _connect_graph(W: sp.csr_matrix, sim_for_bridges: np.ndarray, eps: float) -> Tuple[sp.csr_matrix, List[Dict[str, float]], int]:
    n_comp, labels = connected_components(W, directed=False, return_labels=True)
    before = int(n_comp)
    bridges: List[Dict[str, float]] = []
    W_lil = W.tolil(copy=True)
    sim = np.asarray(sim_for_bridges, dtype=np.float64).copy()
    np.fill_diagonal(sim, -np.inf)

    while n_comp > 1:
        cross_mask = labels[:, None] != labels[None, :]
        scores = np.where(cross_mask, sim, -np.inf)
        flat_idx = int(np.argmax(scores))
        i, j = np.unravel_index(flat_idx, scores.shape)
        score = float(scores[i, j])
        used_fallback = False
        if not np.isfinite(score) or score <= 0.0:
            # This is only used when retained items have no observed similarity at all.
            comp_a = np.flatnonzero(labels == labels[0])[0]
            comp_b = np.flatnonzero(labels != labels[0])[0]
            i, j = int(comp_a), int(comp_b)
            score = float(eps)
            used_fallback = True
        W_lil[i, j] = max(float(W_lil[i, j]), score)
        W_lil[j, i] = max(float(W_lil[j, i]), score)
        bridges.append({"item_i": int(i), "item_j": int(j), "weight": score,
                        "fallback_eps": bool(used_fallback)})
        W = W_lil.tocsr()
        n_comp, labels = connected_components(W, directed=False, return_labels=True)

    return W_lil.tocsr(), bridges, before


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    dataset_dir = Path(args.data_path) / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}. Unzip datasets.zip first.")

    train_data, pseudo_data, n_user, n_item = _load_data(dataset_dir)
    if n_item < 2:
        raise ValueError("At least two items are required to build an item graph.")

    cf_sim = cosine_similarity(train_data.T, dense_output=True).astype(np.float64)
    np.fill_diagonal(cf_sim, 0.0)

    f_train = pseudo_data["train_data"].tocsr()[: args.topk_pseudo_user]
    content_sim = cosine_similarity(f_train.T, dense_output=True).astype(np.float64)
    np.fill_diagonal(content_sim, 0.0)

    if args.graph_mode == "cf":
        base_sim = cf_sim
    elif args.graph_mode == "content":
        base_sim = content_sim
    else:
        if not (0.0 <= args.cf_weight <= 1.0):
            raise ValueError("--cf_weight must be in [0, 1].")
        base_sim = args.cf_weight * cf_sim + (1.0 - args.cf_weight) * content_sim

    W = _topk_sparsify(base_sim, args.topk_edges)
    W, bridges, components_before = _connect_graph(W, base_sim, args.bridge_eps)
    components_after, _ = connected_components(W, directed=False, return_labels=True)

    W_dense = W.toarray().astype(np.float64)
    use_normed = args.laplacian == "normalized"
    L = laplacian(W_dense, normed=use_normed).astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals[np.abs(eigvals) < 1e-10] = 0.0
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals_shifted = eigvals + float(args.lap_gamma)

    X_real = train_data.toarray().astype(np.float64)
    real_spec = X_real @ eigvecs
    s_real = np.maximum(np.mean(real_spec ** 2, axis=0), args.s_min)

    X_pseudo = f_train.toarray().astype(np.float64)
    pseudo_spec = X_pseudo @ eigvecs
    s_pseudo = np.maximum(np.mean(pseudo_spec ** 2, axis=0), args.s_min)

    degrees = np.asarray(W.sum(axis=1)).reshape(-1)
    nonzero_eigs = eigvals[eigvals > 1e-10]
    diagnostics = {
        "dataset": args.dataset,
        "n_user": int(n_user),
        "n_item": int(n_item),
        "graph_mode": args.graph_mode,
        "cf_weight": float(args.cf_weight),
        "topk_edges": int(args.topk_edges),
        "topk_pseudo_user": int(args.topk_pseudo_user),
        "laplacian": args.laplacian,
        "lap_gamma": float(args.lap_gamma),
        "components_before_bridging": int(components_before),
        "components_after_bridging": int(components_after),
        "num_bridge_edges_added": int(len(bridges)),
        "bridges": bridges,
        "num_edges_undirected": int(W.nnz // 2),
        "degree_min": float(degrees.min()),
        "degree_mean": float(degrees.mean()),
        "degree_max": float(degrees.max()),
        "lambda_0": float(eigvals[0]),
        "lambda_1": float(nonzero_eigs[0]) if len(nonzero_eigs) else None,
        "lambda_max": float(eigvals[-1]),
        "s_real_min": float(s_real.min()),
        "s_real_max": float(s_real.max()),
        "s_pseudo_min": float(s_pseudo.min()),
        "s_pseudo_max": float(s_pseudo.max()),
    }

    if args.output is None:
        out_path = ROOT / "graph_assets" / f"{args.dataset}_{args.graph_mode}_k{args.topk_edges}_gad_mas.npz"
    else:
        out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        adjacency=W_dense.astype(np.float32),
        laplacian=L.astype(np.float32),
        eigvals=eigvals.astype(np.float32),
        eigvals_shifted=eigvals_shifted.astype(np.float32),
        eigvecs=eigvecs.astype(np.float32),
        s_real=s_real.astype(np.float32),
        s_pseudo=s_pseudo.astype(np.float32),
        diagnostics_json=np.array(json.dumps(diagnostics)),
    )
    diag_path = out_path.with_suffix(".json")
    diag_path.write_text(json.dumps(diagnostics, indent=2))

    print("Item graph and GAD/MAS-SNR assets created")
    print(json.dumps(diagnostics, indent=2))
    print(f"Saved spectral assets: {out_path}")
    print(f"Saved diagnostics:     {diag_path}")


if __name__ == "__main__":
    main()
