"""Entropy-directed graph construction for spatial transcriptomics.

Pipeline
--------
1. Spatial candidate edges: physical kNN on spot coordinates.
2. Per-spot Shannon entropy of the gene-expression profile
   (low entropy = informative / high certainty; high entropy = mixed / noisy).
3. Directed edges among spatial candidates: low-entropy -> high-entropy,
   so high-certainty spots propagate features toward uncertain spots
   (information-diffusion direction; noise does not reverse-contaminate).
4. Optional edge weights from the entropy gap ``H(v) - H(u)``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def build_spatial_knn_adj(coords: np.ndarray, k: int = 6, include_self: bool = False) -> sp.csr_matrix:
    """Undirected spatial kNN adjacency (symmetrized)."""
    n = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    rows, cols = [], []
    start = 0 if include_self else 1
    for i in range(n):
        for j in indices[i, start:]:
            if i == j:
                continue
            rows.append(i)
            cols.append(int(j))
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj = adj.maximum(adj.T)
    adj.data = np.ones_like(adj.data)
    return adj.tocsr()


def spot_shannon_entropy(expr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-spot Shannon entropy of the gene-expression distribution.

    For spot i with non-negative expression ``x_ig``:
        p_ig = x_ig / sum_g x_ig
        H_i  = -sum_g p_ig log p_ig

    Negative entries (e.g. PCA fallback) are clipped to 0. Spots with near-zero
    total expression are assigned the maximum entropy ``log(G)`` (fully uncertain).
    """
    x = np.asarray(expr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expr must be 2D [N, G], got shape {x.shape}")
    x = np.maximum(x, 0.0)
    row_sum = x.sum(axis=1, keepdims=True)
    n_genes = x.shape[1]
    max_h = float(np.log(max(n_genes, 1)))

    p = np.divide(x, np.maximum(row_sum, eps))
    # 0 * log(0) := 0
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0.0, np.log(p), 0.0)
    H = -np.sum(p * logp, axis=1)
    H = np.maximum(H, 0.0)
    H[row_sum.ravel() <= eps] = max_h
    return H.astype(np.float64)


def build_entropy_digraph(
    coords: np.ndarray,
    features: np.ndarray,
    spatial_k: int = 6,
    top_k: int = 6,
    use_entropy_weight: bool = True,
    weight_eps: float = 1e-3,
    expr: np.ndarray | None = None,
):
    """Build a spatial digraph on the same kNN candidates.

    Edges are oriented strictly from low entropy to high entropy and only the
    top-``top_k`` outgoing neighbours by entropy gap are kept. Edge weight is
    ``H(v) - H(u) + weight_eps`` when ``use_entropy_weight`` is true, else 1.

    Entropy is computed on ``expr`` when provided (preferred: non-negative gene
    expression); otherwise it falls back to ``features``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    assert coords.ndim == 2 and coords.shape[1] >= 2
    assert features.ndim == 2 and features.shape[0] == coords.shape[0]

    entropy_src = features if expr is None else np.asarray(expr, dtype=np.float64)
    if entropy_src.shape[0] != coords.shape[0]:
        raise ValueError(
            f"expr/features rows ({entropy_src.shape[0]}) != coords rows ({coords.shape[0]})"
        )

    spatial_adj = build_spatial_knn_adj(coords, k=spatial_k, include_self=False)
    H = spot_shannon_entropy(entropy_src)
    n = features.shape[0]

    rows, cols, data = [], [], []
    for u in range(n):
        nbrs = spatial_adj[u].indices
        if len(nbrs) == 0:
            continue
        gaps = H[nbrs] - H[u]
        # Only propagate low-entropy -> high-entropy.
        higher = np.where(gaps > 0.0)[0]
        if len(higher) == 0:
            continue
        order = higher[np.argsort(-gaps[higher])]
        keep = order[: min(int(top_k), len(order))]
        for j in keep:
            v = int(nbrs[j])
            gap = float(gaps[j])
            w = float(gap + weight_eps) if use_entropy_weight else 1.0
            rows.append(u)
            cols.append(v)
            data.append(w)

    digraph = sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(n, n),
    ).tocsr()

    outdeg = np.asarray((digraph > 0).sum(axis=1)).ravel()
    isolated = np.where(outdeg == 0)[0]

    meta = {
        "entropy": H,
        "entropy_mean": float(H.mean()),
        "entropy_std": float(H.std()),
        "entropy_min": float(H.min()),
        "entropy_max": float(H.max()),
        "n_spatial_edges": int(spatial_adj.nnz // 2),
        "n_directed_edges": int(digraph.nnz),
        "n_fallback_nodes": int(len(isolated)),
        "spatial_k": spatial_k,
        "top_k": int(top_k),
        "mean_out_degree": float(outdeg.mean()),
        "use_entropy_weight": bool(use_entropy_weight),
        "entropy_from_expr": bool(expr is not None),
    }
    return digraph, spatial_adj, meta
