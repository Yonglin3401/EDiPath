"""Spatial transcriptomics data loading helpers for EDiPath."""

from __future__ import annotations

import os

import numpy as np


class SpatialDataset:
    def __init__(self, x, coords, y=None, name="spatial", expr=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.coords = np.asarray(coords, dtype=np.float32)
        self.y = None if y is None else np.asarray(y)
        # Non-negative expression for Shannon-entropy digraph (optional).
        self.expr = None if expr is None else np.asarray(expr, dtype=np.float32)
        self.name = name
        self.num_node = self.x.shape[0]
        self.num_features = self.x.shape[1]
        if self.y is not None:
            # map labels to 0..C-1 if needed
            uniq = np.unique(self.y[~np.isnan(self.y)]) if self.y.dtype.kind == "f" else np.unique(self.y)
            self.num_classes = len(uniq)
        else:
            self.num_classes = None

    @property
    def num_node_classes(self):
        return self.num_classes


def load_spatial_h5ad(
    path,
    use_rep="X_pca",
    n_pcs=50,
    label_key=None,
    layer=None,
    section_filter=None,
    section_key="section",
    name=None,
):
    """Load AnnData (.h5ad) with spatial coordinates.

    Preference order for features:
      1) adata.obsm[use_rep] if present
      2) PCA of adata.X / layer with ``n_pcs``

    section_filter: int or list of ints; keep only obs[section_key] in that set.
    """
    import scanpy as sc
    from sklearn.decomposition import PCA

    adata = sc.read_h5ad(path)
    if section_filter is not None:
        if section_key not in adata.obs:
            raise ValueError(f"{path} missing obs['{section_key}'] for section_filter")
        want = {section_filter} if isinstance(section_filter, int) else set(section_filter)
        mask = adata.obs[section_key].isin(want)
        n_before = adata.n_obs
        adata = adata[mask].copy()
        print(f"  section_filter={sorted(want)}: {n_before} -> {adata.n_obs} cells")

    if "spatial" not in adata.obsm:
        # MOB_3D also has 'coord'
        if "coord" in adata.obsm:
            adata.obsm["spatial"] = np.asarray(adata.obsm["coord"])
        else:
            raise ValueError(f"{path} missing adata.obsm['spatial']")

    coords = np.asarray(adata.obsm["spatial"])

    # Raw / layer expression for Shannon entropy (clip negatives later in digraph).
    if layer is not None:
        expr_mat = adata.layers[layer]
    else:
        expr_mat = adata.X
    if hasattr(expr_mat, "toarray"):
        expr_mat = expr_mat.toarray()
    expr = np.asarray(expr_mat, dtype=np.float32)

    if use_rep in adata.obsm:
        x = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    else:
        mat = expr
        n_pcs = min(n_pcs, mat.shape[0] - 1, mat.shape[1])
        x = PCA(n_components=n_pcs, random_state=42).fit_transform(mat).astype(np.float32)

    y = None
    if label_key and label_key in adata.obs:
        y_raw = adata.obs[label_key].astype("category")
        y = y_raw.cat.codes.to_numpy()

    ds_name = name or os.path.splitext(os.path.basename(path))[0]
    return SpatialDataset(x=x, coords=coords, y=y, name=ds_name, expr=expr), adata


def make_synthetic_spatial(n=200, n_domains=4, feat_dim=32, seed=42):
    """Simple grid-like synthetic SRT for smoke tests."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)[:n].astype(np.float32)

    # 2x2 spatial domains
    mid = side / 2.0
    y = ((coords[:, 0] >= mid).astype(int) + 2 * (coords[:, 1] >= mid).astype(int))
    y = y % n_domains

    centers = rng.normal(size=(n_domains, feat_dim))
    x = centers[y] + 0.3 * rng.normal(size=(n, feat_dim))
    x = x.astype(np.float32)
    # Non-negative synthetic expression: domain-specific sparse-ish profiles.
    expr_centers = np.abs(rng.normal(size=(n_domains, feat_dim))) + 0.1
    expr = expr_centers[y] + 0.05 * np.abs(rng.normal(size=(n, feat_dim)))
    expr = expr.astype(np.float32)
    return SpatialDataset(x=x, coords=coords, y=y, name="synthetic", expr=expr)
