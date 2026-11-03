# -*- coding: utf-8 -*-
"""Four-way clustering + ARI/BF1 + spatial plots for EDiPath.

Methods: mclust / kmeans / leiden / louvain.
Self-contained — does not import external benchmark scripts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

RANDOM_SEED = 2023
DEFAULT_LEIDEN_RES = 1.0
DEFAULT_LOUVAIN_RES = 1.1
DEFAULT_BF1_SPATIAL_K = 6
METHOD_DISPLAY = "EDiPath"


def boundary_mask(y: np.ndarray, neigh: np.ndarray) -> np.ndarray:
    """Spot is a boundary if any spatial neighbor has a different label."""
    return (y[neigh] != y[:, None]).any(axis=1)


def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, neigh: np.ndarray):
    """Spot-level boundary F1 without 1-hop tolerance."""
    gt_b = boundary_mask(y_true, neigh)
    pr_b = boundary_mask(y_pred, neigh)
    tp = np.logical_and(pr_b, gt_b).sum()
    prec = tp / pr_b.sum() if pr_b.sum() else 0.0
    rec = tp / gt_b.sum() if gt_b.sum() else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return float(prec), float(rec), float(f1), int(gt_b.sum()), int(pr_b.sum())


def build_spatial_neigh(coords: np.ndarray, k: int = DEFAULT_BF1_SPATIAL_K) -> np.ndarray:
    """(n, k) spatial neighbor index matrix (excludes self), for boundary_f1."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 2:
        return np.zeros((n, 0), dtype=np.int64)
    kk = int(min(max(k, 1), n - 1))
    nn = NearestNeighbors(n_neighbors=kk + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    return indices[:, 1:].astype(np.int64)


def ari_bf1_summary(metrics: dict) -> dict:
    """ARI/BF1 fields (+ best_* via) for sweep summary.json rows."""
    keys = (
        "ARI_mclust", "ARI_kmeans", "ARI_leiden", "ARI_louvain", "best_ARI",
        "BF1_mclust", "BF1_kmeans", "BF1_leiden", "BF1_louvain", "best_BF1",
        "BF1_prec_mclust", "BF1_prec_kmeans", "BF1_prec_leiden", "BF1_prec_louvain",
        "BF1_rec_mclust", "BF1_rec_kmeans", "BF1_rec_leiden", "BF1_rec_louvain",
        "bf1_spatial_k", "n_gt_boundary", "best_BF1_cluster",
    )
    fields = {k: metrics[k] for k in keys if k in metrics}
    ari_items = [
        (k.replace("ARI_", ""), v) for k, v in metrics.items()
        if k.startswith("ARI_") and v is not None
    ]
    fields["best_cluster"] = (
        max(ari_items, key=lambda t: t[1])[0] if ari_items else None
    )
    if "best_BF1_cluster" not in fields:
        bf1_items = [
            (k.replace("BF1_", ""), v) for k, v in metrics.items()
            if k.startswith("BF1_") and not k.startswith("BF1_prec_")
            and not k.startswith("BF1_rec_") and v is not None
        ]
        fields["best_BF1_cluster"] = (
            max(bf1_items, key=lambda t: t[1])[0] if bf1_items else None
        )
    return fields

# Built-in plot defaults for common datasets (override via kwargs / CLI).
DATASET_PLOT_DEFAULTS = {
    "CNS3": dict(
        spatial_plot_transform="ccw90_flip_y",
        leiden_res=0.42,
        louvain_res=0.46,
        spot_fill=1.0,
        spatial_color_overrides={"1.0": "#aec7e8", "2.0": "#1f77b4"},
        n_clusters=16,
        label_key="Main_molecular_tissue_region",
    ),
    "DPI_2dpi": dict(
        leiden_res=1.10,
        louvain_res=1.35,
        spot_fill=1.35,
        n_clusters=18,
        label_key="Annotation",
    ),
    "2DPI_1": dict(
        leiden_res=1.10,
        louvain_res=1.35,
        spot_fill=1.35,
        n_clusters=18,
        label_key="Annotation",
    ),
    "MOB_slice3": dict(
        leiden_res=0.19,
        louvain_res=0.23,
        spot_fill=1.2,
        n_clusters=6,
        label_key=None,
    ),
    "Breast_cancer": dict(
        spot_fill=0.8,
        n_clusters=20,
        label_key="fine_annot_type",
    ),
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def configure_spatial_axes(ax):
    ax.set_aspect("equal")
    ax.axis("off")


def _apply_spatial_op(x, y, op):
    op = op.strip().lower()
    if op in ("", "none"):
        return x, y
    if op == "cw90":
        return y, -x
    if op == "ccw90":
        return -y, x
    if op == "rot180":
        return -x, -y
    if op in ("flipx", "flip_x"):
        return -x, y
    if op in ("flipy", "flip_y"):
        return x, -y
    raise ValueError(f"Unknown spatial transform op: '{op}'")


def transform_spatial_plot_coords(pos, transform=None):
    x = np.asarray(pos[:, 0], dtype=float)
    y = np.asarray(pos[:, 1], dtype=float)
    if not transform:
        return x, y
    if transform == "ccw90_flip_y":
        transform = "ccw90,flipy"
    for op in transform.split(","):
        x, y = _apply_spatial_op(x, y, op)
    return x, y


def _relabel_exact_k(labels, emb, target_n, seed=RANDOM_SEED):
    """Force exactly target_n clusters: merge closest centroids, or split largest."""
    from sklearn.cluster import KMeans

    labels = np.asarray(labels).astype(object).copy()
    emb = np.asarray(emb, dtype=np.float64)
    uniq = list(dict.fromkeys(labels.tolist()))
    split_id = 0

    while len(uniq) > target_n:
        cent = {u: emb[labels == u].mean(axis=0) for u in uniq}
        best = None
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d = float(np.linalg.norm(cent[uniq[i]] - cent[uniq[j]]))
                if best is None or d < best[0]:
                    best = (d, uniq[i], uniq[j])
        _, keep, drop = best
        labels[labels == drop] = keep
        uniq.remove(drop)

    while len(uniq) < target_n:
        sizes = {u: int((labels == u).sum()) for u in uniq}
        big = max(sizes, key=sizes.get)
        if sizes[big] < 2:
            break
        sub = emb[labels == big]
        km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(sub)
        new = f"{big}_s{split_id}"
        split_id += 1
        idx = np.flatnonzero(labels == big)
        labels[idx[km == 1]] = new
        uniq.append(new)

    return labels


def tune_graph_cluster_resolution(adata, algo, target_n, init_res=1.0,
                                  lo=0.05, hi=5.0, max_iter=25,
                                  use_rep=None, enforce_exact=True):
    """Binary-search Leiden/Louvain resolution to hit target_n clusters."""
    import scanpy as sc

    key = algo
    best = (init_res, 10**9, None)

    def _run(res):
        if algo == "leiden":
            sc.tl.leiden(adata, key_added=key, resolution=float(res),
                        random_state=RANDOM_SEED)
        else:
            sc.tl.louvain(adata, key_added=key, resolution=float(res),
                         random_state=RANDOM_SEED)
        return int(adata.obs[key].nunique())

    def _finish(res, n):
        if enforce_exact and use_rep is not None and n != target_n:
            _run(res)
            labels = _relabel_exact_k(adata.obs[key].astype(str).to_numpy(),
                                      adata.obsm[use_rep], target_n)
            adata.obs[key] = pd.Categorical(labels.astype(str))
            n_new = int(adata.obs[key].nunique())
            print(f"  [exact-k] {algo}: {n} -> {n_new} (target {target_n})")
            n = n_new
        return float(res), int(n)

    n0 = _run(init_res)
    best = (init_res, abs(n0 - target_n), n0)
    if n0 == target_n:
        return _finish(init_res, n0)

    left, right = float(lo), float(hi)
    if n0 < target_n:
        left = max(left, float(init_res))
    else:
        right = min(right, float(init_res))

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        n = _run(mid)
        diff = abs(n - target_n)
        if diff < best[1] or (diff == best[1] and abs(mid - init_res) < abs(best[0] - init_res)):
            best = (mid, diff, n)
        if n == target_n:
            return _finish(mid, n)
        if n < target_n:
            left = mid
        else:
            right = mid
        if right - left < 1e-4:
            break

    n_final = _run(best[0])
    return _finish(best[0], int(n_final))


def mclust_subprocess(adata, n_clusters, use_rep="EDiPath", random_seed=RANDOM_SEED):
    """Run R/mclust in a subprocess; return label array or None on failure."""
    here = os.path.dirname(os.path.abspath(__file__))
    helper = os.path.join(os.path.dirname(here), "scripts", "mclust_helper.py")
    tmp = tempfile.mkdtemp(prefix="mclust_")
    emb_npy = os.path.join(tmp, "emb.npy")
    out_npy = os.path.join(tmp, "labels.npy")
    np.save(emb_npy, np.ascontiguousarray(adata.obsm[use_rep]))
    try:
        r = subprocess.run(
            [sys.executable, helper, emb_npy, str(n_clusters), out_npy, str(random_seed)],
            capture_output=True, text=True, timeout=1800,
        )
        if r.returncode == 0 and os.path.exists(out_npy):
            return np.load(out_npy, allow_pickle=True).astype(str)
        print(f"  [warn] mclust subprocess failed (rc={r.returncode}): "
              f"{(r.stderr or '')[-200:]}")
    except Exception as e:
        print(f"  [warn] mclust call error: {e}")
    return None


def _estimate_spot_size(x, y, figsize, fill=1.35):
    pts = np.column_stack([x, y]).astype(float)
    n = len(pts)
    if n < 2:
        return 30.0
    nbrs = NearestNeighbors(n_neighbors=min(6, n)).fit(pts)
    dist, _ = nbrs.kneighbors(pts)
    nn = float(np.median(dist[:, 1]))
    span = max(float(x.max() - x.min()), float(y.max() - y.min()))
    if span <= 0 or nn <= 0:
        return 30.0
    fig_pts = min(figsize) * 72 * 0.9
    ppu = fig_pts / span
    diameter = nn * ppu * fill
    s_geom = diameter ** 2
    s_vis = (1.41 * fill) ** 2
    return float(np.clip(max(s_geom, s_vis), 0.5, 600.0))


def plot_spatial(adata, color_key, title, save_path, spot_size=None,
                 plot_transform=None, show_legend=True, fill=1.35,
                 color_overrides=None):
    pos = adata.obsm["spatial"]
    x, y = transform_spatial_plot_coords(pos, plot_transform)
    cats = pd.Categorical(adata.obs[color_key].astype(str))

    categories = list(cats.categories)
    try:
        order = sorted(categories, key=lambda c: float(c))
    except ValueError:
        order = categories
    cmap = plt.get_cmap("tab20" if len(order) > 10 else "tab10")
    color_of = {c: cmap(i % cmap.N) for i, c in enumerate(order)}
    if color_overrides:
        for k, v in color_overrides.items():
            for c in list(color_of):
                try:
                    if float(c) == float(k):
                        color_of[c] = v
                        break
                except ValueError:
                    if str(c) == str(k):
                        color_of[c] = v
                        break

    figsize = (6.4, 5.2)
    fig, ax = plt.subplots(figsize=figsize)
    s = spot_size if spot_size else _estimate_spot_size(x, y, figsize, fill=fill)
    for c in order:
        m = np.asarray(cats == c)
        ax.scatter(x[m], y[m], color=color_of[c], s=s, linewidths=0)
    ax.set_title(title, fontsize=13)
    configure_spatial_axes(ax)
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=color_of[c], markeredgewidth=0, label=str(c))
            for c in order
        ]
        ax.legend(
            handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
            frameon=False, fontsize=8, title=str(color_key), title_fontsize=9,
            handletextpad=0.3, labelspacing=0.3,
            ncol=1 if len(order) <= 20 else 2,
        )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def cluster_eval_plot(
    adata,
    name,
    n_clusters,
    label_key,
    out_dir,
    fig_dir=None,
    emb_key="EDiPath",
    title_prefix=METHOD_DISPLAY,
    leiden_res=None,
    louvain_res=None,
    force_n_clusters=False,
    no_plot=False,
    save_h5ad=True,
    plot_transform=None,
    spot_fill=1.35,
    color_overrides=None,
    gt_plot_keys=None,
    extra_meta=None,
    bf1_spatial_k=DEFAULT_BF1_SPATIAL_K,
):
    """Run mclust/kmeans/leiden/louvain on adata.obsm[emb_key], save metrics + plots."""
    import scanpy as sc

    if emb_key not in adata.obsm:
        raise KeyError(f"adata.obsm missing embedding key '{emb_key}'")

    defaults = DATASET_PLOT_DEFAULTS.get(name, {})
    if leiden_res is None:
        leiden_res = defaults.get("leiden_res", DEFAULT_LEIDEN_RES)
    if louvain_res is None:
        louvain_res = defaults.get("louvain_res", DEFAULT_LOUVAIN_RES)
    if plot_transform is None:
        plot_transform = defaults.get("spatial_plot_transform")
    if spot_fill is None:
        spot_fill = defaults.get("spot_fill", 1.35)
    if color_overrides is None:
        color_overrides = defaults.get("spatial_color_overrides")

    out_dir = ensure_dir(out_dir)
    fig_dir = ensure_dir(fig_dir or os.path.join(out_dir, "figures"))
    results = {}

    # (A) mclust
    used_mclust = False
    mlabels = mclust_subprocess(adata, n_clusters, use_rep=emb_key, random_seed=RANDOM_SEED)
    if mlabels is not None and len(mlabels) == adata.n_obs:
        adata.obs["mclust"] = mlabels
        used_mclust = True
        print(f"  mclust OK  n={pd.Series(mlabels).nunique()}")
    else:
        print("  [warn] mclust unavailable; use KMeans as primary domain.")

    # (B) KMeans
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    adata.obs["kmeans"] = km.fit_predict(adata.obsm[emb_key]).astype(str)
    print(f"  kmeans n={adata.obs['kmeans'].nunique()}")

    # (C/D) Leiden / Louvain
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=15)
    if force_n_clusters:
        leiden_res, n_leid = tune_graph_cluster_resolution(
            adata, "leiden", n_clusters, init_res=leiden_res, use_rep=emb_key)
        louvain_res, n_louv = tune_graph_cluster_resolution(
            adata, "louvain", n_clusters, init_res=louvain_res, use_rep=emb_key)
        print(f"  [force k={n_clusters}] leiden(res={leiden_res:.4f}) n={n_leid}  "
              f"louvain(res={louvain_res:.4f}) n={n_louv}")
    else:
        sc.tl.leiden(adata, key_added="leiden", resolution=leiden_res,
                     random_state=RANDOM_SEED)
        sc.tl.louvain(adata, key_added="louvain", resolution=louvain_res,
                      random_state=RANDOM_SEED)
        print(f"  leiden(res={leiden_res}) n={adata.obs['leiden'].nunique()}  "
              f"louvain(res={louvain_res}) n={adata.obs['louvain'].nunique()}")

    adata.obs["domain"] = adata.obs["mclust"] if used_mclust else adata.obs["kmeans"]
    # Primary domain key (kept for main_spatial compatibility)
    adata.obs["EDiPath_domain"] = adata.obs["domain"]

    has_labels = label_key is not None and label_key in adata.obs
    if has_labels:
        mask = ~pd.isnull(adata.obs[label_key])
        sub = adata.obs[mask]
        gt = sub[label_key].astype(str)
        methods = (["mclust"] if used_mclust else []) + ["kmeans", "leiden", "louvain"]
        for m in methods:
            results[f"ARI_{m}"] = float(adjusted_rand_score(gt, sub[m].astype(str)))
        results["best_ARI"] = max(v for k, v in results.items() if k.startswith("ARI_"))

        # Boundary F1 on the same labeled subset (spatial kNN, default k=6).
        if "spatial" not in adata.obsm:
            print("  [warn] adata.obsm missing 'spatial'; skip BF1.")
        else:
            idx = np.flatnonzero(np.asarray(mask))
            coords = np.asarray(adata.obsm["spatial"])[idx]
            neigh = build_spatial_neigh(coords, k=int(bf1_spatial_k))
            y_true = np.asarray(gt, dtype=object)
            results["bf1_spatial_k"] = int(bf1_spatial_k)
            for m in methods:
                y_pred = np.asarray(sub[m].astype(str), dtype=object)
                prec, rec, f1, n_gt_b, _n_pr_b = boundary_f1(y_true, y_pred, neigh)
                results[f"BF1_{m}"] = f1
                results[f"BF1_prec_{m}"] = prec
                results[f"BF1_rec_{m}"] = rec
                results["n_gt_boundary"] = n_gt_b
            bf1_items = [
                (k.replace("BF1_", ""), v) for k, v in results.items()
                if k.startswith("BF1_") and not k.startswith("BF1_prec_")
                and not k.startswith("BF1_rec_")
            ]
            results["best_BF1"] = max(v for _, v in bf1_items)
            results["best_BF1_cluster"] = max(bf1_items, key=lambda t: t[1])[0]

        print("  ARI:", {k: round(v, 4) for k, v in results.items() if k.startswith("ARI_") or k == "best_ARI"})
        if any(k.startswith("BF1_") and not k.startswith("BF1_prec_") and not k.startswith("BF1_rec_")
               for k in results):
            print("  BF1:", {
                k: round(v, 4) for k, v in results.items()
                if (k.startswith("BF1_") and not k.startswith("BF1_prec_")
                    and not k.startswith("BF1_rec_")) or k == "best_BF1"
            })
    else:
        print("  No GT labels; skip ARI/BF1.")

    results.update(dict(
        dataset=name,
        n_clusters=n_clusters,
        label_key=label_key,
        n_obs=int(adata.n_obs),
        used_mclust=used_mclust,
        emb_key=emb_key,
        title_prefix=title_prefix,
        leiden_res=leiden_res,
        louvain_res=louvain_res,
        force_n_clusters=force_n_clusters,
        n_leiden=int(adata.obs["leiden"].nunique()),
        n_louvain=int(adata.obs["louvain"].nunique()),
        n_kmeans=int(adata.obs["kmeans"].nunique()),
        n_mclust=int(adata.obs["mclust"].nunique()) if used_mclust else None,
    ))
    if extra_meta:
        results.update(extra_meta)
    save_json(results, os.path.join(out_dir, "metrics.json"))

    if not no_plot:
        methods = (["mclust"] if used_mclust else []) + ["kmeans", "leiden", "louvain"]
        for m in methods:
            bits = []
            if f"ARI_{m}" in results:
                bits.append(f"ARI={results['ARI_' + m]:.4f}")
            if f"BF1_{m}" in results:
                bits.append(f"BF1={results['BF1_' + m]:.4f}")
            metric_txt = f" ({', '.join(bits)})" if bits else ""
            plot_spatial(
                adata, m, f"{title_prefix} {m}{metric_txt}",
                os.path.join(fig_dir, f"{name}_{title_prefix}_{m}.png"),
                plot_transform=plot_transform, fill=spot_fill,
                color_overrides=color_overrides,
            )
        plot_spatial(
            adata, "domain", f"{title_prefix} domain (k={n_clusters})",
            os.path.join(fig_dir, f"{name}_{title_prefix}_domain.png"),
            plot_transform=plot_transform, fill=spot_fill,
            color_overrides=color_overrides,
        )
        keys = list(gt_plot_keys or [])
        if has_labels and label_key not in keys:
            keys = [label_key] + keys
        keys = [k for k in keys if k and k in adata.obs]
        if has_labels and label_key in keys:
            plot_spatial(
                adata, label_key, "Ground Truth",
                os.path.join(fig_dir, f"{name}_ground_truth.png"),
                plot_transform=plot_transform, fill=spot_fill,
            )
        for gk in keys:
            plot_spatial(
                adata, gk, f"Ground Truth ({gk})",
                os.path.join(fig_dir, f"{name}_ground_truth_{gk}.png"),
                plot_transform=plot_transform, fill=spot_fill,
            )

    if save_h5ad:
        adata.write(os.path.join(out_dir, "adata_emb.h5ad"))
    cols = [c for c in ["domain", "mclust", "kmeans", "leiden", "louvain", "EDiPath_domain"]
            if c in adata.obs]
    adata.obs[cols].to_csv(os.path.join(out_dir, "domain.csv"))
    print(f"Done -> {out_dir}")
    return results
