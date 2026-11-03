# -*- coding: utf-8 -*-
"""Named dataset presets for one-liner runs: python main_spatial.py --dataset CNS3

Data is **not** shipped with this repository. Set ``EDI_PATH_DATA_ROOT`` (or edit
``DATA_ROOT`` below) to the directory holding the prepared ``.h5ad`` files, and
place each dataset at the relative path listed in ``DATASET_PRESETS``.

DEC defaults:
  pretrain_epochs=200, dec_epochs=200, dec_interval=20, dec_tol=0.0,
  dec_kl_w=1, dec_n_clusters=16
Post-clustering uses each dataset's own ``n_clusters`` (may differ from DEC k).
"""

from __future__ import annotations

import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("EDI_PATH_DATA_ROOT", "./data")

# Default DEC block (shared by the presets below).
_DEFAULT_DEC = dict(
    pretrain_epochs=200,
    dec_epochs=200,
    dec_interval=20,
    dec_tol=0.0,
    dec_kl_w=1.0,
    dec_n_clusters=16,
)

# Recommended train + eval settings (default hyperparameters).
DATASET_PRESETS = {
    "CNS3": {
        "h5ad_path": os.path.join(DATA_ROOT, "CNS/CNS3.h5ad"),
        "dataset_key": "CNS3",
        "label_key": "Main_molecular_tissue_region",
        "n_clusters": 16,
        "use_rep": "X_pca",
        "n_pcs": 50,
        "spatial_k": 6,
        "gene_k": 5,
        "walk_time": 5,
        "walk_len": 5,
        "walk_q": 0.2,
        "hidden_dim": 128,
        "hidden_att_dim": 64,
        "dropout": 0.2,
        "tau_quantile": 0.3,
        "use_homogeneity_weight": 1,
        "lr": 1e-3,
        "force_n_clusters": 1,
        "cluster_eval": 1,
        **_DEFAULT_DEC,
        "dec_epochs": 100,
        "dec_kl_w": 0.05,
        "dec_n_clusters": 16,
        "plateau_patience": 35,
    },
    "DPI_2dpi": dict(
        h5ad_path=os.path.join(DATA_ROOT, "DPI/2DPI_1.h5ad"),
        dataset_key="DPI_2dpi",
        label_key="Annotation",
        n_clusters=18,
        use_rep="X_pca",
        n_pcs=50,
        spatial_k=6,
        gene_k=2,
        walk_time=8,
        walk_len=3,
        walk_q=0.3,
        hidden_dim=64,
        hidden_att_dim=32,
        dropout=0.2,
        tau_quantile=0.3,
        use_homogeneity_weight=1,
        lr=1e-3,
        force_n_clusters=1,
        cluster_eval=1,
        **_DEFAULT_DEC,
    ),
    # DLPFC Visium (7 layers); place a prepared 151673.h5ad under data/dlpfc/.
    "151673": {
        "h5ad_path": os.path.join(DATA_ROOT, "dlpfc/151673.h5ad"),
        "dataset_key": "151673",
        "label_key": "layer_guess",
        "n_clusters": 7,
        "use_rep": "X_pca",
        "n_pcs": 50,
        "spatial_k": 6,
        "gene_k": 4,
        "walk_time": 5,
        "walk_len": 5,
        "walk_q": 0.3,
        "hidden_dim": 64,
        "hidden_att_dim": 32,
        "dropout": 0.2,
        "tau_quantile": 0.3,
        "use_homogeneity_weight": 1,
        "lr": 1e-3,
        "force_n_clusters": 1,
        "cluster_eval": 1,
        **_DEFAULT_DEC,
        "dec_epochs": 100,
        "dec_kl_w": 0.05,
        "dec_n_clusters": 7,
        "plateau_patience": 35,
    },
    # 10x Visium breast cancer (example .h5ad under datasets/); fine_annot_type (20 classes).
    "Breast_cancer": {
        "h5ad_path": os.path.join(_REPO_ROOT, "datasets", "Breast_cancer.h5ad"),
        "dataset_key": "Breast_cancer",
        "label_key": "fine_annot_type",
        "n_clusters": 20,
        "use_rep": "X_pca",
        "n_pcs": 50,
        "spot_fill": 0.8,
        "spatial_k": 6,
        "gene_k": 4,
        "walk_time": 5,
        "walk_len": 5,
        "walk_q": 0.3,
        "hidden_dim": 64,
        "hidden_att_dim": 32,
        "dropout": 0.2,
        "tau_quantile": 0.3,
        "use_homogeneity_weight": 1,
        "lr": 1e-3,
        "force_n_clusters": 1,
        "cluster_eval": 1,
        **_DEFAULT_DEC,
        "dec_epochs": 100,
        "dec_kl_w": 0.05,
        "dec_n_clusters": 20,
        "plateau_patience": 35,
    },
}


def apply_dataset_preset(args, argv=None):
    """Fill CLI fields from DATASET_PRESETS[args.dataset].

    Explicit CLI flags always win over the preset.
    """
    key = getattr(args, "dataset", None) or ""
    if not key:
        return args
    if key not in DATASET_PRESETS:
        raise SystemExit(
            f"Unknown --dataset '{key}'. Available: {', '.join(sorted(DATASET_PRESETS))}"
        )

    preset = DATASET_PRESETS[key]
    argv = argv if argv is not None else []
    explicit = set()
    for a in argv:
        if not a.startswith("--"):
            continue
        name = a[2:].split("=", 1)[0].replace("-", "_")
        if name:
            explicit.add(name)

    for name, value in preset.items():
        if name in explicit:
            continue
        setattr(args, name, value)

    if not getattr(args, "dataset_key", ""):
        args.dataset_key = key
    return args
