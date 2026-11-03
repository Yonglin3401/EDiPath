"""EDiPath entry point: entropy digraph + path/local InfoNCE + DEC.

Usage
-----
python main_spatial.py --synthetic 1 --use_cuda 0 --cluster_eval 0
python main_spatial.py --dataset CNS3 --gpu_id 0
python main_spatial.py --h5ad_path /path/to/data.h5ad --use_rep X_pca \
    --label_key Annotation --n_clusters 18 --gpu_id 0
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import warnings

import numpy as np
import torch

from configs.data_config import add_data_config
from configs.dataset_presets import DATASET_PRESETS, apply_dataset_preset
from configs.model_config import add_model_config
from configs.training_config import add_training_config
from datasets.spatial_data import load_spatial_h5ad, make_synthetic_spatial
from logger import Logger
from models.directed.edipath import EDiPathModel
from tasks.cluster_eval import DATASET_PLOT_DEFAULTS, cluster_eval_plot
from tasks.spatial_domain import SpatialDomainIdentification

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_spatial_config(parser):
    parser.add_argument(
        "--dataset", type=str, default="",
        choices=[""] + sorted(DATASET_PRESETS.keys()),
        help="Named preset. Fills path, labels, n_clusters and train hparams; "
             "any explicit CLI flag still overrides the preset.",
    )
    parser.add_argument("--h5ad_path", type=str, default="", help="Path to .h5ad spatial data")
    parser.add_argument("--use_rep", type=str, default="X_pca", help="adata.obsm key for features")
    parser.add_argument("--n_pcs", type=int, default=50, help="PCA dims if use_rep missing")
    parser.add_argument("--label_key", type=str, default="", help="adata.obs label column (eval only)")
    parser.add_argument("--synthetic", type=int, default=0, help="1: run synthetic smoke test")
    parser.add_argument("--synthetic_n", type=int, default=200)
    parser.add_argument("--n_clusters", type=int, default=-1,
                        help="Post-clustering / ARI k (mclust/kmeans/leiden/louvain); "
                             "-1 = infer from labels")
    parser.add_argument("--dec_n_clusters", type=int, default=-1,
                        help="DEC soft-assignment centers during training. "
                             "-1 = same as n_clusters")
    parser.add_argument("--spatial_k", type=int, default=6,
                        help="Spatial kNN candidate degree")
    parser.add_argument("--gene_k", type=int, default=6,
                        help="Top-k higher-entropy spatial neighbors kept as out-edges")
    parser.add_argument("--sim_tau", type=float, default=0.5,
                        help="Cosine threshold fallback for walk homophily")
    parser.add_argument("--tau_quantile", type=float, default=0.3,
                        help="Adaptive tau via spatial-edge sim quantile")
    # ---- Path / local graph contrastive learning ----
    parser.add_argument("--contrast_w", type=float, default=1.0,
                        help="InfoNCE contrastive loss weight")
    parser.add_argument("--contrast_tau", type=float, default=0.5,
                        help="InfoNCE temperature tau")
    parser.add_argument("--contrast_symmetric", type=int, default=1,
                        help="1: average path->local and local->path InfoNCE")
    parser.add_argument("--use_projector", type=int, default=1,
                        help="1: SimCLR-style projection heads before InfoNCE")
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--dec_epochs", type=int, default=200)
    parser.add_argument("--dec_kl_w", type=float, default=1.0,
                        help="DEC KL loss weight")
    parser.add_argument("--dec_interval", type=int, default=20)
    parser.add_argument("--dec_tol", type=float, default=0.0)
    parser.add_argument("--plateau_patience", type=int, default=35,
                        help="Pretrain early-stop patience")
    parser.add_argument(
        "--use_homogeneity_weight", type=int, default=1,
        help="Alias of --use_entropy_weight (edge weight = entropy gap)",
    )
    parser.add_argument(
        "--use_entropy_weight", type=int, default=-1,
        help="1/0: weight directed edges by entropy gap H(v)-H(u); "
             "-1 = follow --use_homogeneity_weight",
    )
    parser.add_argument("--path_cache_dir", type=str, default="./path_preprocess_st",
                        help="Directory for cached directed-walk path features")
    parser.add_argument("--save_embedding", type=str, default="", help="Optional .npy path to save z")
    # Downstream clustering + plots (mclust/kmeans/leiden/louvain)
    parser.add_argument("--cluster_eval", type=int, default=1,
                        help="1: after train, run 4-way clustering + ARI + figures")
    parser.add_argument("--force_n_clusters", type=int, default=1,
                        help="1: tune leiden/louvain resolution to match n_clusters")
    parser.add_argument("--leiden_res", type=float, default=None)
    parser.add_argument("--louvain_res", type=float, default=None)
    parser.add_argument("--plot_transform", type=str, default=None,
                        help="e.g. ccw90_flip_y; empty = dataset default / none")
    parser.add_argument("--spot_fill", type=float, default=None)
    parser.add_argument("--no_plot", type=int, default=0, help="1: skip saving spatial figures")
    parser.add_argument("--dataset_key", type=str, default="",
                        help="Optional key for plot defaults; default = h5ad basename")
    parser.add_argument("--section_filter", type=int, default=None,
                        help="Keep only obs[section_key]==this value (multi-section data)")
    parser.add_argument("--section_key", type=str, default="section")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(add_help=False)
    add_data_config(parser)
    add_model_config(parser)
    add_training_config(parser)
    add_spatial_config(parser)
    args = parser.parse_args()
    apply_dataset_preset(args, argv=sys.argv[1:])

    if not args.synthetic and not args.h5ad_path and not args.dataset:
        raise SystemExit(
            "Provide --dataset CNS3|DPI_2dpi|151673|Breast_cancer (preset), "
            "or --h5ad_path / --synthetic 1"
        )

    now = datetime.datetime.now()
    data_tag = "synthetic" if args.synthetic else (
        (args.dataset_key or args.dataset)
        or os.path.splitext(os.path.basename(args.h5ad_path))[0]
        or args.data_name
    )
    root = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root, "log", "edipath", data_tag)
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, now.strftime("%Y-%m-%d %H-%M-%S") + ".log"))
    logger.info(f"program start: {now}  (EDiPath: entropy digraph + InfoNCE + DEC)")
    if args.dataset:
        logger.info(f"Using dataset preset: {args.dataset}")
    seed_everything(args.seed)

    device = torch.device(
        f"cuda:{args.gpu_id}" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"device: {device}")

    if args.synthetic:
        dataset = make_synthetic_spatial(n=args.synthetic_n, seed=args.seed)
        adata = None
        logger.info(f"Synthetic dataset: n={dataset.num_node}, f={dataset.num_features}")
    else:
        if not args.h5ad_path:
            raise ValueError("Provide --h5ad_path or set --synthetic 1")
        label_key = args.label_key if args.label_key else None
        dataset, adata = load_spatial_h5ad(
            args.h5ad_path,
            use_rep=args.use_rep,
            n_pcs=args.n_pcs,
            label_key=label_key,
            section_filter=args.section_filter,
            section_key=args.section_key or "section",
            name=args.dataset_key or args.dataset or None,
        )
        logger.info(
            f"Loaded {args.h5ad_path}: n={dataset.num_node}, f={dataset.num_features}, "
            f"labels={'yes' if dataset.y is not None else 'no'}"
            + (f", section={args.section_filter}" if args.section_filter is not None else "")
        )

    n_clusters = args.n_clusters
    if n_clusters is None or n_clusters <= 0:
        if dataset.y is not None:
            n_clusters = int(len(np.unique(dataset.y)))
        else:
            n_clusters = 10
            logger.info("n_clusters not set; defaulting to 10")

    dec_n_clusters = getattr(args, "dec_n_clusters", None)
    if dec_n_clusters is None or int(dec_n_clusters) <= 0:
        dec_n_clusters = n_clusters
    else:
        dec_n_clusters = int(dec_n_clusters)
    logger.info(f"n_clusters(eval)={n_clusters}  dec_n_clusters(DEC)={dec_n_clusters}")

    # Reseed after I/O so digraph/walk/train are reproducible.
    seed_everything(args.seed)
    use_entropy_weight = (
        bool(args.use_homogeneity_weight)
        if int(getattr(args, "use_entropy_weight", -1)) < 0
        else bool(args.use_entropy_weight)
    )
    cache_name = (
        f"{dataset.name}_edipath_s{args.seed}"
        f"_sk{args.spatial_k}_gk{args.gene_k}"
        f"_ew{int(use_entropy_weight)}"
        f"_wt{args.walk_time}_wl{args.walk_len}_wq{args.walk_q}"
        f"_tq{args.tau_quantile}"
    )
    logger.info(
        f"walk/train knobs: plateau_patience={args.plateau_patience} cache={cache_name} "
        f"use_entropy_weight={int(use_entropy_weight)} "
        f"contrast_w={args.contrast_w} contrast_tau={args.contrast_tau} "
        f"symmetric={args.contrast_symmetric} projector={args.use_projector} "
        f"expr={'yes' if getattr(dataset, 'expr', None) is not None else 'fallback-features'}"
    )
    model = EDiPathModel(
        data_name=cache_name,
        n=dataset.num_node,
        feat_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        hidden_att_dim=args.hidden_att_dim,
        dropout=args.dropout,
        cof=args.walk_cof,
        delta=args.walk_delta,
        walk_q=args.walk_q,
        walk_len=args.walk_len,
        walk_time=args.walk_time,
        n_clusters=dec_n_clusters,
        sim_tau=args.sim_tau,
        tau_quantile=args.tau_quantile,
        exclude_source=True,
        use_dec=True,
        spatial_k=args.spatial_k,
        gene_k=args.gene_k,
        use_entropy_weight=use_entropy_weight,
        cache_dir=args.path_cache_dir,
        contrast_tau=args.contrast_tau,
        use_projector=bool(args.use_projector),
    )

    runner = SpatialDomainIdentification(
        logger=logger,
        dataset=dataset,
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrain_epochs=args.pretrain_epochs,
        dec_epochs=args.dec_epochs,
        contrast_w=args.contrast_w,
        contrast_tau=args.contrast_tau,
        contrast_symmetric=bool(args.contrast_symmetric),
        dec_kl_w=args.dec_kl_w,
        dec_interval=args.dec_interval,
        dec_tol=args.dec_tol,
        n_clusters=dec_n_clusters,
        plateau_patience=args.plateau_patience,
    )

    if args.save_embedding:
        np.save(args.save_embedding, runner.z)
        logger.info(f"Saved embedding to {args.save_embedding}")

    if adata is not None:
        adata.obsm["EDiPath"] = np.ascontiguousarray(runner.z.astype(np.float64))
        if "pred_dec" in runner.metrics:
            adata.obs["EDiPath_domain"] = runner.metrics["pred_dec"].astype(str)

        ds_key = args.dataset_key or dataset.name
        result_dir = os.path.join(root, "result", ds_key)
        os.makedirs(result_dir, exist_ok=True)

        out_h5ad = os.path.join(result_dir, f"{ds_key}_edipath.h5ad")
        try:
            adata.write_h5ad(out_h5ad)
            logger.info(f"Wrote annotated AnnData to {out_h5ad}")
        except Exception as e:
            logger.info(f"Skip writing h5ad ({e})")

        if args.cluster_eval:
            plot_defs = DATASET_PLOT_DEFAULTS.get(ds_key, {})
            label_key = args.label_key if args.label_key else plot_defs.get("label_key")
            seed_everything(args.seed)
            title_prefix = "EDiPath"
            logger.info(
                f"4-way cluster_eval: dataset_key={ds_key} k={n_clusters} "
                f"(DEC used dec_n_clusters={dec_n_clusters}) "
                f"force_n_clusters={bool(args.force_n_clusters)} "
                f"contrast_tau={args.contrast_tau} title={title_prefix} -> {result_dir}"
            )
            metrics = cluster_eval_plot(
                adata,
                name=ds_key,
                n_clusters=n_clusters,
                label_key=label_key,
                out_dir=result_dir,
                emb_key="EDiPath",
                title_prefix=title_prefix,
                leiden_res=args.leiden_res,
                louvain_res=args.louvain_res,
                force_n_clusters=bool(args.force_n_clusters),
                no_plot=bool(args.no_plot),
                save_h5ad=True,
                plot_transform=args.plot_transform,
                spot_fill=args.spot_fill,
                extra_meta={
                    "source": "main_spatial",
                    "method": "EDiPath",
                    "h5ad_path": args.h5ad_path,
                    "dec_n_clusters": dec_n_clusters,
                    "use_entropy_weight": use_entropy_weight,
                    "contrast_w": args.contrast_w,
                    "contrast_tau": args.contrast_tau,
                    "ari_kmeans_train": runner.metrics.get("ari"),
                    "ari_dec_train": runner.metrics.get("ari_dec"),
                },
            )
            logger.info(f"cluster_eval metrics: {metrics}")
