# EDiPath

**Entropy-Directed Graph Contrastive Learning for Spatial Domain Identification.**

EDiPath identifies spatial domains in spatial transcriptomics (SRT) by combining a
Shannon-entropy–oriented directed graph, directed random-walk path encoding, and
path/local graph contrastive learning followed by DEC refinement.

---

## Installation

```bash
# Python >= 3.9
pip install -r requirements.txt
# install a PyTorch build matching your CUDA version, e.g.
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

`mclust` is optional. It runs through `rpy2` in an isolated subprocess; if
`rpy2`/R/mclust is unavailable, EDiPath automatically falls back to KMeans as the
primary domain assignment (Leiden/Louvain still run). To enable it, install R with
the `mclust` package and `pip install rpy2`, and optionally export
`R_HOME=/path/to/R`.

---

## Data

Provide an `.h5ad` file with:

- `adata.obsm["spatial"]` — 2-D spatial coordinates (required);
- a feature representation in `adata.obsm["X_pca"]` (optional; otherwise PCA of
  `adata.X` is used);
- `adata.X` — non-negative gene expression (used for Shannon entropy; optional but
  recommended);
- optionally `adata.obs[<label_key>]` — ground-truth domain labels (evaluation only).

An example Visium breast-cancer file is included at
`datasets/Breast_cancer.h5ad` (3,798 spots, `fine_annot_type` with 20 classes,
plus coarse `annot_type` / `ground_truth`).

Other named presets in `configs/dataset_presets.py` resolve paths under
`DATA_ROOT`, which defaults to `./data` and can be overridden with
`EDI_PATH_DATA_ROOT`:

```bash
export EDI_PATH_DATA_ROOT=/path/to/prepared_h5ad_files
```

---

## Quick start

```bash
# 1) Synthetic smoke test (no data / no GPU needed)
python main_spatial.py --synthetic 1 --use_cuda 0 --cluster_eval 0

# 2) Included example: Visium breast cancer
python main_spatial.py --dataset Breast_cancer --gpu_id 0

# 3) Other named presets (place the matching .h5ad under DATA_ROOT first)
python main_spatial.py --dataset CNS3 --gpu_id 0
python main_spatial.py --dataset 151673 --gpu_id 0

# 4) Run on an arbitrary .h5ad
python main_spatial.py \
  --h5ad_path /path/to/data.h5ad \
  --use_rep X_pca --label_key Annotation --n_clusters 18 --gpu_id 0
```

Outputs:

- training logs → `log/edipath/<dataset>/<timestamp>.log`
- annotated AnnData (`obsm["EDiPath"]`, predicted domains) → `result/<dataset>/`
- clustering metrics (`metrics.json`), `domain.csv` and spatial figures → `result/<dataset>/figures/`

---

## License

Released for research use. Please cite the corresponding paper if you use this code.
