# -*- coding: utf-8 -*-
"""Isolated mclust subprocess helper for EDiPath.

R/mclust fatal errors stay in this process and do not crash the main pipeline.
Usage: python mclust_helper.py <emb_npy> <n_clusters> <out_npy> [random_seed]
"""
import os
import sys

import numpy as np

# R_HOME is optional: if unset, rpy2 uses the R installation it was built against.
# Export R_HOME in the shell instead of hard-coding a machine-specific path here.


def main():
    emb_npy, n_clusters, out_npy = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2023

    data = np.load(emb_npy)
    if data.shape[1] > 50:
        from sklearn.decomposition import PCA
        data = PCA(n_components=50, random_state=seed).fit_transform(data)
    data = np.ascontiguousarray(data)

    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    robjects.r.library("mclust")
    robjects.r.assign("data_mat", data)
    robjects.r.assign("n_clust", n_clusters)
    robjects.r.assign("seed", seed)
    res = robjects.r(
        """
        set.seed(seed)
        data_mat <- as.matrix(data_mat)
        fit <- Mclust(data_mat, G=n_clust, modelNames='EEE')
        fit$classification
        """
    )
    np.save(out_npy, np.array(res).astype(str))


if __name__ == "__main__":
    main()
