"""EDiPath model: entropy digraph -> directed walks -> path/local InfoNCE -> DEC."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models.predictor.contrastive import PathContrastiveEncoder
from operators.graph_operator.spatial_digraph import build_entropy_digraph
from operators.random_walk_operator.spatial_directed_rw import SpatialDirectedRandomWalk


class EDiPathModel(nn.Module):
    """EDiPath: entropy-directed graph contrastive learning + DEC.

    Pipeline
    --------
    1. ``preprocess`` builds the entropy digraph from spatial coordinates and
       per-spot gene-expression entropy, then samples directed walks and caches
       the aggregated path features.
    2. ``forward`` produces the path-view embedding ``z_path`` (used for
       clustering) and the local-view embedding ``z_local`` (contrastive partner).
    3. Training drives symmetric InfoNCE between the two views, followed by DEC.
    """

    def __init__(
        self,
        data_name,
        n,
        feat_dim,
        hidden_dim,
        hidden_att_dim,
        dropout,
        cof,
        delta,
        walk_q,
        walk_len,
        walk_time,
        n_clusters=10,
        sim_tau=0.5,
        tau_quantile=0.3,
        exclude_source=True,
        use_dec=True,
        spatial_k=6,
        gene_k=6,
        use_entropy_weight=True,
        cache_dir="./path_preprocess_st",
        contrast_tau=0.5,
        use_projector=True,
    ):
        super().__init__()
        self.n = n
        self.feat_dim = feat_dim
        self.spatial_k = spatial_k
        self.gene_k = gene_k
        self.use_entropy_weight = bool(use_entropy_weight)

        self.walk_op = SpatialDirectedRandomWalk(
            n=n,
            data_name=data_name,
            cof=cof,
            delta=delta,
            walk_q=walk_q,
            walk_len=walk_len,
            walk_time=walk_time,
            sim_tau=sim_tau,
            tau_quantile=tau_quantile,
            exclude_source=exclude_source,
            cache_dir=cache_dir,
        )
        self.base_model = PathContrastiveEncoder(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            hidden_att_dim=hidden_att_dim,
            dropout=dropout,
            n_clusters=n_clusters,
            use_dec=use_dec,
            contrast_tau=contrast_tau,
            use_projector=use_projector,
        )
        self.raw_feature = None
        self.graph_meta = None

    def _sparse_to_digraph(self, adj: sp.spmatrix):
        adj = adj.tocoo()
        g = nx.DiGraph()
        g.add_nodes_from(range(self.n))
        for i, j, w in zip(adj.row, adj.col, adj.data):
            if i == j:
                continue
            g.add_edge(int(i), int(j), weight=float(w))
        return g

    def preprocess(self, feature, coords=None, adj=None, expr=None):
        """Build the entropy digraph, run directed walks, cache path features.

        Parameters
        ----------
        feature : array-like [N, F]
            Node features used by GCL / walks (typically PCA).
        coords : array-like [N, 2], required if adj is None
        adj : scipy sparse, optional prebuilt digraph adjacency
        expr : array-like [N, G], optional
            Non-negative gene expression for Shannon entropy. Preferred over
            ``feature`` when building the entropy digraph.
        """
        feature = np.asarray(feature, dtype=np.float32)
        self.raw_feature = torch.FloatTensor(feature)
        self.walk_op.feature = self.raw_feature

        if adj is None:
            if coords is None:
                raise ValueError("Either coords or adj must be provided for spatial EDiPath.")
            digraph_adj, _, meta = build_entropy_digraph(
                coords=np.asarray(coords),
                features=feature,
                spatial_k=self.spatial_k,
                top_k=self.gene_k,
                use_entropy_weight=self.use_entropy_weight,
                expr=None if expr is None else np.asarray(expr),
            )
            self.graph_meta = meta
            adj = digraph_adj
        else:
            self.graph_meta = {"n_directed_edges": int(sp.coo_matrix(adj).nnz)}

        self.walk_op.graph = self._sparse_to_digraph(adj)
        path_feature, mask = self.walk_op.walk()
        self.base_model.path_feature = path_feature
        self.base_model.mask = mask
        return path_feature, mask

    def forward(self, idx=None):
        z_path, z_local, q = self.base_model(self.raw_feature)
        if idx is None:
            return z_path, z_local, q
        q_idx = None if q is None else q[idx]
        return z_path[idx], z_local[idx], q_idx

    def get_embedding(self):
        self.eval()
        with torch.no_grad():
            z, _, q = self.forward()
        return z, q
