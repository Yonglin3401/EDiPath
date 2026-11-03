"""Path-view + local-view graph contrastive learning (InfoNCE) + DEC.

The attention module aggregates directed-walk path features into a path-view
embedding ``z_path``; an MLP encodes the raw features into a local-view embedding
``z_local``. Symmetric InfoNCE aligns the two views, and DEC (KMeans-initialized
soft assignment + KL) refines the clustering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """DEC soft target distribution."""
    weight = (q ** 2) / torch.sum(q, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


class PathContrastiveEncoder(nn.Module):
    """Attention path aggregator (z_path) + local MLP (z_local) + optional DEC.

    Forward returns:
      z_path:  [N, hidden_dim] path-view embeddings (used for clustering / eval)
      z_local: [N, hidden_dim] local-feature embeddings
      q:       [N, n_clusters] soft cluster assignment (or None)
    """

    def __init__(
        self,
        feat_dim,
        hidden_dim,
        hidden_att_dim,
        dropout,
        n_clusters=10,
        dec_alpha=1.0,
        use_dec=True,
        contrast_tau=0.5,
        use_projector=True,
        projector_dim=None,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.n_clusters = n_clusters
        self.dec_alpha = dec_alpha
        self.use_dec = use_dec
        self.contrast_tau = float(contrast_tau)
        self.use_projector = bool(use_projector)
        proj_dim = int(projector_dim) if projector_dim else hidden_dim

        self.path_feature = None
        self.mask = None

        # Path attention aggregation
        self.att1 = nn.Linear(feat_dim, hidden_att_dim)
        self.att2 = nn.Linear(hidden_att_dim, 1)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # Path view encoder: agg(path) -> z_path
        self.path_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Local view encoder: X -> z_local
        self.local_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional projection heads for InfoNCE (SimCLR-style)
        if self.use_projector:
            self.path_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim),
            )
            self.local_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, proj_dim),
            )
        else:
            self.path_proj = nn.Identity()
            self.local_proj = nn.Identity()

        if use_dec:
            self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dim))
            nn.init.xavier_normal_(self.cluster_layer.data)
        else:
            self.register_parameter("cluster_layer", None)

    def aggregate_paths(self):
        att = self.att2(self.lrelu(self.att1(self.path_feature))).squeeze(-1)
        att = att.masked_fill(~self.mask.bool(), float("-inf"))
        att = F.softmax(att, dim=1)
        # rows with all -inf (no path) -> nan; fix to zeros
        att = torch.nan_to_num(att, nan=0.0)
        weighted = self.path_feature * att.unsqueeze(-1)
        return weighted.sum(1)

    def soft_assign(self, z):
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2)
            / self.dec_alpha
        )
        q = q.pow((self.dec_alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def forward(self, feature=None):
        """feature: raw gene expression [N, F] for the local view."""
        agg = self.aggregate_paths()
        z_path = self.path_encoder(agg)

        if feature is None:
            raise ValueError("Local view requires raw node features.")
        z_local = self.local_encoder(feature)

        q = (
            self.soft_assign(z_path)
            if self.use_dec and self.cluster_layer is not None
            else None
        )
        return z_path, z_local, q

    def info_nce_loss(self, z_path, z_local, temperature=None, symmetric=True):
        """InfoNCE between path view and local view (maximize agreement).

        L = -log exp(sim(z_u^path, z_u^local)/tau)
                 / sum_v exp(sim(z_u^path, z_v^local)/tau)

        If symmetric=True, also average the reverse direction (local as query).
        """
        tau = self.contrast_tau if temperature is None else float(temperature)
        h_path = F.normalize(self.path_proj(z_path), dim=1)
        h_local = F.normalize(self.local_proj(z_local), dim=1)

        logits = (h_path @ h_local.t()) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_pl = F.cross_entropy(logits, labels)
        if not symmetric:
            return loss_pl
        loss_lp = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_pl + loss_lp)

    def kl_loss(self, q, p):
        return F.kl_div(q.log(), p, reduction="batchmean")
