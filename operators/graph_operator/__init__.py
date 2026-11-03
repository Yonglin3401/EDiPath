"""Spatial digraph construction from Shannon-entropy gradients."""

from .spatial_digraph import (
    build_entropy_digraph,
    build_spatial_knn_adj,
    spot_shannon_entropy,
)

__all__ = [
    "build_entropy_digraph",
    "build_spatial_knn_adj",
    "spot_shannon_entropy",
]
