"""Directed random walk over the entropy digraph and path-feature construction."""

from __future__ import annotations

import os
import time

import numpy as np
import torch


class SpatialDirectedRandomWalk:
    """Sample directed walks and build attention-path features.

    For every node, ``walk_time`` directed walks of length ``walk_len`` are
    sampled on the entropy digraph. Transition probabilities mix topology
    (structural role) and feature cosine similarity. Each walk is compressed to
    a single feature vector with geometric positional weights ``cof``; the
    number of walks is adapted until the running mean path feature stabilizes
    (delta < ``delta``).
    """

    def __init__(
        self,
        n,
        data_name,
        cof,
        delta,
        walk_q,
        walk_len,
        walk_time,
        sim_tau=0.5,
        tau_quantile=None,
        exclude_source=True,
        cache_dir="./path_preprocess_st",
    ):
        self.graph = None
        self.feature = None  # torch FloatTensor [N, F]
        self.feat_np = None  # L2-normalized numpy for fast cosine
        self.n = n
        self.data_name = data_name
        self.delta = delta
        self.cof = cof
        self.walk_q = walk_q
        self.walk_len = walk_len
        self.walk_time = walk_time
        self.sim_tau = sim_tau
        self.tau_quantile = tau_quantile
        self.exclude_source = exclude_source
        self.cache_dir = cache_dir

        self.succ = None
        self.pred = None
        self.p_feat = {}
        self.p_topo = {}
        self.p_dst = {}
        self.weights = {}
        self._resolved_tau = sim_tau

    def _cos_np(self, u, v):
        return float(np.dot(self.feat_np[u], self.feat_np[v]))

    def _prepare_feat(self):
        x = self.feature.detach().cpu().numpy().astype(np.float64)
        nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        self.feat_np = x / nrm

    def _cache_neighbors(self):
        self.succ = [list(self.graph.successors(i)) for i in range(self.n)]
        self.pred = [list(self.graph.predecessors(i)) for i in range(self.n)]

    def _init_edge_probs(self):
        self._prepare_feat()
        self._cache_neighbors()
        edge_sims = []
        undirected_nbrs = [set(self.succ[i]) | set(self.pred[i]) for i in range(self.n)]

        for u, v in self.graph.edges:
            cosine = self._cos_np(u, v)
            w = float(self.graph[u][v].get("weight", 1.0))
            self.p_feat[(u, v)] = cosine * w
            if self.graph.has_edge(v, u):
                self.p_feat[(v, u)] = self._cos_np(v, u) * float(self.graph[v][u].get("weight", 1.0))
            else:
                self.p_feat[(v, u)] = cosine

            neigh_u = undirected_nbrs[u]
            neigh_v = undirected_nbrs[v]
            comm = len(neigh_u & neigh_v)
            deg_v = max(len(neigh_v), 1)
            deg_u = max(len(neigh_u), 1)
            self.p_topo[(u, v)] = 1.0 - (comm + 1) / deg_v
            self.p_topo[(v, u)] = 1.0 - (comm + 1) / deg_u
            edge_sims.append(cosine)

        if self.tau_quantile is not None and len(edge_sims) > 0:
            self._resolved_tau = float(np.quantile(edge_sims, self.tau_quantile))
        else:
            self._resolved_tau = float(self.sim_tau)

    def _homophily_entropy(self, walk):
        if len(walk) <= 1:
            return 0.0
        src = self.feat_np[walk[0]]
        labels = []
        for i, node in enumerate(walk):
            sim = float(np.dot(src, self.feat_np[node]))
            labels.append(i if sim < self._resolved_tau else 0)
        vals, counts = np.unique(labels, return_counts=True)
        p = counts.astype(np.float64) / counts.sum()
        return float(-(p * np.log(p + 1e-12)).sum())

    def _transition(self, cur):
        succ = self.succ[cur]
        pred = self.pred[cur]
        if len(succ) == 0 and len(pred) == 0:
            return cur
        if len(succ) == 0:
            cur_nbrs = pred
            direction = 0
        elif np.random.random() < self.walk_q:
            seen = set()
            cur_nbrs = []
            for x in pred + succ:
                if x not in seen:
                    seen.add(x)
                    cur_nbrs.append(x)
            direction = 2
        else:
            cur_nbrs = succ
            direction = 1

        key = (cur, direction)
        if key not in self.p_dst:
            p_topo_dst = np.array([self.p_topo.get((cur, nb), 0.0) for nb in cur_nbrs], dtype=np.float64)
            p_feat_dst = np.array([self.p_feat.get((cur, nb), 0.0) for nb in cur_nbrs], dtype=np.float64)
            p_topo = np.exp(p_topo_dst - p_topo_dst.max())
            p_topo /= p_topo.sum() + 1e-12
            p_feat = np.exp(p_feat_dst - p_feat_dst.max())
            p_feat /= p_feat.sum() + 1e-12
            p_dst = (p_topo + p_feat) / 2.0
            p_dst /= p_dst.sum() + 1e-12
            self.p_dst[key] = p_dst
        return int(np.random.choice(cur_nbrs, p=self.p_dst[key]))

    def _path_feature(self, walk):
        nodes = walk[1:] if (self.exclude_source and len(walk) > 1) else walk
        if len(nodes) == 0:
            return torch.zeros(self.feature.shape[1], dtype=torch.float32)
        path = self.feature[torch.tensor(nodes, dtype=torch.long)]
        return self._weighted(path)

    def _weighted(self, path):
        length = path.shape[0]
        if length not in self.weights:
            weights = self.cof ** torch.arange(length, dtype=torch.float32)
            weights = weights / weights.sum()
            self.weights[length] = weights
        return torch.mv(path.t(), self.weights[length])

    def walk(self):
        tq = "none" if self.tau_quantile is None else f"{self.tau_quantile:.3f}"
        tag = (
            f"{self.cof},{self.delta},{self.walk_q},{self.walk_len},{self.walk_time},"
            f"sim{self.sim_tau:.3f},tq{tq},ex{int(self.exclude_source)}"
        )
        path = os.path.join(self.cache_dir, self.data_name, tag)
        feat_path = os.path.join(path, "feat.pt")
        mask_path = os.path.join(path, "mask.pt")

        if os.path.exists(feat_path) and os.path.exists(mask_path):
            path_feature = torch.load(feat_path, map_location="cpu")
            mask = torch.load(mask_path, map_location="cpu")
            return path_feature, mask

        os.makedirs(path, exist_ok=True)
        self._init_edge_probs()

        max_n = 0
        path_lists = []
        t0 = time.time()
        report_every = max(1, self.n // 10)

        for node in range(self.n):
            delta = self.delta
            walk_time = 0
            path_feature_list = None
            while delta >= self.delta or walk_time < self.walk_time:
                walk = [node]
                walk_time += 1
                for _ in range(self.walk_len):
                    walk.append(self._transition(walk[-1]))

                path_feat = self._path_feature(walk)
                if walk_time == 1:
                    path_feature_list = path_feat
                    delta = self.delta
                elif walk_time == 2:
                    path_feature_avg_last = path_feature_list
                    path_feature_list = torch.stack((path_feature_list, path_feat), dim=0)
                    delta = torch.norm(path_feature_avg_last - path_feature_list.mean(0), 2).item()
                else:
                    path_feature_avg_last = path_feature_list.mean(0)
                    path_feature_list = torch.cat((path_feature_list, path_feat.unsqueeze(0)), dim=0)
                    delta = torch.norm(path_feature_avg_last - path_feature_list.mean(0), 2).item()

            if path_feature_list.ndim == 1:
                path_feature_list = path_feature_list.unsqueeze(0)
            path_lists.append(path_feature_list)
            max_n = max(path_feature_list.shape[0], max_n)

            if (node + 1) % report_every == 0 or node + 1 == self.n:
                elapsed = time.time() - t0
                print(
                    f"[EDiPath walk] {node+1}/{self.n} nodes, "
                    f"elapsed={elapsed:.1f}s, eta={elapsed/(node+1)*(self.n-node-1):.1f}s",
                    flush=True,
                )

        padded = torch.zeros(self.n, max_n, self.feature.shape[1])
        mask = torch.zeros(self.n, max_n)
        for i, wf in enumerate(path_lists):
            padded[i, : wf.shape[0]] = wf
            mask[i, : wf.shape[0]] = 1

        torch.save(padded, feat_path)
        torch.save(mask, mask_path)
        return padded, mask
