"""Unsupervised spatial-domain training: entropy digraph + path/local InfoNCE + DEC.

Gradients supervise the path attention via agreement between:
  - z_path  : attention-aggregated directed-path view
  - z_local : MLP on raw gene-expression features
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.optim import Adam

from models.predictor.contrastive import target_distribution


class SpatialDomainIdentification:
    def __init__(
        self,
        logger,
        dataset,
        model,
        device,
        lr=1e-3,
        weight_decay=5e-4,
        pretrain_epochs=200,
        dec_epochs=200,
        contrast_w=1.0,
        contrast_tau=0.5,
        contrast_symmetric=True,
        dec_kl_w=0.1,
        dec_interval=20,
        dec_tol=0.0,
        n_clusters=None,
        plateau_patience=20,
        plateau_delta=1e-4,
    ):
        self.logger = logger
        self.dataset = dataset
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.pretrain_epochs = pretrain_epochs
        self.dec_epochs = dec_epochs
        self.contrast_w = float(contrast_w)
        self.contrast_tau = float(contrast_tau)
        self.contrast_symmetric = bool(contrast_symmetric)
        self.dec_kl_w = dec_kl_w
        self.dec_interval = dec_interval
        self.dec_tol = dec_tol
        self.plateau_patience = plateau_patience
        self.plateau_delta = plateau_delta

        if n_clusters is not None:
            self.n_clusters = n_clusters
        elif dataset.y is not None:
            self.n_clusters = int(len(np.unique(dataset.y)))
        else:
            self.n_clusters = model.base_model.n_clusters

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.labels = None if dataset.y is None else np.asarray(dataset.y)

        # Sync temperature into the base model
        self.model.base_model.contrast_tau = self.contrast_tau

        self.history = {
            "pretrain_loss": [],
            "dec_loss": [],
            "contrast_loss": [],
            "kl_loss": [],
            "ari": [],
            "nmi": [],
        }
        self.execute()

    def _move_cached_tensors(self):
        self.model.base_model.path_feature = self.model.base_model.path_feature.to(self.device)
        self.model.base_model.mask = self.model.base_model.mask.to(self.device)
        self.model.raw_feature = self.model.raw_feature.to(self.device)
        self.model = self.model.to(self.device)

    def _contrast_loss(self, z_path, z_local):
        return self.model.base_model.info_nce_loss(
            z_path,
            z_local,
            temperature=self.contrast_tau,
            symmetric=self.contrast_symmetric,
        )

    def _eval_clustering(self, z):
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        pred = kmeans.fit_predict(z_np)
        metrics = {"pred": pred}
        if self.labels is not None:
            metrics["ari"] = adjusted_rand_score(self.labels, pred)
            metrics["nmi"] = normalized_mutual_info_score(self.labels, pred)
        return metrics

    def pretrain(self):
        self.logger.info(
            f"GCL pretrain (InfoNCE): epochs={self.pretrain_epochs}, "
            f"contrast_w={self.contrast_w}, tau={self.contrast_tau}, "
            f"symmetric={int(self.contrast_symmetric)}"
        )
        self.model.train()
        best = float("inf")
        stop = 0
        for epoch in range(self.pretrain_epochs):
            self.optimizer.zero_grad()
            z_path, z_local, _ = self.model()
            loss_c = self._contrast_loss(z_path, z_local)
            loss = self.contrast_w * loss_c
            loss.backward()
            self.optimizer.step()

            loss_v = float(loss.item())
            self.history["pretrain_loss"].append(loss_v)
            self.history["contrast_loss"].append(float(loss_c.item()))

            if loss_v + self.plateau_delta < best:
                best = loss_v
                stop = 0
            else:
                stop += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    f"[pretrain] epoch={epoch+1:03d} loss={loss_v:.6f} "
                    f"contrast={loss_c.item():.6f}"
                )

            if stop >= self.plateau_patience:
                self.logger.info(
                    f"Pretrain plateau at epoch {epoch+1} (best={best:.6f})."
                )
                break

    def init_dec_centers(self):
        self.model.eval()
        with torch.no_grad():
            z, _, _ = self.model()
        z_np = z.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=max(20, self.n_clusters * 2),
            random_state=42,
        )
        y_pred = kmeans.fit_predict(z_np)
        self.model.base_model.cluster_layer.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.device
        )
        self.logger.info(f"DEC centers initialized by KMeans (k={self.n_clusters}).")
        return y_pred

    def train_with_dec(self):
        self.logger.info(
            f"DEC fine-tune: epochs={self.dec_epochs}, kl_w={self.dec_kl_w}, "
            f"interval={self.dec_interval}, contrast_w={self.contrast_w}"
        )
        y_pred_last = self.init_dec_centers()
        self.model.train()
        tmp_p = None

        for epoch in range(self.dec_epochs):
            if epoch % self.dec_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    _, _, q = self.model()
                tmp_p = target_distribution(q.detach().cpu()).to(self.device)
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.mean(y_pred != y_pred_last)
                y_pred_last = y_pred.copy()

                msg = f"[dec] update targets epoch={epoch} delta_label={delta_label:.4f}"
                if self.labels is not None:
                    ari = adjusted_rand_score(self.labels, y_pred)
                    nmi = normalized_mutual_info_score(self.labels, y_pred)
                    self.history["ari"].append(ari)
                    self.history["nmi"].append(nmi)
                    msg += f" ARI={ari:.4f} NMI={nmi:.4f}"
                self.logger.info(msg)

                if epoch > 0 and delta_label < self.dec_tol:
                    self.logger.info("Reached DEC tolerance. Stop fine-tune.")
                    break
                self.model.train()

            self.optimizer.zero_grad()
            z_path, z_local, q = self.model()
            loss_c = self._contrast_loss(z_path, z_local)
            loss_kl = F.kl_div(q.log(), tmp_p, reduction="batchmean")
            loss = self.contrast_w * loss_c + self.dec_kl_w * loss_kl
            loss.backward()
            self.optimizer.step()

            self.history["dec_loss"].append(float(loss.item()))
            self.history["contrast_loss"].append(float(loss_c.item()))
            self.history["kl_loss"].append(float(loss_kl.item()))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    f"[dec] epoch={epoch+1:03d} loss={loss.item():.6f} "
                    f"contrast={loss_c.item():.6f} kl={loss_kl.item():.6f}"
                )

    def execute(self):
        t0 = time.time()

        self.logger.info("Preprocessing entropy digraph + directed walks ...")
        expr = getattr(self.dataset, "expr", None)
        self.model.preprocess(
            feature=self.dataset.x,
            coords=self.dataset.coords,
            expr=expr,
        )
        if self.model.graph_meta is not None:
            meta = {k: v for k, v in self.model.graph_meta.items() if k != "entropy"}
            self.logger.info(f"Graph meta: {meta}")
        self.logger.info(f"Walk tau: {self.model.walk_op._resolved_tau:.4f}")
        self._move_cached_tensors()

        # sync cluster count if needed
        if self.model.base_model.n_clusters != self.n_clusters:
            self.logger.info(
                f"Resize DEC layer: {self.model.base_model.n_clusters} -> {self.n_clusters}"
            )
            self.model.base_model.n_clusters = self.n_clusters
            layer = torch.nn.Parameter(
                torch.Tensor(self.n_clusters, self.model.base_model.hidden_dim)
            )
            torch.nn.init.xavier_normal_(layer.data)
            self.model.base_model.cluster_layer = layer.to(self.device)
            self.optimizer = Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        self.pretrain()
        self.train_with_dec()

        self.model.eval()
        with torch.no_grad():
            z, _, q = self.model()
        metrics = self._eval_clustering(z)
        if q is not None:
            pred_q = q.argmax(1).detach().cpu().numpy()
            metrics["pred_dec"] = pred_q
            if self.labels is not None:
                metrics["ari_dec"] = adjusted_rand_score(self.labels, pred_q)
                metrics["nmi_dec"] = normalized_mutual_info_score(self.labels, pred_q)

        self.z = z.detach().cpu().numpy()
        self.metrics = metrics
        elapsed = time.time() - t0
        self.logger.info(f"Finished in {elapsed:.2f}s")
        if self.labels is not None:
            self.logger.info(
                f"Final KMeans ARI={metrics.get('ari', float('nan')):.4f} "
                f"NMI={metrics.get('nmi', float('nan')):.4f}"
            )
            if "ari_dec" in metrics:
                self.logger.info(
                    f"Final DEC ARI={metrics['ari_dec']:.4f} NMI={metrics['nmi_dec']:.4f}"
                )
        return metrics
