#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS_Base_Degree.py  —  Baseline + Item Degree PE + Bundle Size PE (spmm-optimized).

s_base(u, b) = UI_u · bundle_emb[b] + UB_u · UB_b[b]

bundle_emb[b] = spmm(bi_agg, UI_i + deg_pe) + size_pe[b]

Where:
  deg_pe  = degree_proj(log1p([bi_deg, ui_deg]))   [NI, d]  (per item)
  size_pe = size_pe_emb(bundle_size)               [NB, d]  (per bundle)

Optimization:
  bi_agg_graph [NB, NI] sparse (values=1/bundle_size)
  → One spmm replaces the [NB, n_t, d] 3D tensor aggregation.
  → bundle_emb pre-computed in get_embeddings, passed via embs dict.
  → compute_scores: simple lookup (no 3D tensor)
  → _score_all_bundles: two torch.mm calls
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laplace_transform(graph):
    rowsum_sqrt = sp.diags(1.0 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1.0 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    return rowsum_sqrt @ graph @ colsum_sqrt


def _to_tensor(graph, device):
    graph = graph.tocoo()
    indices = np.vstack((graph.row, graph.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(graph.data),
        torch.Size(graph.shape),
    ).coalesce().to(device)


def _np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice(
        [0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio]
    )
    return mask * values


# ---------------------------------------------------------------------------
# DSS_Base_Degree Model
# ---------------------------------------------------------------------------

class DSS_Base(nn.Module):
    """
    Baseline DSS with degree-based PE: s_base only, BPR loss.
    Uses sparse bi_agg_graph for O(1) bundle item-mean computation.

    PE Channels (per item):
      deg_pe = degree_proj(log1p([bi_deg, ui_deg]))  → [NI, d]
    PE Channels (per bundle):
      size_pe = size_pe_emb(bundle_size)             → [NB, d]

    bundle_emb = spmm(bi_agg, UI_i + deg_pe) + size_pe
    """

    def __init__(self, conf, raw_graph, bundle_info=None):
        super().__init__()
        self.conf        = conf
        self.device      = conf["device"]
        self.emb_size    = conf["embedding_size"]
        self.num_users   = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items   = conf["num_items"]
        self.num_layers  = conf["num_layers"]
        self.c_temp      = conf.get("c_temp", 0.2)

        self.UI_eps = conf.get("UI_ratio", 0.0)
        self.UB_eps = conf.get("UB_ratio", 0.0)
        self.BI_eps = conf.get("BI_ratio", 0.0)

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.beta_ui = float(conf.get("beta_ui", 0.1))

        # Bundle-level buffers
        if bundle_info is not None:
            self.register_buffer("bundle_items", bundle_info["bundle_items"])
            self.register_buffer("bundle_size",  bundle_info["bundle_size"])
        else:
            NB = self.num_bundles
            self.register_buffer("bundle_items", torch.full((NB, 1), -1, dtype=torch.long))
            self.register_buffer("bundle_size",  torch.ones(NB, dtype=torch.long))

        # Embedding parameters
        d = self.emb_size
        self.users_feature   = nn.Parameter(torch.FloatTensor(self.num_users,   d))
        self.items_feature   = nn.Parameter(torch.FloatTensor(self.num_items,   d))
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, d))
        for p in [self.users_feature, self.items_feature, self.bundles_feature]:
            nn.init.xavier_normal_(p)

        self.alpha_base = float(conf.get("alpha_base", 1.0))

        # ------------------------------------------------------------------
        # Degree PE: project [bi_deg, ui_deg] → d
        # ------------------------------------------------------------------
        deg_bi = torch.FloatTensor(self.bi_graph.sum(axis=0).A.ravel())  # [NI]
        deg_ui = torch.FloatTensor(self.ui_graph.sum(axis=0).A.ravel())  # [NI]
        self.register_buffer("item_degree", torch.stack([deg_bi, deg_ui], dim=-1))  # [NI, 2]

        self.degree_proj = nn.Linear(2, d, bias=False)

        # ------------------------------------------------------------------
        # Size PE: embedding table for bundle size  [1 .. max_size] → d
        # ------------------------------------------------------------------
        max_size = int(self.bundle_size.max().item())
        self.size_pe_emb = nn.Embedding(max_size + 1, d)

        # ------------------------------------------------------------------
        # Weight scalars (fixed, not learned — ablation-friendly)
        # ------------------------------------------------------------------
        self.w_deg  = float(conf.get("w_deg",  0.05))
        self.w_size = float(conf.get("w_size", 0.05))

        # Build propagation graphs + sparse aggregation matrix
        self._build_graphs()
        self._build_bi_agg_graph()

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def _make_prop_graph(self, bip, dropout=0.0):
        n, m = bip.shape
        prop = sp.bmat([
            [sp.csr_matrix((n, n)), bip],
            [bip.T, sp.csr_matrix((m, m))],
        ])
        if dropout > 0 and self.conf.get("aug_type") == "ED":
            coo  = prop.tocoo()
            vals = _np_edge_dropout(coo.data, dropout)
            prop = sp.coo_matrix((vals, (coo.row, coo.col)), shape=coo.shape).tocsr()
        return _to_tensor(_laplace_transform(prop), self.device)

    def _build_graphs(self):
        user_size   = self.ub_graph.sum(axis=1).A.ravel() + 1e-8
        ub_norm     = sp.diags(1.0 / user_size) @ self.ub_graph
        bundle_size = self.bi_graph.sum(axis=1).A.ravel() + 1e-8
        bi_norm     = sp.diags(1.0 / bundle_size) @ self.bi_graph
        ubi_graph   = self.ui_graph + self.beta_ui * (ub_norm @ bi_norm)

        self.UI_graph_ori = self._make_prop_graph(ubi_graph)
        self.BI_graph_ori = self._make_prop_graph(self.bi_graph)
        self.UB_graph_ori = self._make_prop_graph(self.ub_graph)
        self.UI_graph     = self._make_prop_graph(ubi_graph,     self.UI_eps)
        self.BI_graph     = self._make_prop_graph(self.bi_graph, self.BI_eps)
        self.UB_graph     = self._make_prop_graph(self.ub_graph, self.UB_eps)

    def _build_bi_agg_graph(self):
        """
        Sparse [NB, NI] matrix.  entry (b, i) = 1 / bundle_size[b].
        torch.spmm(bi_agg_graph, enriched_UI_i) → [NB, d]  item-mean per bundle.
        """
        NB, NI = self.num_bundles, self.num_items
        bi_np  = self.bundle_items.cpu().numpy()
        n_t    = bi_np.shape[1]

        b_idx  = np.repeat(np.arange(NB), n_t)
        i_idx  = bi_np.reshape(-1)
        valid  = i_idx >= 0
        b_v, i_v = b_idx[valid], i_idx[valid]

        sizes  = np.clip((bi_np >= 0).sum(axis=1), 1, None).astype(np.float32)
        wts    = 1.0 / sizes[b_v]

        bi_agg = sp.coo_matrix((wts, (b_v, i_v)), shape=(NB, NI)).tocsr()
        self.bi_agg_graph = _to_tensor(bi_agg, self.device)   # [NB, NI] sparse

    def _refresh_ed_graphs(self):
        user_size   = self.ub_graph.sum(axis=1).A.ravel() + 1e-8
        ub_norm     = sp.diags(1.0 / user_size) @ self.ub_graph
        bundle_size = self.bi_graph.sum(axis=1).A.ravel() + 1e-8
        bi_norm     = sp.diags(1.0 / bundle_size) @ self.bi_graph
        ubi_graph   = self.ui_graph + self.beta_ui * (ub_norm @ bi_norm)

        self.UI_graph = self._make_prop_graph(ubi_graph,     self.UI_eps)
        self.BI_graph = self._make_prop_graph(self.bi_graph, self.BI_eps)
        self.UB_graph = self._make_prop_graph(self.ub_graph, self.UB_eps)

    # ------------------------------------------------------------------
    # LightGCN propagation
    # ------------------------------------------------------------------

    def _propagate(self, graph, A_feat, B_feat, eps=0.0, test=False):
        feat  = torch.cat([A_feat, B_feat], dim=0)
        all_f = [feat]
        for _ in range(self.num_layers):
            feat = torch.spmm(graph, feat)
            if not test and self.conf.get("aug_type") == "Noise" and eps > 0:
                noise = torch.rand_like(feat)
                feat  = feat + torch.sign(feat) * F.normalize(noise, dim=-1) * eps
            all_f.append(F.normalize(feat, p=2, dim=1))
        coef = torch.ones(1, len(all_f), 1, device=self.device) / len(all_f)
        agg  = (torch.stack(all_f, dim=1) * coef).sum(dim=1)
        nA   = A_feat.shape[0]
        return agg[:nA], agg[nA:]

    # ------------------------------------------------------------------
    # Embedding extraction — pre-computes bundle_emb via spmm + PE
    # ------------------------------------------------------------------

    def get_embeddings(self, test=False):
        g_ui = self.UI_graph_ori if test else self.UI_graph
        g_bi = self.BI_graph_ori if test else self.BI_graph
        g_ub = self.UB_graph_ori if test else self.UB_graph

        UI_u, UI_i = self._propagate(g_ui, self.users_feature,   self.items_feature,   self.UI_eps, test)
        BI_b, BI_i = self._propagate(g_bi, self.bundles_feature, self.items_feature,   self.BI_eps, test)
        UB_u, UB_b = self._propagate(g_ub, self.users_feature,   self.bundles_feature, self.UB_eps, test)

        # --- Item-level Degree PE ---
        # deg_pe: [NI, d]  — learned projection of log1p(degree)
        deg_pe = self.degree_proj(torch.log1p(self.item_degree))  # [NI, d]

        # --- Aggregate (UI_i + w_deg * deg_pe) per bundle via spmm ---
        enriched_UI_i = UI_i + self.w_deg * deg_pe                # [NI, d]
        mean_emb = torch.spmm(self.bi_agg_graph, enriched_UI_i)   # [NB, d]

        # --- Bundle-level Size PE ---
        size_emb = self.size_pe_emb(self.bundle_size.clamp(
            max=self.size_pe_emb.num_embeddings - 1))              # [NB, d]
        bundle_emb = mean_emb + self.w_size * size_emb             # [NB, d]

        return {
            "UI_users":   UI_u,        # [N_u, d]
            "UB_users":   UB_u,        # [N_u, d]
            "BI_bundles": BI_b,        # [N_b, d]  (CL loss)
            "BI_items":   BI_i,        # [N_i, d]
            "UB_bundles": UB_b,        # [N_b, d]
            "UI_items":   UI_i,        # [N_i, d]  (kept for compatibility)
            "bundle_emb": bundle_emb,  # [N_b, d]  enriched item-mean repr (NEW)
        }

    def get_multi_modal_representations(self, test=False):
        return self.get_embeddings(test=test)

    # ------------------------------------------------------------------
    # compute_scores  (training) — simple lookup, no 3D tensor
    # s_base = UI_u · bundle_emb[b] + UB_u · UB_b[b]
    # ------------------------------------------------------------------

    def compute_scores(self, embs, users, bundles):
        """bundles: [bs, 1+neg] → [bs, 1+neg]"""
        bs, n_b = bundles.shape
        flat_b  = bundles.reshape(-1)    # [M]
        d       = self.emb_size
        M       = bs * n_b

        users = users.view(-1)
        eu_UI = embs["UI_users"][users].unsqueeze(1).expand(-1, n_b, -1).reshape(M, d)
        eu_UB = embs["UB_users"][users].unsqueeze(1).expand(-1, n_b, -1).reshape(M, d)

        b_emb = embs["bundle_emb"][flat_b]   # [M, d]  enriched item-mean
        b_ub  = embs["UB_bundles"][flat_b]   # [M, d]

        s = (eu_UI * b_emb).sum(-1) + (eu_UB * b_ub).sum(-1)   # [M]
        return s.reshape(bs, n_b)

    # ------------------------------------------------------------------
    # Contrastive Loss
    # ------------------------------------------------------------------

    def _info_nce(self, anchor, positive):
        anchor   = F.normalize(anchor,   p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        pos_score = (anchor * positive).sum(dim=1)
        ttl_score = torch.matmul(anchor, positive.T)
        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.exp(ttl_score / self.c_temp).sum(dim=1)
        return -torch.mean(torch.log(pos_score / ttl_score + 1e-8))

    def cal_c_loss(self, embs, users, pos_bundles):
        ui_u = embs["UI_users"][users]
        ub_u = embs["UB_users"][users]
        u_cl = self._info_nce(ui_u, ub_u)
        bi_b = embs["BI_bundles"][pos_bundles]
        ub_b = embs["UB_bundles"][pos_bundles]
        b_cl = self._info_nce(bi_b, ub_b)
        return (u_cl + b_cl) / 2.0

    # ------------------------------------------------------------------
    # Loss: BPR + optional Contrastive
    # ------------------------------------------------------------------

    def get_loss(self, embs, users, bundles, ED_drop=False):
        if ED_drop:
            self._refresh_ed_graphs()
        scores = self.compute_scores(embs, users, bundles)
        pos  = scores[:, 0]
        negs = scores[:, 1:]
        bpr  = sum(F.softplus(negs[:, i] - pos).mean()
                   for i in range(negs.shape[1])) / negs.shape[1]
        if self.conf.get("c_lambda", 0.0) > 0:
            pos_bundles = bundles[:, 0]
            c_loss = self.cal_c_loss(embs, users.view(-1), pos_bundles)
        else:
            c_loss = torch.tensor(0.0, device=self.device)
        return bpr, c_loss

    # ------------------------------------------------------------------
    # Evaluation — pure torch.mm
    # ------------------------------------------------------------------

    def evaluate(self, embs, users):
        return self._score_all_bundles(embs, users)

    def evaluate_with_components(self, embs, users):
        scores = self._score_all_bundles(embs, users)
        zeros  = torch.zeros_like(scores)
        return scores, scores, zeros, zeros

    def _score_all_bundles(self, embs, users):
        users = users.squeeze()
        if users.dim() == 0:
            users = users.unsqueeze(0)

        UI_u       = embs["UI_users"][users]   # [bs, d]
        UB_u       = embs["UB_users"][users]   # [bs, d]
        bundle_emb = embs["bundle_emb"]        # [NB, d]
        UB_b       = embs["UB_bundles"]        # [NB, d]

        # s = UI_u @ bundle_emb.T + UB_u @ UB_b.T
        return torch.mm(UI_u, bundle_emb.t()) + torch.mm(UB_u, UB_b.t())