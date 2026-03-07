#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS_Base_Degree.py  —  Personalized Weighted Bundle Embedding.

Score(u, b) = e_UI_u · v*_u,b  +  e_UB_u · e_UB_b

v*_u,b = Σ_i softmax(α_u,i) · e_UI_i   (personalized weighted avg)

α_u,i = Y_UI(u,i) + λ · (Y_UB × Y_BI)(u,i)
         ↑직접 구매 이력    ↑번들 통한 간접 이력

vs DSS_Base: e_b(mean) = (1/n) Σ e_i  (uniform, user-agnostic)

Key: Only the bundle pooling changes. Structure identical to DSS_Base.
Peak memory: [bs, n_t, d] per bundle (3D, no 4D tensor).
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


def _build_sparse_t(matrix, device):
    """scipy sparse matrix → transposed torch sparse tensor on device."""
    m_t = matrix.transpose().tocoo()
    indices = np.vstack((m_t.row, m_t.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(m_t.data),
        torch.Size(m_t.shape),
    ).coalesce().to(device)


# ---------------------------------------------------------------------------
# DSS_Base_Degree Model
# ---------------------------------------------------------------------------

class DSS_Base(nn.Module):
    """
    DSS Baseline with Personalized Weighted Bundle Embedding.

    bundle representation:
        DSS_Base       : e_b = uniform mean of item embeddings (user-agnostic)
        DSS_Base_Degree: v*_u,b = weighted mean using (Y_UI + λ·Y_UB×Y_BI)
                                  as per-item attention (user-specific)

    Scoring:
        Score(u,b) = e_UI_u · v*_u,b  +  e_UB_u · e_UB_b
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

        # ------------------------------------------------------------------
        # Attention signal 1: Y_UI (직접 구매 이력)
        # [NI, NU] transposed — for user batch lookup via spmm
        # ------------------------------------------------------------------
        self.ui_sparse_t = _build_sparse_t(self.ui_graph, self.device)  # [NI, NU]

        # ------------------------------------------------------------------
        # Attention signal 2: Y_UB × Y_BI (번들 통한 간접 이력)
        # [NI, NU] transposed
        # ------------------------------------------------------------------
        A_UB_BI = self.ub_graph @ self.bi_graph
        self.ub_bi_sparse_t = _build_sparse_t(A_UB_BI, self.device)    # [NI, NU]

        # ------------------------------------------------------------------
        # λ: learnable balance between direct (UI) and indirect (UB-BI) signal
        # ------------------------------------------------------------------
        self.lambda_ubui = nn.Parameter(torch.tensor(0.5))

        # Build propagation graphs
        self._build_graphs()

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
    # Helper: compute per-user α_u,i attention scores for a batch [bs]
    # Returns: [bs, NI]  (dense, from sparse mm)
    # ------------------------------------------------------------------

    def _compute_alpha(self, users_1d):
        bs = users_1d.shape[0]
        user_one_hot = torch.zeros((bs, self.num_users), device=self.device)
        user_one_hot.scatter_(1, users_1d.unsqueeze(1), 1.0)

        # [NI, NU] × [NU, bs] → [NI, bs] → [bs, NI]
        freq_ui   = torch.sparse.mm(self.ui_sparse_t,    user_one_hot.t()).t()
        freq_ubbi = torch.sparse.mm(self.ub_bi_sparse_t, user_one_hot.t()).t()

        lam = torch.clamp(self.lambda_ubui, min=0.0)  # λ ≥ 0 보장
        return freq_ui + lam * freq_ubbi  # [bs, NI]

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def get_embeddings(self, test=False):
        g_ui = self.UI_graph_ori if test else self.UI_graph
        g_bi = self.BI_graph_ori if test else self.BI_graph
        g_ub = self.UB_graph_ori if test else self.UB_graph

        UI_u, UI_i = self._propagate(g_ui, self.users_feature,   self.items_feature,   self.UI_eps, test)
        BI_b, BI_i = self._propagate(g_bi, self.bundles_feature, self.items_feature,   self.BI_eps, test)
        UB_u, UB_b = self._propagate(g_ub, self.users_feature,   self.bundles_feature, self.UB_eps, test)

        return {
            "UI_users":   UI_u,   # [N_u, d]
            "UB_users":   UB_u,   # [N_u, d]
            "BI_bundles": BI_b,   # [N_b, d]  (CL loss)
            "BI_items":   BI_i,   # [N_i, d]
            "UB_bundles": UB_b,   # [N_b, d]
            "UI_items":   UI_i,   # [N_i, d]
        }

    def get_multi_modal_representations(self, test=False):
        return self.get_embeddings(test=test)

    # ------------------------------------------------------------------
    # compute_scores (training)
    # Score(u,b) = e_UI_u · v*_u,b + e_UB_u · e_UB_b
    # v*_u,b = softmax(α_u,i) · e_UI_i  (personalized weighted avg)
    # bundles: [bs, 1+neg]
    # ------------------------------------------------------------------

    def compute_scores(self, embs, users, bundles):
        bs, n_b = bundles.shape
        flat_b  = bundles.reshape(-1)   # [bs * n_b]
        d       = self.emb_size

        users_1d = users.view(-1)       # [bs]
        UI_u = embs["UI_users"][users_1d]   # [bs, d]
        UB_u = embs["UB_users"][users_1d]   # [bs, d]
        UI_i = embs["UI_items"]             # [NI, d]
        UB_b = embs["UB_bundles"]           # [NB, d]

        # α_u,i: [bs, NI]  — compute once for all bundles
        alpha_all = self._compute_alpha(users_1d)

        all_scores = []
        for j in range(n_b):
            b_j     = flat_b[j::n_b]                              # [bs]
            items_j = self.bundle_items[b_j].clamp(min=0)         # [bs, n_t]
            mask_j  = (self.bundle_items[b_j] >= 0)               # [bs, n_t]

            # --- personalized attention weights ---
            alpha_j = alpha_all.gather(1, items_j)                # [bs, n_t]
            alpha_j = alpha_j.masked_fill(~mask_j, -1e9)
            w_j     = torch.softmax(alpha_j, dim=-1)              # [bs, n_t]

            # --- personalized bundle vector ---
            iembs_j = UI_i[items_j]                               # [bs, n_t, d]
            v_star_j = (w_j.unsqueeze(-1) * iembs_j).sum(dim=1)  # [bs, d]

            # --- score ---
            ub_b_j   = UB_b[b_j]                                  # [bs, d]
            score_j  = (UI_u * v_star_j).sum(-1) + (UB_u * ub_b_j).sum(-1)  # [bs]
            all_scores.append(score_j)

        return torch.stack(all_scores, dim=1)  # [bs, n_b]

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

        # L2 Regularization
        bs = users.shape[0]
        bundles_flat = bundles.view(-1)
        u_emb_0 = self.users_feature[users.view(-1)]
        b_emb_0 = self.bundles_feature[bundles_flat]

        safe_items = self.bundle_items[bundles_flat].clamp(min=0)
        i_emb_0 = self.items_feature[safe_items]
        valid_mask = (self.bundle_items[bundles_flat] >= 0).float()
        i_emb_0 = i_emb_0 * valid_mask.unsqueeze(-1)

        reg_loss = (1/2) * (u_emb_0.norm(2).pow(2) +
                            b_emb_0.norm(2).pow(2) +
                            i_emb_0.norm(2).pow(2)) / float(bs)

        return bpr, c_loss, reg_loss

    # ------------------------------------------------------------------
    # Evaluation
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

        UI_u = embs["UI_users"][users]    # [bs, d]
        UB_u = embs["UB_users"][users]    # [bs, d]
        UB_b = embs["UB_bundles"]         # [NB, d]
        UI_i = embs["UI_items"]           # [NI, d]
        NB   = self.num_bundles
        d    = self.emb_size

        # UB part: can be computed with a single mm (user-agnostic in bundle dim)
        ub_score = torch.mm(UB_u, UB_b.t())  # [bs, NB]

        # α_u,i: [bs, NI] — compute once
        alpha_all = self._compute_alpha(users)

        # UI part: loop over bundles (avoids [bs, NB, n_t] 3D tensor altogether)
        il_scores = []
        for b_idx in range(NB):
            items_b = self.bundle_items[[b_idx]].expand(users.shape[0], -1).clamp(min=0)  # [bs, n_t]
            mask_b  = (self.bundle_items[[b_idx]].expand(users.shape[0], -1) >= 0)        # [bs, n_t]

            alpha_b = alpha_all.gather(1, items_b)     # [bs, n_t]
            alpha_b = alpha_b.masked_fill(~mask_b, -1e9)
            w_b     = torch.softmax(alpha_b, dim=-1)   # [bs, n_t]

            iembs_b  = UI_i[items_b]                               # [bs, n_t, d]
            v_star_b = (w_b.unsqueeze(-1) * iembs_b).sum(dim=1)   # [bs, d]

            il_scores.append((UI_u * v_star_b).sum(-1))            # [bs]

        il_score = torch.stack(il_scores, dim=1)  # [bs, NB]

        return il_score + ub_score