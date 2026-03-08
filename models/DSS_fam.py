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
        # α_u,i lookup: scipy CSR 행 선택 방식
        # ui_csr   : Y_UI        [NU, NI]
        # ubbi_csr : Y_UB × Y_BI [NU, NI]
        # ------------------------------------------------------------------
        A_UB_BI = self.ub_graph @ self.bi_graph
        
        # ------------------------------------------------------------------
        # λ: fixed balance between direct (UI) and indirect (UB-BI) signal
        # ------------------------------------------------------------------
        self.lambda_ubui = conf.get("lambda_ubui", 1.0)

        self.gamma_bonus = nn.Parameter(torch.tensor(0.01))
        self.beta        = nn.Parameter(torch.tensor(0.5))

        # Build propagation graphs FIRST, then alpha_pt
        self._build_graphs()

        # Precompute α_u,i CSR -> GPU sparse tensor (after graphs are on GPU)
        alpha_csr = self.ui_graph.tocsr() + self.lambda_ubui * A_UB_BI.tocsr()
        alpha_coo = alpha_csr.tocoo()
        i = torch.LongTensor([alpha_coo.row, alpha_coo.col])
        v = torch.FloatTensor(alpha_coo.data)
        self.alpha_pt = torch.sparse_coo_tensor(
            i, v, torch.Size(alpha_coo.shape)
        ).to(self.device).coalesce()

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

    @torch.no_grad()
    def _compute_alpha(self, users_1d):
        alpha_sparse = self.alpha_pt.index_select(0, users_1d)
        return alpha_sparse.to_dense()   # [bs, NI]

    @torch.no_grad()
    def _compute_alpha_items(self, users_1d, item_idx_t):
        """GPU 내에서 sparse tensor 슬라이싱으로 즉시 추출"""
        alpha_sparse = self.alpha_pt.index_select(0, users_1d) # [bs, NI]
        alpha_dense  = alpha_sparse.to_dense()                 # [bs, NI]
        return alpha_dense.index_select(1, item_idx_t)         # [bs, n_item]

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
        d       = self.emb_size

        users_1d = users.view(-1)       # [bs]
        UI_u = embs["UI_users"][users_1d]   # [bs, d]
        UB_u = embs["UB_users"][users_1d]   # [bs, d]
        UI_i = embs["UI_items"]             # [NI, d]
        UB_b = embs["UB_bundles"]           # [NB, d]

        # --- 배치 전체 번들 아이템 처리 (Vectorized) ---
        items_all = self.bundle_items[bundles].clamp(min=0)  # [bs, n_b, n_t]
        n_t       = items_all.shape[-1]
        mask_all  = (self.bundle_items[bundles] >= 0)        # [bs, n_b, n_t]
        
        flat_items = items_all.reshape(-1)
        unique_t, inv_idx = torch.unique(flat_items, return_inverse=True)

        alpha_cmp = self._compute_alpha_items(users_1d, unique_t)  # [bs, n_unique]

        pos_all = inv_idx.reshape(bs, n_b, n_t)  # [bs, n_b, n_t]
        alpha_all = alpha_cmp.gather(1, pos_all.view(bs, -1)).view(bs, n_b, n_t)
        
        alpha_all_raw = alpha_all.masked_fill(~mask_all, 0.0)
        
        beta = torch.sigmoid(self.beta)

        w_fam_all = torch.softmax(alpha_all.masked_fill(~mask_all, -1e9), dim=-1)
        w_nov_all = torch.softmax((-alpha_all).masked_fill(~mask_all, -1e9), dim=-1)

        n_unique     = unique_t.shape[0]
        unique_iembs = UI_i[unique_t]                              # [n_unique, d]

        # w를 unique 공간에서 집계 (backward: sgemm)
        w_fam_unique = torch.zeros(bs, n_b, n_unique, device=self.device)
        w_nov_unique = torch.zeros(bs, n_b, n_unique, device=self.device)
        idx4d = inv_idx.view(bs, n_b, n_t)
        w_fam_unique.scatter_add_(2, idx4d, w_fam_all)
        w_nov_unique.scatter_add_(2, idx4d, w_nov_all)

        v_trust_all  = w_fam_unique @ unique_iembs                 # [bs, n_b, d]
        v_motive_all = w_nov_unique @ unique_iembs                 # [bs, n_b, d]
        v_star_all   = beta * v_trust_all + (1.0 - beta) * v_motive_all

        n_valid_all    = mask_all.float().sum(dim=-1).clamp(min=1)
        mean_alpha_all = alpha_all_raw.sum(dim=-1) / n_valid_all
        bonus_all      = self.gamma_bonus * torch.log1p(mean_alpha_all)

        ub_b_all = UB_b[bundles] # [bs, n_b, d]
        
        score_all = (UI_u.unsqueeze(1) * v_star_all).sum(-1) + (UB_u.unsqueeze(1) * ub_b_all).sum(-1) + bonus_all

        return score_all  # [bs, n_b]

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
        # 벡터화된 BPR loss (파이썬 for loop 제거)
        bpr  = F.softplus(negs - pos.unsqueeze(1)).mean()

        if self.conf.get("c_lambda", 0.0) > 0:
            pos_bundles = bundles[:, 0]
            c_loss = self.cal_c_loss(embs, users.view(-1), pos_bundles)
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        # L2 reg는 optimizer weight_decay에서 처리되므로 여기에서는 0 반환
        reg_loss = torch.tensor(0.0, device=self.device)

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

        # 번들 루프 → chunk 벡터화
        CHUNK        = 256
        beta         = torch.sigmoid(self.beta)   # scalar
        il_chunks    = []
        bonus_chunks = []

        for start in range(0, NB, CHUNK):
            end     = min(start + CHUNK, NB)
            items_c = self.bundle_items[start:end].clamp(min=0)  # [C, T]
            mask_c  = self.bundle_items[start:end] >= 0          # [C, T]
            C, T    = items_c.shape

            alpha_c      = alpha_all[:, items_c.view(-1)].view(bs, C, T)  # [bs, C, T]
            alpha_c_raw  = alpha_c * mask_c.unsqueeze(0)
            alpha_c_soft = alpha_c.masked_fill(~mask_c.unsqueeze(0), -1e9)

            # Familiar / Novel head
            w_fam = torch.softmax( alpha_c_soft, dim=-1)          # [bs, C, T]
            w_nov = torch.softmax(-alpha_c_soft, dim=-1)

            iembs    = UI_i[items_c.view(-1)].view(C, T, d)       # [C, T, d]
            v_trust  = (w_fam.unsqueeze(-1) * iembs.unsqueeze(0)).sum(2)  # [bs, C, d]
            v_motive = (w_nov.unsqueeze(-1) * iembs.unsqueeze(0)).sum(2)
            v_star   = beta * v_trust + (1.0 - beta) * v_motive

            il_chunks.append((UI_u.unsqueeze(1) * v_star).sum(-1))  # [bs, C]

            n_valid    = mask_c.float().sum(-1).clamp(min=1)       # [C]
            mean_alpha = alpha_c_raw.sum(-1) / n_valid.unsqueeze(0)
            bonus_chunks.append(self.gamma_bonus * torch.log1p(mean_alpha))

        il_score    = torch.cat(il_chunks,    dim=1)  # [bs, NB]
        bonus_total = torch.cat(bonus_chunks, dim=1)  # [bs, NB]

        return il_score + ub_score + bonus_total