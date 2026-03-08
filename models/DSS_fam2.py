#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS_fam2.py  —  Stage 2: Personalized β_u (Degree-Aware Familiar/Novel Mixing)

Stage 1 (DSS_fam.py):
    β = sigmoid(learnable scalar)  — 모든 유저 동일

Stage 2 (이 파일):
    β_u = ui_deg_u / (ui_deg_u + ub_deg_u + ε)  — 유저별 고유값, 파라미터 없음

    직접 구매 비율 높은 유저  →  β_u → 1  →  v_trust(familiar) 위주
    번들 구매 비율 높은 유저  →  β_u → 0  →  v_motive(novel)  위주

Score(u,b) = UI_u · v*_u,b + UB_u · UB_b + γ · log1p(mean_α)

v*_u,b = β_u · v_trust_u,b + (1−β_u) · v_motive_u,b
v_trust_u,b  = softmax(+α_u) · e_i  (아는 아이템 집중)
v_motive_u,b = softmax(−α_u) · e_i  (모르는 아이템 집중)
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
# DSS_fam2 Model
# ---------------------------------------------------------------------------

class DSS_Base(nn.Module):

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

        # α_u,i lookup: scipy CSR 행 선택 방식
        A_UB_BI = self.ub_graph @ self.bi_graph
        
        # ------------------------------------------------------------------
        # λ: fixed balance between direct (UI) and indirect (UB-BI) signal
        # ------------------------------------------------------------------
        self.lambda_ubui = conf.get("lambda_ubui", 1.0)
        
        # Precompute α_u,i CSR matrix to save memory allocations during training
        # alpha = ui_freq + lambda * ubbi_freq
        self.alpha_csr = self.ui_graph.tocsr() + self.lambda_ubui * A_UB_BI.tocsr()

        self.gamma_bonus = nn.Parameter(torch.tensor(0.01))

        # ------------------------------------------------------------------
        # Stage 2: β_u = ui_deg_u / (ui_deg_u + ub_deg_u + ε)
        # 파라미터 없음. 데이터에서 직접 계산해 buffer로 등록.
        #
        # β_u → 1 : 직접 구매 비율 높음 → Familiar(아는 아이템) 위주
        # β_u → 0 : 번들 구매 비율 높음 → Novel(새 아이템)   위주
        # ------------------------------------------------------------------
        ui_deg = np.array(self.ui_graph.sum(axis=1)).ravel().astype(np.float32)  # [NU]
        ub_deg = np.array(self.ub_graph.sum(axis=1)).ravel().astype(np.float32)  # [NU]
        beta_u = ui_deg / (ui_deg + ub_deg + 1e-8)                               # [NU]
        self.register_buffer("user_beta", torch.FloatTensor(beta_u))             # [NU]

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
    # α_u,i: per-user item familiarity  [bs, NI]
    # ------------------------------------------------------------------

    def _compute_alpha(self, users_1d):
        idx = users_1d.cpu().numpy()
        alpha_np = self.alpha_csr[idx].toarray().astype(np.float32)
        return torch.from_numpy(alpha_np).to(self.device)   # [bs, NI]

    def _compute_alpha_items(self, users_1d, item_idx_np):
        """훈련 시: 유니크 아이템만으로 제한해 precomputed alpha에서 slice"""
        user_np = users_1d.cpu().numpy()
        alpha_np = self.alpha_csr[user_np].toarray().astype(np.float32)[:, item_idx_np]
        return torch.from_numpy(alpha_np).to(self.device)  # [bs, n_item]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self, test=False):
        g_ui = self.UI_graph_ori if test else self.UI_graph
        g_bi = self.BI_graph_ori if test else self.BI_graph
        g_ub = self.UB_graph_ori if test else self.UB_graph

        UI_u, UI_i = self._propagate(g_ui, self.users_feature,   self.items_feature,   self.UI_eps, test)
        BI_b, BI_i = self._propagate(g_bi, self.bundles_feature, self.items_feature,   self.BI_eps, test)
        UB_u, UB_b = self._propagate(g_ub, self.users_feature,   self.bundles_feature, self.UB_eps, test)

        return {
            "UI_users":   UI_u,
            "UB_users":   UB_u,
            "BI_bundles": BI_b,
            "BI_items":   BI_i,
            "UB_bundles": UB_b,
            "UI_items":   UI_i,
        }

    def get_multi_modal_representations(self, test=False):
        return self.get_embeddings(test=test)

    # ------------------------------------------------------------------
    # compute_scores  (training)
    # ------------------------------------------------------------------

    def compute_scores(self, embs, users, bundles):
        bs, n_b  = bundles.shape
        flat_b   = bundles.reshape(-1)
        users_1d = users.view(-1)            # [bs]

        UI_u = embs["UI_users"][users_1d]    # [bs, d]
        UB_u = embs["UB_users"][users_1d]    # [bs, d]
        UI_i = embs["UI_items"]              # [NI, d]
        UB_b = embs["UB_bundles"]            # [NB, d]

        # 배치 번들 유니크 아이템만: [bs, n_unique] (~200 vs 123K)
        all_items = self.bundle_items[flat_b].clamp(min=0)
        unique_np = np.unique(all_items.cpu().numpy().ravel())
        unique_t  = torch.from_numpy(unique_np).long().to(self.device)
        alpha_cmp = self._compute_alpha_items(users_1d, unique_np)  # [bs, n_unique]

        beta_u = self.user_beta[users_1d]            # [bs]

        all_scores = []
        for j in range(n_b):
            b_j     = flat_b[j::n_b]                              # [bs]
            items_j = self.bundle_items[b_j].clamp(min=0)         # [bs, n_t]
            mask_j  = (self.bundle_items[b_j] >= 0)               # [bs, n_t]

            pos_j = torch.searchsorted(unique_t, items_j.reshape(-1)).reshape(bs, -1)
            alpha_j     = alpha_cmp.gather(1, pos_j)              # [bs, n_t]
            alpha_j_raw = alpha_j.masked_fill(~mask_j, 0.0)
            pad_inf     = ~mask_j

            w_fam = torch.softmax(alpha_j.masked_fill(pad_inf, -1e9), dim=-1)
            w_nov = torch.softmax((-alpha_j).masked_fill(pad_inf, -1e9), dim=-1)

            iembs_j    = UI_i[items_j]                             # [bs, n_t, d]
            v_trust_j  = (w_fam.unsqueeze(-1) * iembs_j).sum(1)   # [bs, d]
            v_motive_j = (w_nov.unsqueeze(-1) * iembs_j).sum(1)

            b_u = beta_u.unsqueeze(-1)                             # [bs, 1]
            v_star_j = b_u * v_trust_j + (1.0 - b_u) * v_motive_j  # [bs, d]

            n_valid_j    = mask_j.float().sum(dim=-1).clamp(min=1)
            mean_alpha_j = alpha_j_raw.sum(dim=-1) / n_valid_j
            bonus_j      = self.gamma_bonus * torch.log1p(mean_alpha_j)

            ub_b_j  = UB_b[b_j]
            score_j = (UI_u * v_star_j).sum(-1) + (UB_u * ub_b_j).sum(-1) + bonus_j
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
    # Loss
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

        bs           = users.shape[0]
        bundles_flat = bundles.view(-1)
        u_emb_0 = self.users_feature[users.view(-1)]
        b_emb_0 = self.bundles_feature[bundles_flat]

        safe_items = self.bundle_items[bundles_flat].clamp(min=0)
        i_emb_0    = self.items_feature[safe_items]
        valid_mask = (self.bundle_items[bundles_flat] >= 0).float()
        i_emb_0    = i_emb_0 * valid_mask.unsqueeze(-1)

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

        ub_score  = torch.mm(UB_u, UB_b.t())           # [bs, NB]
        alpha_all = self._compute_alpha(users)          # [bs, NI]

        # Stage 2: 유저별 β_u  [bs]
        beta_u = self.user_beta[users]                  # [bs]

        # 번들 루프 → chunk 벡터화 (유저별 β_u 유지)
        CHUNK        = 256
        b_u          = beta_u.unsqueeze(1).unsqueeze(-1)  # [bs, 1, 1]
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

            w_fam = torch.softmax( alpha_c_soft, dim=-1)          # [bs, C, T]
            w_nov = torch.softmax(-alpha_c_soft, dim=-1)

            iembs    = UI_i[items_c.view(-1)].view(C, T, d)       # [C, T, d]
            v_trust  = (w_fam.unsqueeze(-1) * iembs.unsqueeze(0)).sum(2)   # [bs, C, d]
            v_motive = (w_nov.unsqueeze(-1) * iembs.unsqueeze(0)).sum(2)

            # 유저별 β_u 믹싱: b_u [bs, 1, 1] broadcast
            v_star = b_u * v_trust + (1.0 - b_u) * v_motive       # [bs, C, d]

            il_chunks.append((UI_u.unsqueeze(1) * v_star).sum(-1)) # [bs, C]

            n_valid    = mask_c.float().sum(-1).clamp(min=1)       # [C]
            mean_alpha = alpha_c_raw.sum(-1) / n_valid.unsqueeze(0)
            bonus_chunks.append(self.gamma_bonus * torch.log1p(mean_alpha))

        il_score    = torch.cat(il_chunks,    dim=1)  # [bs, NB]
        bonus_total = torch.cat(bonus_chunks, dim=1)  # [bs, NB]

        return il_score + ub_score + bonus_total
