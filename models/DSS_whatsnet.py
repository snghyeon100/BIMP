#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS  —  WHATsNet-style Bundle↔Item Bidirectional Message Passing

Score(u, b) = s_base(u,b) + λ * s_new(u,b)

  s_base(u,b) = UI_u · mean(b_items_UI) + UB_u · UB_b

  Bidirectional layers (L rounds):
    Init:
      X_i^(0) = LN(W_ui·UI_i + W_bi·BI_i)          # item state
      H_b^(0) = LN(W_biB·BI_b)                       # bundle state

    Each layer l:
      (A) Item→Bundle:   Z_{i,b} = X_i^(l-1) + W_pe·PE(i,b)
                         V_b  = WithinATT(Z) = MAB(Z, MAB(I, Z))
                         H_b^(l) = MAB(H_b^(l-1), V_b)    ← bundle as Query

      (B) Bundle→Item:   E_{b,i} = H_b^(l) + W_pe·PE(i,b)
                         G_i  = WithinATT(E) = MAB(E, MAB(I, E))
                         X_i^(l) = MAB(X_i^(l-1), G_i)    ← item as Query

  Stage 2 (user-Query softmax weighted sum, Option A):
    q_u = LN(W_uiq·UI_u + W_ubq·UB_u)
    logits = (Wq·q_u) · (Wk·V_b^T) / sqrt(d)
    alpha  = softmax(logits / tau)   [padded positions = -inf]
    h_{b|u} = Σ alpha_i * (Wv·V_b_i)
    s_new  = (Wq·q_u) · h_{b|u}

  λ = sigmoid(logit_lambda),   logit_lambda initialised = -3  (λ≈0.047 at start)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def cal_bpr_loss(pred):
    """pred: [bs, 1+neg_num]"""
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos  = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos  = pred[:, 0].unsqueeze(1)
    return torch.mean(-torch.log(torch.sigmoid(pos - negs)))


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    return rowsum_sqrt @ graph @ colsum_sqrt


def to_tensor(graph):
    graph = graph.tocoo()
    indices = np.vstack((graph.row, graph.col))
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(graph.data),
        torch.Size(graph.shape),
    ).coalesce()


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice(
        [0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio]
    )
    return mask * values


# ---------------------------------------------------------------------------
# MAB  —  Multihead Attention Block
# ---------------------------------------------------------------------------

class MAB(nn.Module):
    """
    MAB(X, Y):  Query = X [B, n_q, d],  Key/Value = Y [B, n_k, d]
    mask_y: [B, n_k] bool — True for valid positions (pads = False)
    Returns: [B, n_q, d]
    """
    def __init__(self, d: int, num_heads: int, attn_drop: float = 0.0):
        super().__init__()
        assert d % num_heads == 0, "d must be divisible by num_heads"
        self.h  = num_heads
        self.hd = d // num_heads

        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff  = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.scale = math.sqrt(self.hd)

    def forward(self, X, Y, mask_y=None):
        B, n_q, d = X.shape
        n_k       = Y.shape[1]
        h, hd     = self.h, self.hd

        Q = self.W_q(X).view(B, n_q, h, hd).transpose(1, 2)   # [B, h, n_q, hd]
        K = self.W_k(Y).view(B, n_k, h, hd).transpose(1, 2)   # [B, h, n_k, hd]
        V = self.W_v(Y).view(B, n_k, h, hd).transpose(1, 2)   # [B, h, n_k, hd]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B, h, n_q, n_k]
        if mask_y is not None:
            # mask_y: [B, n_k] → [B, 1, 1, n_k]
            attn = attn.masked_fill(~mask_y.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                  # [B, h, n_q, hd]
        out = out.transpose(1, 2).contiguous().view(B, n_q, d)
        out = self.W_o(out)

        H = self.ln1(X + out)           # residual + layer-norm
        return self.ln2(H + self.ff(H)) # feed-forward + residual + layer-norm


# ---------------------------------------------------------------------------
# DSS Model
# ---------------------------------------------------------------------------

class DSS(nn.Module):
    """
    DSS with WHATsNet-style Bundle<->Item Bidirectional Message Passing.

    Embeddings (LightGCN-propagated):
      UI view  : UI_users [N_u,d],  UI_items [N_i,d]
      BI view  : BI_bundles[N_b,d], BI_items [N_i,d]
      UB view  : UB_users [N_u,d],  UB_bundles[N_b,d]

    s_base(u,b) = UI_u · mean(b_items_UI) + UB_u · UB_b
    Final score = s_base + sigmoid(logit_lambda) * s_new
    """

    def __init__(self, conf, raw_graph, bundle_info):
        super().__init__()
        self.conf        = conf
        self.device      = conf["device"]
        self.emb_size    = conf["embedding_size"]
        self.num_users   = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items   = conf["num_items"]
        self.num_layers  = conf["num_layers"]
        self.c_temp      = conf.get("c_temp", 0.2)

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # ------------------------------------------------------------------
        # Bundle-level precomputed tensors (no grad)
        # ------------------------------------------------------------------
        self.register_buffer("bundle_items",  bundle_info["bundle_items"])   # [N_b, n_t]
        self.register_buffer("bundle_size",   bundle_info["bundle_size"])
        self.register_buffer("bundle_degree", bundle_info["bundle_degree"])

        # ------------------------------------------------------------------
        # Embedding tables
        # ------------------------------------------------------------------
        d = self.emb_size
        self.users_feature   = nn.Parameter(torch.FloatTensor(self.num_users,   d))
        self.items_feature   = nn.Parameter(torch.FloatTensor(self.num_items,   d))
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, d))
        for p in [self.users_feature, self.items_feature, self.bundles_feature]:
            nn.init.xavier_normal_(p)

        # ------------------------------------------------------------------
        # Graph-level reinforcement
        # ------------------------------------------------------------------
        self.beta_ui = float(conf.get("beta_ui", 0.1))

        # ------------------------------------------------------------------
        # Dropout / noise ratios
        # ------------------------------------------------------------------
        self.UB_eps = conf.get("UB_ratio", 0.0)
        self.UI_eps = conf.get("UI_ratio", 0.0)
        self.BI_eps = conf.get("BI_ratio", 0.0)

        # ------------------------------------------------------------------
        # BIMP hyper-parameters
        # ------------------------------------------------------------------
        self.num_bimp_layers   = conf.get("num_bimp_layers", 1)
        self.num_inducing_bimp = conf.get("num_inducing_bimp", 4)
        self.num_heads         = conf.get("num_heads", 2)
        self.max_incident_K    = conf.get("max_incident_K", 30)
        self.eval_incident_K   = conf.get("eval_incident_K", self.max_incident_K)
        self.bimp_tau          = float(conf.get("bimp_tau", 1.0))
        attn_drop              = conf.get("attn_drop", 0.1)

        # ------------------------------------------------------------------
        # §1.1 Item initial state:  X_i^(0) = LN(W_ui·UI_i + W_bi·BI_i)
        # ------------------------------------------------------------------
        self.W_ui        = nn.Linear(d, d, bias=False)
        self.W_bi        = nn.Linear(d, d, bias=False)
        self.ln_item_init = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §1.2 Bundle initial state: H_b^(0) = LN(W_biB·BI_b)
        # ------------------------------------------------------------------
        self.W_biB          = nn.Linear(d, d, bias=False)
        self.ln_bundle_init = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §1.3 User query: q_u = LN(W_uiq·UI_u + W_ubq·UB_u)
        # ------------------------------------------------------------------
        self.W_uiq      = nn.Linear(d, d, bias=False)
        self.W_ubq      = nn.Linear(d, d, bias=False)
        self.ln_user_q  = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §5 Positional Encoding: W_pe: Linear(3 → d)
        #    channels: rank_ui_deg, rank_bi_deg, rev_rank_ui_deg
        # ------------------------------------------------------------------
        self.W_pe = nn.Linear(3, d, bias=False)

        # Precompute item degree vectors (UI and BI) for PE
        # These are computed post-graph-build so we store them as buffers
        # (filled in _build_graphs)
        self.register_buffer("_ui_item_deg", torch.zeros(self.num_items))
        self.register_buffer("_bi_item_deg", torch.zeros(self.num_items))

        # ------------------------------------------------------------------
        # Bidirectional MABs
        #   I→B direction: WithinATT inner/outer + main
        #   B→I direction: WithinATT inner/outer + main
        # ------------------------------------------------------------------
        h = self.num_heads
        self.mab_i2b_within = MAB(d, h, attn_drop=attn_drop)  # inner: MAB(I, tokens)
        self.mab_i2b_outer  = MAB(d, h, attn_drop=attn_drop)  # outer: MAB(tokens, S)
        self.mab_i2b_main   = MAB(d, h, attn_drop=attn_drop)  # H_b = MAB(H_b_prev, V_b)

        self.mab_b2i_within = MAB(d, h, attn_drop=attn_drop)  # inner: MAB(I, tokens)
        self.mab_b2i_outer  = MAB(d, h, attn_drop=attn_drop)  # outer: MAB(tokens, S)
        self.mab_b2i_main   = MAB(d, h, attn_drop=attn_drop)  # X_i = MAB(X_i_prev, G_i)

        # Learnable inducing points for WithinATT  [num_inducing_bimp, d]
        self.bimp_I = nn.Parameter(torch.FloatTensor(self.num_inducing_bimp, d))
        nn.init.xavier_normal_(self.bimp_I)

        # ------------------------------------------------------------------
        # §6 Stage2 Q/K/V projections
        # ------------------------------------------------------------------
        self.Wq_s2 = nn.Linear(d, d, bias=False)
        self.Wk_s2 = nn.Linear(d, d, bias=False)
        self.Wv_s2 = nn.Linear(d, d, bias=False)

        # ------------------------------------------------------------------
        # §7 Baseline residual scalar λ = sigmoid(logit_lambda)
        #    Init logit=-3 → λ≈0.047 (start close to baseline-only)
        # ------------------------------------------------------------------
        self.logit_lambda = nn.Parameter(torch.tensor(-3.0))

        # Config compatibility
        self.alpha_base = float(conf.get("alpha_base", 1.0))

        self._build_graphs()
        self._build_item_bundles()

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _make_prop_graph(self, bip, dropout=0.0):
        n, m  = bip.shape
        prop  = sp.bmat([
            [sp.csr_matrix((n, n)), bip],
            [bip.T, sp.csr_matrix((m, m))],
        ])
        if dropout > 0 and self.conf.get("aug_type") == "ED":
            coo  = prop.tocoo()
            vals = np_edge_dropout(coo.data, dropout)
            prop = sp.coo_matrix((vals, (coo.row, coo.col)), shape=coo.shape).tocsr()
        return to_tensor(laplace_transform(prop)).to(self.device)

    def _build_graphs(self):
        # Row-normalize UB and BI for reinforcement
        user_size   = self.ub_graph.sum(axis=1).A.ravel() + 1e-8
        ub_norm     = sp.diags(1.0 / user_size) @ self.ub_graph
        bundle_size = self.bi_graph.sum(axis=1).A.ravel() + 1e-8
        bi_norm     = sp.diags(1.0 / bundle_size) @ self.bi_graph
        ubi_graph   = self.ui_graph + self.beta_ui * (ub_norm @ bi_norm)

        self.UI_graph_ori = self._make_prop_graph(ubi_graph)
        self.BI_graph_ori = self._make_prop_graph(self.bi_graph)
        self.UB_graph_ori = self._make_prop_graph(self.ub_graph)

        self.UI_graph = self._make_prop_graph(ubi_graph,     self.UI_eps)
        self.BI_graph = self._make_prop_graph(self.bi_graph, self.BI_eps)
        self.UB_graph = self._make_prop_graph(self.ub_graph, self.UB_eps)

        # Item degree for PE (from raw graphs, CPU)
        ui_deg = np.array(self.ui_graph.sum(axis=0)).ravel().astype(np.float32)  # [N_i]
        bi_deg = np.array(self.bi_graph.sum(axis=0)).ravel().astype(np.float32)  # [N_i]
        self._ui_item_deg = torch.tensor(ui_deg, dtype=torch.float, device=self.device)
        self._bi_item_deg = torch.tensor(bi_deg, dtype=torch.float, device=self.device)

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
    # §3.2 Build item→bundle CSR (reverse adjacency)
    # ------------------------------------------------------------------

    def _build_item_bundles(self):
        """
        Build item_ptr [N_i+1] and item_adj [total_edges] (CSR format).
        item_adj[item_ptr[i]:item_ptr[i+1]] = bundle ids for item i.
        Padding -1 entries in bundle_items are excluded.
        """
        N_i  = self.num_items
        N_b  = self.num_bundles
        bi_np = self.bundle_items.cpu().numpy()   # [N_b, n_t]

        # Build lists
        from collections import defaultdict
        ib = defaultdict(list)
        for b in range(N_b):
            for it in bi_np[b]:
                if it >= 0:
                    ib[int(it)].append(b)

        ptr = [0]
        adj = []
        for i in range(N_i):
            bundles_i = ib.get(i, [])
            adj.extend(bundles_i)
            ptr.append(len(adj))

        self.register_buffer("item_ptr",
            torch.tensor(ptr, dtype=torch.long, device=self.device))
        self.register_buffer("item_adj",
            torch.tensor(adj, dtype=torch.long, device=self.device))

    # ------------------------------------------------------------------
    # LightGCN propagation
    # ------------------------------------------------------------------

    def _propagate(self, graph, A_feat, B_feat, eps, test):
        feat  = torch.cat([A_feat, B_feat], dim=0)
        all_f = [feat]
        for _ in range(self.num_layers):
            feat = torch.spmm(graph, feat)
            if not test and self.conf.get("aug_type") == "Noise" and eps > 0:
                noise = torch.rand_like(feat)
                feat  = feat + torch.sign(feat) * F.normalize(noise, dim=-1) * eps
            all_f.append(F.normalize(feat, p=2, dim=1))
        coef  = torch.ones(1, len(all_f), 1, device=self.device) / len(all_f)
        agg   = (torch.stack(all_f, dim=1) * coef).sum(dim=1)
        nA    = A_feat.shape[0]
        return agg[:nA], agg[nA:]

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
            "UI_users":   UI_u,
            "UI_items":   UI_i,
            "BI_bundles": BI_b,
            "BI_items":   BI_i,
            "UB_users":   UB_u,
            "UB_bundles": UB_b,
        }

    # ------------------------------------------------------------------
    # §1.1 – §1.3  Initial states and user query
    # ------------------------------------------------------------------

    def _bimp_item_init(self, UI_i, BI_i):
        """X_i^(0) = LN(W_ui·UI_i + W_bi·BI_i)  →  [N_i, d]"""
        return self.ln_item_init(self.W_ui(UI_i) + self.W_bi(BI_i))

    def _bimp_bundle_init(self, BI_b):
        """H_b^(0) = LN(W_biB·BI_b)  →  [N_b, d]"""
        return self.ln_bundle_init(self.W_biB(BI_b))

    def _user_query(self, UI_u, UB_u):
        """q_u = LN(W_uiq·UI_u + W_ubq·UB_u)  →  same shape as inputs"""
        return self.ln_user_q(self.W_uiq(UI_u) + self.W_ubq(UB_u))

    # ------------------------------------------------------------------
    # §5  Positional Encoding
    # ------------------------------------------------------------------

    def _compute_pe(self, items_2d, valid):
        """
        items_2d : [B, n_t]  item indices (clamped ≥0)
        valid    : [B, n_t]  bool (True = real item)
        Returns  : [B, n_t, d]  PE embedding to add to tokens
        """
        B, n_t = items_2d.shape
        ui_deg = self._ui_item_deg[items_2d]   # [B, n_t]
        bi_deg = self._bi_item_deg[items_2d]   # [B, n_t]

        # Per-bundle rank normalised to [0, 1]
        def rank_norm(deg, valid_mask):
            # sort descending within each bundle row
            # -inf for pads so they always sink to end
            deg_f = deg.float().masked_fill(~valid_mask, -1.0)
            # argsort twice for rank
            order = deg_f.argsort(dim=-1, descending=True)
            rank  = torch.zeros_like(deg_f)
            rank.scatter_(1, order, torch.arange(n_t, dtype=torch.float,
                                                 device=self.device).unsqueeze(0).expand(B, -1))
            # Normalise: n-1 denominator (n = count of valid items per bundle)
            n_valid = valid_mask.sum(1, keepdim=True).float().clamp(min=2) - 1   # [B,1]
            rank_n  = (rank / n_valid).clamp(0.0, 1.0)
            rank_n  = rank_n.masked_fill(~valid_mask, 0.0)
            return rank_n

        rk_ui  = rank_norm(ui_deg, valid)           # [B, n_t]  ascending rank
        rk_bi  = rank_norm(bi_deg, valid)
        rev_ui = (1.0 - rk_ui).masked_fill(~valid, 0.0)

        pe_raw = torch.stack([rk_ui, rk_bi, rev_ui], dim=-1)      # [B, n_t, 3]
        return self.W_pe(pe_raw)                                   # [B, n_t, d]

    # ------------------------------------------------------------------
    # §2/4  WithinATT
    # ------------------------------------------------------------------

    def _within_att(self, S_mab, T_mab, tokens, mask):
        """
        S = MAB(inducing, tokens)    # inducing queries tokens
        T = MAB(tokens,   S)         # tokens query summary  ← Query=tokens
        Returns T: [B, n_t, d]
        """
        B   = tokens.shape[0]
        ind = self.bimp_I.unsqueeze(0).expand(B, -1, -1)   # [B, m, d]
        S   = S_mab(ind,    tokens, mask_y=mask)            # [B, m, d]
        T   = T_mab(tokens, S)                              # [B, n_t, d]
        return T

    # ------------------------------------------------------------------
    # §2.2-A  Item→Bundle update
    # ------------------------------------------------------------------

    def _item_to_bundle(self, X_i_g, H_b_g, bundles_uniq):
        """
        Args:
            X_i_g      : [N_i, d] global item states
            H_b_g      : [N_b, d] global bundle states
            bundles_uniq: [U] unique bundle ids

        Returns:
            H_b_new    : [N_b, d] updated bundle states (only bundles_uniq rows changed)
        """
        U   = bundles_uniq.shape[0]
        d   = self.emb_size
        BC  = 1024   # bundle chunk for memory

        new_states = []
        for i in range(0, U, BC):
            ub_chunk = bundles_uniq[i: i + BC]    # [uc]
            uc       = ub_chunk.shape[0]

            items_c  = self.bundle_items[ub_chunk]          # [uc, n_t]
            valid_c  = (items_c >= 0)                       # [uc, n_t]  bool
            items_cs = items_c.clamp(min=0)

            # Z_{i,b} = X_i^(l-1) + W_pe * PE(i,b)
            Z_c   = X_i_g[items_cs]                         # [uc, n_t, d]
            pe_c  = self._compute_pe(items_cs, valid_c)     # [uc, n_t, d]
            Z_c   = Z_c + pe_c
            Z_c   = Z_c.masked_fill(~valid_c.unsqueeze(-1), 0.0)

            # WithinATT: V_b = MAB(Z, MAB(I, Z))
            V_b   = self._within_att(self.mab_i2b_within, self.mab_i2b_outer,
                                     Z_c, valid_c)          # [uc, n_t, d]

            # H_b^(l) = MAB(H_b^(l-1), V_b)   ← bundle as query
            H_prev = H_b_g[ub_chunk].unsqueeze(1)           # [uc, 1, d]
            H_new  = self.mab_i2b_main(H_prev, V_b,
                                        mask_y=valid_c)     # [uc, 1, d]
            new_states.append((ub_chunk, H_new.squeeze(1))) # [uc, d]

        # Scatter back
        H_b_new = H_b_g.clone()
        for idx, vals in new_states:
            H_b_new[idx] = vals
        return H_b_new

    # ------------------------------------------------------------------
    # §2.2-B  Bundle→Item update  (GPU-vectorized, no Python set / item loop)
    # ------------------------------------------------------------------

    def _bundle_to_item(self, X_i_loc, H_b_loc, H_b_init_full,
                        items_uniq, active_mask_local,
                        b_global2local, train):
        """
        GPU-vectorized B2I pass with extra-bundle context injection.

        Active bundles (b ∈ B_batch):  use current H_b_loc[local_idx]
        Extra  bundles (b ∈ N(i)\B_batch): use initial H_b_init_full[global_id]

        E_i = { H_b^(l+1)+PE(b,i) | b∈N_act(i) }
            ∪ { H_b^(0) +PE(b,i) | b∈N_extra(i) }
        """
        d = self.emb_size
        K       = self.max_incident_K if train else self.eval_incident_K
        K_extra = int(self.conf.get("K_extra", K))
        V = items_uniq.shape[0]

        # ------------------------------------------------------------------
        # 1. Expand CSR → flat (item, bundle_global) edges
        # ------------------------------------------------------------------
        ptr_s = self.item_ptr[items_uniq]
        ptr_e = self.item_ptr[items_uniq + 1]
        raw_deg = ptr_e - ptr_s

        if raw_deg.sum() == 0:
            return X_i_loc

        flat_item_local = torch.repeat_interleave(
            torch.arange(V, device=self.device), raw_deg)

        max_raw = int(raw_deg.max().item()) if raw_deg.numel() > 0 else 0
        if max_raw == 0:
            return X_i_loc

        cum = torch.zeros(V + 1, dtype=torch.long, device=self.device)
        cum[1:] = raw_deg.cumsum(0)
        flat_local_offset = (torch.arange(raw_deg.sum(), device=self.device)
                             - cum[flat_item_local])
        flat_abs     = ptr_s[flat_item_local] + flat_local_offset
        flat_bundles = self.item_adj[flat_abs]           # [total_edges] global bundle ids

        # ------------------------------------------------------------------
        # 2. Split into active / extra masks
        # ------------------------------------------------------------------
        act_mask_flat  = active_mask_local[flat_bundles]  # bool
        flat_b_local   = b_global2local[flat_bundles]     # local idx (-1 = inactive)

        # ---- active edges ----
        keep_act       = act_mask_flat
        fb_local_act   = flat_b_local[keep_act]
        fi_act         = flat_item_local[keep_act]

        # ---- extra edges (inactive bundles) ----
        keep_ext       = ~act_mask_flat
        fb_global_ext  = flat_bundles[keep_act.logical_not()]   # global bundle ids
        fi_ext         = flat_item_local[keep_ext]

        # ------------------------------------------------------------------
        # 3. Active branch: recount, build padded [A, max_deg] tensor
        # ------------------------------------------------------------------
        act_deg = torch.zeros(V, dtype=torch.long, device=self.device)
        if fi_act.numel() > 0:
            act_deg.scatter_add_(0, fi_act, torch.ones_like(fi_act))

        ext_deg = torch.zeros(V, dtype=torch.long, device=self.device)
        if fi_ext.numel() > 0:
            ext_deg.scatter_add_(0, fi_ext, torch.ones_like(fi_ext))

        has_neigh = (act_deg + ext_deg) > 0
        if not has_neigh.any():
            return X_i_loc


        act_item_idx = has_neigh.nonzero(as_tuple=True)[0]   # [A]
        A = act_item_idx.shape[0]

        item2comp = torch.full((V,), -1, dtype=torch.long, device=self.device)
        item2comp[act_item_idx] = torch.arange(A, device=self.device)

        # ---- build active padded tensor ----
        def _build_padded(fi, fb, size, K_cap, use_global=False):
            """
            fi: flat item local indices
            fb: flat bundle indices (local if not use_global, else global)
            Returns neigh_pad [A, K_cap], valid [A, K_cap]
            """
            if fi.numel() == 0:
                pad = torch.full((A, K_cap), -1, dtype=torch.long, device=self.device)
                return pad, pad >= 0

            comp = item2comp[fi]
            valid_edge = comp >= 0
            comp = comp[valid_edge]
            fb   = fb[valid_edge]

            deg_c = torch.zeros(A, dtype=torch.long, device=self.device)
            deg_c.scatter_add_(0, comp, torch.ones_like(comp))
            deg_capped = deg_c.clamp(max=K_cap)
            max_d = int(deg_capped.max().item()) if deg_capped.numel() > 0 else 0
            if max_d == 0:
                return (torch.full((A, 1), -1, dtype=torch.long, device=self.device),
                        torch.zeros(A, 1, dtype=torch.bool, device=self.device))

            N_e = comp.shape[0]
            gpos = torch.arange(N_e, device=self.device)
            gs   = torch.zeros(A, dtype=torch.long, device=self.device)
            gs.scatter_(0, comp.flip(0), gpos.flip(0))

            intra = gpos - gs[comp]

            if train:
                noise    = torch.rand(N_e, device=self.device)
                shuf_key = comp.float() * (K_cap + 2) + noise
                so       = shuf_key.argsort()
                comp     = comp[so]; fb = fb[so]
                gs2      = torch.zeros(A, dtype=torch.long, device=self.device)
                gs2.scatter_(0, comp.flip(0), gpos.flip(0))
                intra    = gpos - gs2[comp]

            keep = intra < K_cap
            comp = comp[keep]; fb = fb[keep]; intra = intra[keep]
            fb   = fb.clamp(0, size - 1)
            intra= intra.clamp(0, max_d - 1)

            pad = torch.full((A, max_d), -1, dtype=torch.long, device=self.device)
            pad[comp, intra] = fb
            return pad, pad >= 0

        # Active: local bundle indices → H_b_loc
        neigh_act, valid_act = _build_padded(
            fi_act, fb_local_act, H_b_loc.shape[0], K)

        # Extra: global bundle indices → H_b_init_full
        neigh_ext, valid_ext = _build_padded(
            fi_ext, fb_global_ext, H_b_init_full.shape[0], K_extra)

        # ------------------------------------------------------------------
        # 4. Build embedding tensors + PE, then concatenate
        # ------------------------------------------------------------------
        bundles_uniq_g = self._cur_bundles_uniq_global       # [U] global ids

        def _pe_emb(neigh_pad, valid_mask, global_ids_fn):
            """PE for a set of [A, n_k] neighbour slots."""
            n_k  = neigh_pad.shape[1]
            gids = global_ids_fn(neigh_pad)                  # [A, n_k] global ids
            bd   = self.bundle_degree[gids]                  # [A, n_k]

            def _rank_norm(deg, vmask):
                deg_f = deg.float().masked_fill(~vmask, -1.0)
                order = deg_f.argsort(dim=-1, descending=True)
                rank  = torch.zeros_like(deg_f)
                rank.scatter_(1, order,
                    torch.arange(n_k, dtype=torch.float, device=self.device)
                        .unsqueeze(0).expand(A, -1))
                n_val = vmask.sum(1, keepdim=True).float().clamp(min=2) - 1
                return (rank / n_val).clamp(0, 1).masked_fill(~vmask, 0.0)

            rk  = _rank_norm(bd, valid_mask)
            rev = (1.0 - rk).masked_fill(~valid_mask, 0.0)
            pe  = self.W_pe(torch.stack([rk, rk, rev], dim=-1))  # [A, n_k, d]
            return pe

        # Active embeddings
        pe_act  = _pe_emb(neigh_act, valid_act,
                          lambda p: bundles_uniq_g[p.clamp(0, bundles_uniq_g.shape[0]-1)])
        E_act   = H_b_loc[neigh_act.clamp(0, H_b_loc.shape[0]-1)] + pe_act
        E_act   = E_act.masked_fill(~valid_act.unsqueeze(-1), 0.0)

        # Extra embeddings (h_b^0, global ids)
        pe_ext  = _pe_emb(neigh_ext, valid_ext,
                          lambda p: p.clamp(0, H_b_init_full.shape[0]-1))
        E_ext   = H_b_init_full[neigh_ext.clamp(0, H_b_init_full.shape[0]-1)] + pe_ext
        E_ext   = E_ext.masked_fill(~valid_ext.unsqueeze(-1), 0.0)

        # Concatenate along token dim
        E_comb     = torch.cat([E_act, E_ext], dim=1)        # [A, K+K_extra, d]
        valid_comb = torch.cat([valid_act, valid_ext], dim=1) # [A, K+K_extra]

        # One-time logging (first call per forward)
        if getattr(self, '_log_extra_once', True):
            n_act_mean  = valid_act.sum(1).float().mean().item()
            n_ext_mean  = valid_ext.sum(1).float().mean().item()
            n_use_mean  = valid_comb.sum(1).float().mean().item()
            print(f"[B2I ctx] mean |N_act|={n_act_mean:.1f}  "
                  f"|N_extra|={n_ext_mean:.1f}  |N_use|={n_use_mean:.1f}")
            self._log_extra_once = False

        # ------------------------------------------------------------------
        # 5. WithinATT + MAB update
        # ------------------------------------------------------------------
        if not valid_comb.any():
            return X_i_loc

        G = self._within_att(self.mab_b2i_within, self.mab_b2i_outer,
                             E_comb, valid_comb)               # [A, K+K_extra, d]

        X_prev      = X_i_loc[act_item_idx].unsqueeze(1)      # [A, 1, d]
        X_new       = self.mab_b2i_main(X_prev, G,
                                        mask_y=valid_comb)     # [A, 1, d]
        X_i_loc_new = X_i_loc.clone()
        X_i_loc_new[act_item_idx] = X_new.squeeze(1)
        return X_i_loc_new

    # ------------------------------------------------------------------
    # §2 multi-layer bidirectional pass
    # ------------------------------------------------------------------

    def _bimp_forward(self, embs, bundles_uniq, items_uniq, train):
        """
        Run num_bimp_layers rounds of I→B then B→I  (local subgraph edition).

        Allocates X_i_loc [V, d] and H_b_loc [U, d] — only the active subset —
        instead of full [N_i, d] / [N_b, d] global tensors.

        Args:
            embs         : dict from get_embeddings
            bundles_uniq : [U]  unique bundle ids (global)
            items_uniq   : [V]  unique item ids that appear in bundles_uniq (global)
            train        : bool

        Returns:
            H_b_g      : [N_b, d]   global bundle states (only bundles_uniq rows updated)
            X_i_g      : [N_i, d]   global item states   (only items_uniq rows updated)
            V_b_tokens : [U, n_t, d]  item tokens for Stage2
            valid_V    : [U, n_t]     bool mask
        """
        U = bundles_uniq.shape[0]
        V = items_uniq.shape[0]
        d = self.emb_size
        self._log_extra_once = True   # log N_act/N_extra/N_use once per forward pass

        UI_i = embs["UI_items"]    # [N_i, d]
        BI_i = embs["BI_items"]    # [N_i, d]
        BI_b = embs["BI_bundles"]  # [N_b, d]

        # ------------------------------------------------------------------
        # Local subgraph: allocate only [V, d] and [U, d]
        # ------------------------------------------------------------------
        X_i_init_full = self._bimp_item_init(UI_i, BI_i)  # [N_i, d]  (needed for scatter-back)
        H_b_init_full = self._bimp_bundle_init(BI_b)       # [N_b, d]

        # Local slices
        X_i_loc = X_i_init_full[items_uniq].clone()        # [V, d]
        H_b_loc = H_b_init_full[bundles_uniq].clone()      # [U, d]

        # Ablation: capture pre-BIMP item tokens (used for Stage2 when flag=False)
        use_bimp_vb = self.conf.get("use_bimp_vb_for_stage2", True)
        if not use_bimp_vb:
            X_i_init_loc = X_i_loc.clone()                 # [V, d]  pre-BIMP snapshot

        # ------------------------------------------------------------------
        # active_mask [N_b] and b_global2local [N_b]  for GPU-vectorized B2I
        # ------------------------------------------------------------------
        active_mask = torch.zeros(self.num_bundles, dtype=torch.bool,
                                  device=self.device)
        active_mask[bundles_uniq] = True                    # [N_b]

        b_global2local = torch.full((self.num_bundles,), -1,
                                    dtype=torch.long, device=self.device)
        b_global2local[bundles_uniq] = torch.arange(U, device=self.device)  # [N_b]

        # Store as instance attrs so _bundle_to_item can read them
        self._cur_bundles_uniq_global = bundles_uniq       # [U] global ids

        # ------------------------------------------------------------------
        # Precompute loop-invariant index maps (outside layer loop)
        # ------------------------------------------------------------------
        i_global2local = torch.full((self.num_items,), -1,
                                    dtype=torch.long, device=self.device)
        i_global2local[items_uniq] = torch.arange(V, device=self.device)

        # bundle_items[bundles_uniq]: [U, n_t] global item ids (constant across layers)
        items_loc_2d  = self.bundle_items[bundles_uniq]           # [U, n_t]
        valid_loc     = (items_loc_2d >= 0)                       # [U, n_t]
        items_loc_cs  = items_loc_2d.clamp(min=0)
        items_loc_idx = i_global2local[items_loc_cs].clamp(min=0) # [U, n_t] local item idx

        # PE is also loop-invariant (depends only on item degrees, not layer states)
        pe_loc = self._compute_pe(items_loc_cs, valid_loc)        # [U, n_t, d]

        last_V_b_loc = None   # will hold the final-layer I→B WithinATT output

        for _ in range(self.num_bimp_layers):
            # ------ (A) I→B  (local) ------
            Z_loc   = X_i_loc[items_loc_idx]                          # [U, n_t, d]
            Z_loc   = (Z_loc + pe_loc).masked_fill(~valid_loc.unsqueeze(-1), 0.0)

            V_b_loc = self._within_att(self.mab_i2b_within, self.mab_i2b_outer,
                                       Z_loc, valid_loc)              # [U, n_t, d]
            last_V_b_loc = V_b_loc                                    # save for Stage2

            H_prev  = H_b_loc.unsqueeze(1)                            # [U, 1, d]
            H_new   = self.mab_i2b_main(H_prev, V_b_loc,
                                        mask_y=valid_loc)             # [U, 1, d]
            H_b_loc = H_new.squeeze(1)                                # [U, d]

            # ------ (B) B→I  (local, GPU-vectorized) ------
            if self.conf.get("use_b2i", True):
                X_i_loc = self._bundle_to_item(
                    X_i_loc, H_b_loc, H_b_init_full,
                    items_uniq, active_mask,
                    b_global2local, train)                             # [V, d]

        # ------------------------------------------------------------------
        # Stage2 bundle tokens: use the last I→B WithinATT output \tilde{Z}_b
        # Shape: [U, n_t, d] — same as before, padding already zeroed by masked_fill
        # ------------------------------------------------------------------
        V_b_tokens = last_V_b_loc.masked_fill(~valid_loc.unsqueeze(-1), 0.0)
        valid_V    = valid_loc                                         # [U, n_t]

        # ------------------------------------------------------------------
        # Scatter local states back to global tensors
        # ------------------------------------------------------------------
        H_b_g = H_b_init_full.clone()
        H_b_g[bundles_uniq] = H_b_loc

        X_i_g = X_i_init_full.clone()
        X_i_g[items_uniq] = X_i_loc

        return H_b_g, X_i_g, V_b_tokens, valid_V

    # ------------------------------------------------------------------
    # s_base: UI_u · mean(b_items_UI) + UB_u · UB_b
    # ------------------------------------------------------------------

    def _compute_sbase(self, UI_u_flat, UB_u_flat, UI_i, UB_b, flat_b):
        """
        UI_u_flat : [M, d]
        UB_u_flat : [M, d]
        UI_i      : [N_i, d]
        UB_b      : [N_b, d]
        flat_b    : [M]  bundle ids

        Returns s_base : [M]
        """
        items_2d  = self.bundle_items[flat_b]           # [M, n_t]
        valid_2d  = (items_2d >= 0)                     # [M, n_t]
        items_cs  = items_2d.clamp(min=0)

        item_embs = UI_i[items_cs]                      # [M, n_t, d]
        item_embs = item_embs.masked_fill(~valid_2d.unsqueeze(-1), 0.0)
        count     = valid_2d.sum(1, keepdim=True).float().clamp(min=1)  # [M,1]
        mean_emb  = item_embs.sum(1) / count            # [M, d]

        s_ui  = (UI_u_flat * mean_emb).sum(-1)          # [M]
        s_ub  = (UB_u_flat * UB_b[flat_b]).sum(-1)      # [M]
        return s_ui + s_ub

    # ------------------------------------------------------------------
    # §6 Stage2: user-Query softmax weighted sum
    # ------------------------------------------------------------------

    def _get_lambda(self):
        """Returns lam scalar. Fixed if 'lambda_weight' is in config, else learned."""
        fixed = self.conf.get("lambda_weight", None)
        if fixed is not None:
            return float(fixed)
        return torch.sigmoid(self.logit_lambda)

    def _stage2_score(self, q_flat, V_b, valid):
        """
        q_flat : [M, d]
        V_b    : [M, n_t, d]
        valid  : [M, n_t]  bool

        Returns s_new : [M]
        """
        d   = self.emb_size
        tau = self.bimp_tau

        # Ensure contiguous layout — bmm / Wk fail on non-contiguous strides
        V_b     = V_b.contiguous()
        q_proj  = self.Wq_s2(q_flat)                   # [M, d]
        k_proj  = self.Wk_s2(V_b)                      # [M, n_t, d]
        v_proj  = self.Wv_s2(V_b)                      # [M, n_t, d]

        # logits = (Wq·q_u) · (Wk·token) / sqrt(d)
        logits = torch.bmm(
            q_proj.unsqueeze(1).contiguous(),           # [M, 1, d]
            k_proj.transpose(1, 2).contiguous()         # [M, d, n_t]
        ).squeeze(1) / math.sqrt(d)                     # [M, n_t]

        logits = logits / tau
        logits = logits.masked_fill(~valid, float('-inf'))
        alpha  = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)  # [M, n_t]

        h_bu   = (alpha.unsqueeze(-1) * v_proj).sum(1)   # [M, d]
        s_new  = (q_proj * h_bu).sum(-1)                 # [M]
        return s_new

    # ------------------------------------------------------------------
    # Score computation  (training)
    # ------------------------------------------------------------------

    def compute_scores(self, embs, users, bundles, return_components=False):
        """
        bundles : [bs, n_b]  (col 0 = pos, rest = neg)
        """
        bs, n_b = bundles.shape
        M       = bs * n_b
        flat_b  = bundles.reshape(-1)                   # [M]
        d       = self.emb_size

        UI_u = embs["UI_users"]
        UB_u = embs["UB_users"]
        UI_i = embs["UI_items"]
        UB_b = embs["UB_bundles"]

        def _expand(v):                                 # [bs, d] → [M, d]
            return v[users].unsqueeze(1).expand(-1, n_b, -1).reshape(M, d)

        eu_UI = _expand(UI_u)                           # [M, d]
        eu_UB = _expand(UB_u)                           # [M, d]

        # ---- s_base ----
        s_base = self._compute_sbase(eu_UI, eu_UB, UI_i, UB_b, flat_b)  # [M]

        # ---- BIMP ----
        bundles_uniq, inv_idx = torch.unique(flat_b, return_inverse=True)
        # items_uniq: union of items in bundles_uniq
        items_2d  = self.bundle_items[bundles_uniq]
        valid_2d  = (items_2d >= 0)
        items_uniq = items_2d[valid_2d].unique()

        _, _, V_b_uniq, valid_uniq = self._bimp_forward(
            embs, bundles_uniq, items_uniq, train=True)   # [U, n_t, d]

        # Map back to [M]  — .contiguous() prevents non-contiguous strides into bmm
        V_b   = V_b_uniq[inv_idx].contiguous()          # [M, n_t, d]
        valid = valid_uniq[inv_idx].contiguous()         # [M, n_t]

        # ---- Stage2 ----
        q_u   = self._user_query(eu_UI, eu_UB)          # [M, d]
        s_new = self._stage2_score(q_u, V_b, valid)     # [M]

        # ---- Final: s = s_base + lambda * s_new ----
        lam    = self._get_lambda()
        scores = (s_base + lam * s_new).reshape(bs, n_b)

        if return_components:
            s_new_r = (lam * s_new).reshape(bs, n_b)
            zeros   = torch.zeros_like(scores)
            return scores, s_base.reshape(bs, n_b), s_new_r, zeros
        return scores

    # ------------------------------------------------------------------
    # Contrastive loss
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
    # Forward  (training)
    # ------------------------------------------------------------------

    def forward(self, batch, ED_drop=False):
        if ED_drop and self.conf.get("aug_type") == "ED":
            self._refresh_ed_graphs()
        users, bundles = batch[0].squeeze(1), batch[1]
        pos_bundles    = bundles[:, 0]

        embs     = self.get_embeddings(test=False)
        scores   = self.compute_scores(embs, users, bundles)
        bpr_loss = cal_bpr_loss(scores)

        if self.conf.get("c_lambda", 0.0) > 0:
            c_loss = self.cal_c_loss(embs, users, pos_bundles)
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, c_loss

    # ------------------------------------------------------------------
    # Inference  (evaluation)
    # ------------------------------------------------------------------

    def get_multi_modal_representations(self, test=True):
        return self.get_embeddings(test=test)

    def evaluate(self, embs, users):
        return self._score_all_bundles(embs, users, return_comps=False)

    def evaluate_with_components(self, embs, users):
        return self._score_all_bundles(embs, users, return_comps=True)

    _profile_called = False

    def _score_all_bundles(self, embs, users, return_comps=False):
        """
        Full-pass evaluation.

        Mode A (bimp_eval_single_pass=True, default):
          - Run _bimp_forward ONCE over all NB bundles → V_b_all [NB, n_t, d]
          - Stage2 in user-only chunks (no bundle inner loop)
          - O(1) BIMP calls vs O(NB/BC) before

        Mode B (bimp_eval_single_pass=False, legacy):
          - Bundle-chunk loop (original behaviour)
        """
        import time
        def _t():
            if torch.cuda.is_available(): torch.cuda.synchronize()
            return time.time()

        _do_profile = not DSS._profile_called
        DSS._profile_called = True
        t0 = _t()

        users = users.squeeze()
        if users.dim() == 0:
            users = users.unsqueeze(0)
        bs = users.shape[0]

        UI_u_all = embs["UI_users"][users]          # [bs, d]
        UB_u_all = embs["UB_users"][users]          # [bs, d]
        UI_i     = embs["UI_items"]                 # [N_i, d]
        UB_b     = embs["UB_bundles"]               # [N_b, d]

        NB, d = self.num_bundles, self.emb_size
        UC    = self.conf.get("eval_user_chunk", 16)
        lam   = self._get_lambda()

        scores = torch.zeros((bs, NB), device=self.device)

        # ----------------------------------------------------------------
        # s_base (vectorised, no loop):  UI_u · mean(b_items_UI) + UB_u · UB_b
        # ----------------------------------------------------------------
        bi_np       = self.bundle_items              # [NB, n_t]
        valid_all   = (bi_np >= 0)                   # [NB, n_t]
        items_all_s = bi_np.clamp(min=0)

        BC_b = 2048
        mean_items_all = torch.zeros(NB, d, device=self.device)
        for b0 in range(0, NB, BC_b):
            b1       = min(b0 + BC_b, NB)
            items_bc = items_all_s[b0:b1]
            valid_bc = valid_all[b0:b1]
            emb_bc   = UI_i[items_bc]
            emb_bc   = emb_bc.masked_fill(~valid_bc.unsqueeze(-1), 0.0)
            cnt_bc   = valid_bc.sum(-1, keepdim=True).float().clamp(min=1)
            mean_items_all[b0:b1] = emb_bc.sum(1) / cnt_bc            # [bc, d]

        for u0 in range(0, bs, UC):
            u1     = min(u0 + UC, bs)
            eu_UI  = UI_u_all[u0:u1]
            eu_UB  = UB_u_all[u0:u1]
            scores[u0:u1] = (torch.mm(eu_UI, mean_items_all.t())
                             + torch.mm(eu_UB, UB_b.t()))

        t1 = _t()

        # ----------------------------------------------------------------
        # BIMP + Stage2
        # ----------------------------------------------------------------
        single_pass = self.conf.get("bimp_eval_single_pass", True)

        all_bundles  = torch.arange(NB, device=self.device)
        all_items_2d = self.bundle_items                               # [NB, n_t]
        all_valid_2d = (all_items_2d >= 0)
        all_items_uniq = all_items_2d[all_valid_2d].unique()          # [V_all]

        if single_pass:
            # ------ Mode A: one BIMP call for all bundles ------
            _, _, V_b_all, valid_b_all = self._bimp_forward(
                embs, all_bundles, all_items_uniq, train=False)        # [NB, n_t, d]

            n_t  = V_b_all.shape[1]
            tau  = self.bimp_tau

            # --- Precompute k_proj / v_proj ONCE for all bundles ---
            # (avoid recomputing NB Linear forwards per user chunk)
            V_b_c = V_b_all.contiguous()                               # [NB, n_t, d]
            k_all = self.Wk_s2(V_b_c)                                 # [NB, n_t, d]
            v_all = self.Wv_s2(V_b_c)                                 # [NB, n_t, d]

            # valid mask: [NB, n_t]  (True = real token)
            # Pre-fill -inf where invalid so we can simply add to logits
            INF = torch.full((NB, n_t), float('-inf'),
                             device=self.device, dtype=k_all.dtype)
            INF[valid_b_all] = 0.0                                     # 0 for valid slots

            for u0 in range(0, bs, UC):
                u1    = min(u0 + UC, bs)
                eu_UI = UI_u_all[u0:u1]                                # [uc, d]
                eu_UB = UB_u_all[u0:u1]                                # [uc, d]
                q_uc  = self._user_query(eu_UI, eu_UB)                 # [uc, d]
                q_proj = self.Wq_s2(q_uc)                             # [uc, d]

                # logits: [uc, NB, n_t]  (no expand of V tensors needed)
                logits = torch.einsum('ud,bnd->ubn', q_proj, k_all)
                logits = logits / (tau * (d ** 0.5))
                logits = logits + INF.unsqueeze(0)                     # broadcast mask

                alpha  = torch.nan_to_num(
                    F.softmax(logits, dim=-1), nan=0.0)                # [uc, NB, n_t]

                # weighted sum of v_all tokens
                h_bu   = torch.einsum('ubn,bnd->ubd', alpha, v_all)   # [uc, NB, d]

                # score = q · h_bu  (dot product per (u, b) pair)
                s_new = (q_proj.unsqueeze(1) * h_bu).sum(-1)          # [uc, NB]
                scores[u0:u1] = scores[u0:u1] + lam * s_new

        else:
            # ------ Mode B: chunked (legacy) ------
            BC_bimp = self.conf.get("eval_bundle_chunk", 512)

            for b0 in range(0, NB, BC_bimp):
                b1            = min(b0 + BC_bimp, NB)
                chunk_bundles = torch.arange(b0, b1, device=self.device)
                bc            = chunk_bundles.shape[0]

                items_2d_c    = self.bundle_items[chunk_bundles]
                valid_2d_c    = (items_2d_c >= 0)
                items_uniq_c  = items_2d_c[valid_2d_c].unique()

                _, _, V_b_c, valid_c = self._bimp_forward(
                    embs, chunk_bundles, items_uniq_c, train=False)    # [bc, n_t, d]

                n_t = V_b_c.shape[1]

                for u0 in range(0, bs, UC):
                    u1    = min(u0 + UC, bs)
                    uc    = u1 - u0
                    eu_UI = UI_u_all[u0:u1]
                    eu_UB = UB_u_all[u0:u1]
                    q_uc  = self._user_query(eu_UI, eu_UB)             # [uc, d]

                    q_exp   = q_uc.unsqueeze(1).expand(-1, bc, -1).reshape(uc * bc, d)
                    V_exp   = V_b_c.unsqueeze(0).expand(uc, -1, -1, -1).reshape(uc * bc, n_t, d)
                    val_exp = valid_c.unsqueeze(0).expand(uc, -1, -1).reshape(uc * bc, n_t)

                    s_new_flat = self._stage2_score(q_exp, V_exp, val_exp)
                    scores[u0:u1, b0:b1] = (scores[u0:u1, b0:b1]
                                            + lam * s_new_flat.reshape(uc, bc))

        t2 = _t()

        if _do_profile:
            mode_str = "single-pass" if single_pass else "chunked"
            print(f"[PROFILE _score_all_bundles] mode={mode_str}  "
                  f"sbase={t1-t0:.2f}s  bimp+stage2={t2-t1:.2f}s  total={t2-t0:.2f}s")

        if return_comps:
            zeros = torch.zeros_like(scores)
            return scores, scores, zeros, zeros
        return scores

