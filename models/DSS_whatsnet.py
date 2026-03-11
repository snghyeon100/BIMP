#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS  —  WHATsNet-style Bundle↔Item Bidirectional Message Passing

Score(u, b) = s_base(u,b) + λ * s_new(u,b)

  s_base(u,b) = UI_u · mean(b_items_UI) + UB_u · UB_b

  Bidirectional layers (L rounds):
    Init:
      X_i^(0) = LN(W_ui·UI_i + W_bi·BI_i)          # item state
      H_b^(0) = LN(W_biB·BI_b + W_ubB·UB_b)         # bundle state (BI + UB)

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
import dgl


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

        self.W_v = nn.Linear(d, d, bias=False)
        self.ln1 = nn.LayerNorm(d)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.scale = math.sqrt(self.hd)
        
        # Restored 2-layer FFN
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, X, Y, mask_y=None):
        B, n_q, d = X.shape
        n_k       = Y.shape[1]
        h, hd     = self.h, self.hd

        Q = X.view(B, n_q, h, hd).transpose(1, 2)             # [B, h, n_q, hd]
        K = Y.view(B, n_k, h, hd).transpose(1, 2)             # [B, h, n_k, hd]
        V = self.W_v(Y).view(B, n_k, h, hd).transpose(1, 2)   # [B, h, n_k, hd]

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale   # [B, h, n_q, n_k]
        if mask_y is not None:
            # mask_y: [B, n_k] → [B, 1, 1, n_k]
            attn = attn.masked_fill(~mask_y.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = torch.nan_to_num(F.softmax(attn, dim=-1), nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, V)                                  # [B, h, n_q, hd]
        out = out.transpose(1, 2).contiguous().view(B, n_q, d)

        H = self.ln1(X + out)           # residual + layer-norm
        
        # Restored 2-layer FFN
        out_ffn = self.fc2(F.relu(self.fc1(H)))
        out_ffn = self.ln2(H + out_ffn)
        
        return out_ffn


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
        # §1.2 Bundle initial state: H_b^(0) = LN(W_biB·BI_b + W_ubB·UB_b)
        # ------------------------------------------------------------------
        self.W_biB          = nn.Linear(d, d, bias=False)
        self.W_ubB          = nn.Linear(d, d, bias=False)
        self.ln_bundle_init = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §1.3 User query: q_u = LN(W_uiq·UI_u + W_ubq·UB_u)
        # ------------------------------------------------------------------
        self.W_uiq      = nn.Linear(d, d, bias=False)
        self.W_ubq      = nn.Linear(d, d, bias=False)
        self.ln_user_q  = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §5 Positional Encoding: W_pe: Linear(3 → d)
        #    channels: rank_ui_deg, rank_bi_deg, rank_ub_deg
        # ------------------------------------------------------------------
        self.W_pe = nn.Linear(3, d, bias=False)

        # Precompute item degree vectors (UI and BI) for PE
        # These are computed post-graph-build so we store them as buffers
        # (filled in _build_graphs)
        self.register_buffer("_ui_item_deg", torch.zeros(self.num_items))
        self.register_buffer("_bi_item_deg", torch.zeros(self.num_items))

        # ------------------------------------------------------------------
        # Bidirectional Fast Graph Attention (Replacing Mailbox MAB)
        # ------------------------------------------------------------------
        h = self.num_heads
        dh = d // h
        
        self.W_q_i2b = nn.Linear(d, d, bias=False)
        self.W_k_i2b = nn.Linear(d, d, bias=False)
        self.W_v_i2b = nn.Linear(d, d, bias=False)
        self.W_o_i2b = nn.Linear(d, d, bias=False)
        self.norm_i2b = nn.LayerNorm(d)
        
        self.W_q_b2i = nn.Linear(d, d, bias=False)
        self.W_k_b2i = nn.Linear(d, d, bias=False)
        self.W_v_b2i = nn.Linear(d, d, bias=False)
        self.W_o_b2i = nn.Linear(d, d, bias=False)
        self.norm_b2i = nn.LayerNorm(d)

        # ------------------------------------------------------------------
        # §1.4 Stage 2 (user queries bundle tokens)
        # Note: Projections Wq_s2, Wk_s2, Wv_s2 have been removed for
        # Light-Attention parameter reduction to combat overfitting.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # §7 Baseline residual scalar λ
        #    Fixed to 0.5 as requested
        # ------------------------------------------------------------------
        self.lam_weight = 0.5

        # Config compatibility
        self.alpha_base = float(conf.get("alpha_base", 1.0))

        self._build_graphs()
        self._build_item_bundles()
        self._build_dgl_graphs()

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------


    def _build_dgl_graphs(self):
        bi = self.bundle_items
        NB, n_t = bi.shape
        b_idx = torch.arange(NB).unsqueeze(1).expand(NB, n_t)
        valid = bi >= 0
        
        # [수정] 모든 인덱스 데이터를 CPU에서 준비 (DGL 그래프 생성 시 장치 통일을 위함)
        v_b = b_idx[valid].cpu().long()
        v_i = bi[valid].cpu().long()

        # 1. CPU에서 그래프 생성
        self.g_b2i = dgl.heterograph({
            ('bundle', 'contains', 'item'): (v_b, v_i),
        }, num_nodes_dict={'bundle': self.num_bundles, 'item': self.num_items})

        self.g_i2b = dgl.heterograph({
            ('item', 'in', 'bundle'): (v_i, v_b)
        }, num_nodes_dict={'item': self.num_items, 'bundle': self.num_bundles})
        
        # 2. 그래프를 통째로 모델 장치(GPU)로 이동
        self.g_b2i = self.g_b2i.to(self.device)
        self.g_i2b = self.g_i2b.to(self.device)
        
        # 연산을 위해 valid 마스크도 GPU로 이동
        valid_device = valid.to(self.device)

        def rank_norm(deg, valid_mask):
            # deg와 valid_mask의 장치를 맞춤
            deg_f = deg.float().masked_fill(~valid_mask, -1.0)
            order = deg_f.argsort(dim=-1, descending=True)
            rank = torch.zeros_like(deg_f)
            # torch.arange 생성 시 device를 명시하여 불필요한 복사 방지
            rank.scatter_(1, order, torch.arange(n_t, dtype=torch.float, device=self.device).unsqueeze(0).expand(NB, -1))
            n_valid = valid_mask.sum(1, keepdim=True).float().clamp(min=2) - 1
            rank_n = (rank / n_valid).clamp(0.0, 1.0)
            return rank_n.masked_fill(~valid_mask, 0.0)

        # 3. Positional Encoding 데이터 준비 (GPU 상에서 연산)
        bi_cs = bi.clamp(min=0).to(self.device)
        ui_deg = self._ui_item_deg[bi_cs]
        bi_deg = self._bi_item_deg[bi_cs]
        
        rk_ui = rank_norm(ui_deg, valid_device)
        rk_bi = rank_norm(bi_deg, valid_device)
        
        # Globally normalize Bundle Popularity (UB Degree)
        max_ub = self._ub_bundle_deg.max().clamp(min=1.0)
        rk_ub  = (self._ub_bundle_deg / max_ub).unsqueeze(1).expand(NB, n_t)
        rk_ub  = rk_ub.masked_fill(~valid_device, 0.0)
        
        pe_raw = torch.stack([rk_ui, rk_bi, rk_ub], dim=-1) # [NB, n_t, 3]
        pe_valid = pe_raw[valid_device] # [E, 3]
        
        # 4. GPU에 있는 그래프 데이터에 PE 값 할당
        self.g_i2b.edges['in'].data['pe_raw'] = pe_valid
        self.g_b2i.edges['contains'].data['pe_raw'] = pe_valid

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

        # Bundle degree for PE
        ub_b_deg = np.array(self.ub_graph.sum(axis=0)).ravel().astype(np.float32)  # [N_b]
        self.register_buffer("_ub_bundle_deg", torch.tensor(ub_b_deg, dtype=torch.float, device=self.device))

        # User mixing ratio (beta_u)
        ui_u_deg = np.array(self.ui_graph.sum(axis=1)).ravel().astype(np.float32)  # [N_u]
        ub_u_deg = np.array(self.ub_graph.sum(axis=1)).ravel().astype(np.float32)  # [N_u]
        user_beta = ui_u_deg / (ui_u_deg + ub_u_deg + 1e-8)
        self.register_buffer("user_beta", torch.tensor(user_beta, dtype=torch.float, device=self.device))

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

    def _bimp_bundle_init(self, BI_b, UB_b):
        """H_b^(0) = LN(W_biB·BI_b + W_ubB·UB_b)  →  [N_b, d]"""
        return self.ln_bundle_init(self.W_biB(BI_b) + self.W_ubB(UB_b))

    def _user_query(self, UI_u, UB_u, users_ids):
        """q_u = LN(beta_u*W_uiq·UI_u + (1-beta_u)*W_ubq·UB_u)"""
        bu = self.user_beta[users_ids].unsqueeze(-1)  # shape depends on users_ids
        return self.ln_user_q(bu * self.W_uiq(UI_u) + (1.0 - bu) * self.W_ubq(UB_u))

    # ------------------------------------------------------------------
    # §5  Positional Encoding
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Fast DGL Edge-Level Attention
    # ------------------------------------------------------------------

    def _edge_attn_i2b(self, edges):
        pe = self.W_pe(edges.data['pe_raw'])
        z  = edges.src['h'] + pe
        
        h = self.num_heads
        dh = self.emb_size // h
        
        K = self.W_k_i2b(z).view(-1, h, dh)
        V = self.W_v_i2b(z).view(-1, h, dh)
        
        # dot product
        score = (edges.dst['q'] * K).sum(-1, keepdim=True) / (dh ** 0.5)
        return {'score': score, 'v': V}

    def _msg_attn_i2b(self, edges):
        return {'m': edges.data['v'] * edges.data['alpha']}

    def _edge_attn_b2i(self, edges):
        pe = self.W_pe(edges.data['pe_raw'])
        g  = edges.src['h'] + pe
        
        h = self.num_heads
        dh = self.emb_size // h
        
        K = self.W_k_b2i(g).view(-1, h, dh)
        V = self.W_v_b2i(g).view(-1, h, dh)
        
        score = (edges.dst['q'] * K).sum(-1, keepdim=True) / (dh ** 0.5)
        return {'score': score, 'v': V}

    def _msg_attn_b2i(self, edges):
        return {'m': edges.data['v'] * edges.data['alpha']}

    # ------------------------------------------------------------------
    # §2 multi-layer bidirectional pass
    # ------------------------------------------------------------------

    def _bimp_forward(self, embs, bundles_uniq, items_uniq, train):
        use_bimp_vb = self.conf.get("use_bimp_vb_for_stage2", True)

        X_i_g = self._bimp_item_init(embs["UI_items"], embs["BI_items"])
        H_b_g = self._bimp_bundle_init(embs["BI_bundles"], embs["UB_bundles"])
        
        if not use_bimp_vb:
            X_i_init_full = X_i_g.clone()

        sg_i2b = dgl.node_subgraph(self.g_i2b, {'bundle': bundles_uniq, 'item': items_uniq})
        sg_b2i = dgl.in_subgraph(self.g_b2i, {'item': items_uniq})

        for _ in range(self.num_bimp_layers):
            # (A) I→B
            sg_i2b.nodes['item'].data['h'] = X_i_g[sg_i2b.nodes('item')]
            H_b_curr = H_b_g[sg_i2b.nodes('bundle')]
            
            # Bundle acts as Query
            Q_i2b = self.W_q_i2b(H_b_curr).view(-1, self.num_heads, self.emb_size // self.num_heads)
            sg_i2b.nodes['bundle'].data['q'] = Q_i2b
            
            # Apply edges: compute scores & V from Item -> Bundle
            import dgl.function as fn
            import dgl.ops as ops
            sg_i2b.apply_edges(self._edge_attn_i2b, etype='in')
            sg_i2b.edges['in'].data['alpha'] = ops.edge_softmax(sg_i2b, sg_i2b.edges['in'].data['score'])
            sg_i2b.update_all(self._msg_attn_i2b, fn.sum('m', 'attn_out'), etype='in')
            
            attn_out_i2b = sg_i2b.nodes['bundle'].data['attn_out'].view(-1, self.emb_size)
            H_b_new = self.norm_i2b(H_b_curr + self.W_o_i2b(attn_out_i2b))
            
            sg_i2b.nodes['bundle'].data['h_new'] = H_b_new
            H_b_g[sg_i2b.nodes('bundle')] = H_b_new

            # (B) B→I
            if self.conf.get("use_b2i", True):
                sg_b2i.nodes['bundle'].data['h'] = H_b_g[sg_b2i.nodes('bundle')]
                X_i_curr = X_i_g[sg_b2i.nodes('item')]
                
                Q_b2i = self.W_q_b2i(X_i_curr).view(-1, self.num_heads, self.emb_size // self.num_heads)
                sg_b2i.nodes['item'].data['q'] = Q_b2i
                
                sg_b2i.apply_edges(self._edge_attn_b2i, etype='contains')
                sg_b2i.edges['contains'].data['alpha'] = ops.edge_softmax(sg_b2i, sg_b2i.edges['contains'].data['score'])
                sg_b2i.update_all(self._msg_attn_b2i, fn.sum('m', 'attn_out'), etype='contains')
                
                attn_out_b2i = sg_b2i.nodes['item'].data['attn_out'].view(-1, self.emb_size)
                X_i_new = self.norm_b2i(X_i_curr + self.W_o_b2i(attn_out_b2i))
                
                sg_b2i.nodes['item'].data['h_new'] = X_i_new
                X_i_g[sg_b2i.nodes('item')] = X_i_new

        items_loc_2d  = self.bundle_items[bundles_uniq]
        valid_loc     = (items_loc_2d >= 0)
        items_loc_cs  = items_loc_2d.clamp(min=0)

        if use_bimp_vb:
            Z_loc = X_i_g[items_loc_cs]
            Z_loc = Z_loc.masked_fill(~valid_loc.unsqueeze(-1), 0.0)
            
            U = Z_loc.shape[0]
            chunk_size = 64
            V_b_raw_list = []
            for i in range(0, U, chunk_size):
                end_idx = min(i + chunk_size, U)
                z_chunk = Z_loc[i:end_idx]
                mask_chunk = valid_loc[i:end_idx]
                vb_chunk = self._within_att(self.mab_i2b_within, self.mab_i2b_outer, z_chunk, mask=mask_chunk)
                V_b_raw_list.append(vb_chunk)
            V_b_raw = torch.cat(V_b_raw_list, dim=0)
        else:
            V_b_raw = X_i_init_full[items_loc_cs]

        V_b_tokens = V_b_raw.masked_fill(~valid_loc.unsqueeze(-1), 0.0)
        
        return H_b_g, X_i_g, V_b_tokens, valid_loc

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
        """Returns fixed lam scalar (0.5)."""
        return self.lam_weight

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
        # logits = (q_u) · (token) / sqrt(d)
        logits = torch.bmm(
            q_flat.unsqueeze(1).contiguous(),           # [M, 1, d]
            V_b.transpose(1, 2).contiguous()            # [M, d, n_t]
        ).squeeze(1) / math.sqrt(d)                     # [M, n_t]

        logits = logits / tau
        logits = logits.masked_fill(~valid, float('-inf'))
        alpha  = torch.nan_to_num(F.softmax(logits, dim=-1), nan=0.0)  # [M, n_t]
        self.last_alpha = alpha.detach() # Save for statistical analysis

        h_bu   = (alpha.unsqueeze(-1) * V_b).sum(1)      # [M, d] - direct mix of V_b
        s_new  = (q_flat * h_bu).sum(-1)                 # [M]    - direct dot product
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
        users_flat = users.unsqueeze(1).expand(-1, n_b).reshape(M)
        q_u   = self._user_query(eu_UI, eu_UB, users_flat)      # [M, d]
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
        users, bundles = batch[0].squeeze(1), batch[1]
        
        if ED_drop and self.conf.get("aug_type") == "ED":
            self._refresh_ed_graphs()
            
        embs     = self.get_embeddings(test=False)
        scores   = self.compute_scores(embs, users, bundles)
        pos  = scores[:, 0]
        negs = scores[:, 1:]
        # 벡터화된 BPR loss (동일 연산 구조 통일)
        bpr_loss  = F.softplus(negs - pos.unsqueeze(1)).mean()

        if self.conf.get("c_lambda", 0.0) > 0:
            pos_bundles = bundles[:, 0]
            c_loss = self.cal_c_loss(embs, users.view(-1), pos_bundles)
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        # L2 reg는 optimizer weight_decay에서 처리되므로 여기에서는 0 반환
        reg_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, c_loss, reg_loss

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
            k_all = V_b_c                                              # [NB, n_t, d]
            v_all = V_b_c                                              # [NB, n_t, d]

            # valid mask: [NB, n_t]  (True = real token)
            # Pre-fill -inf where invalid so we can simply add to logits
            INF = torch.full((NB, n_t), float('-inf'),
                             device=self.device, dtype=k_all.dtype)
            INF[valid_b_all] = 0.0                                     # 0 for valid slots

            for u0 in range(0, bs, UC):
                u1    = min(u0 + UC, bs)
                eu_UI = UI_u_all[u0:u1]                                # [uc, d]
                eu_UB = UB_u_all[u0:u1]                                # [uc, d]
                users_uc = users[u0:u1]
                q_uc  = self._user_query(eu_UI, eu_UB, users_uc)       # [uc, d]
                q_proj = q_uc                                          # [uc, d]

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
                    users_uc = users[u0:u1]
                    q_uc  = self._user_query(eu_UI, eu_UB, users_uc)             # [uc, d]

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

