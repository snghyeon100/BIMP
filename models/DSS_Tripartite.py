#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSS_Tripartite.py  —  Coupled Heterogeneous GCN (Tripartite Circulation).

Instead of isolating UI, BI, and UB propagation into separate matrices,
we fuse them into a single massive graph: [N_u + N_b + N_i].
Users, Bundles, and Items propagate concurrently in a single universe.

Features:
- Zero Memory Blowup (No Attention Tensors, pure SpMM)
- Extreme Sparsity Resistant (Information flows naturally via U->B->I in 2-hops) 
- Sub-millisecond evaluation
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
# DSS_Tripartite Model
# ---------------------------------------------------------------------------

class DSS_Tripartite(nn.Module):

    def __init__(self, conf, raw_graph, bundle_info=None):
        super().__init__()
        self.conf        = conf
        self.device      = conf["device"]
        self.emb_size    = conf["embedding_size"]
        self.num_users   = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items   = conf["num_items"]
        self.num_layers  = conf["num_layers"]

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        # Use average of configured dropouts for the giant graph
        self.UI_eps = float(conf.get("UI_ratio", 0.0))
        self.UB_eps = float(conf.get("UB_ratio", 0.0))
        self.BI_eps = float(conf.get("BI_ratio", 0.0))
        self.drop_ratio = (self.UI_eps + self.UB_eps + self.BI_eps) / 3.0

        # Bundle-level buffers (kept for BPR Reg Loss indexing)
        if bundle_info is not None:
            self.register_buffer("bundle_items", bundle_info["bundle_items"])
            self.register_buffer("bundle_size",  bundle_info["bundle_size"])
        else:
            NB = self.num_bundles
            self.register_buffer("bundle_items", torch.full((NB, 1), -1, dtype=torch.long))
            self.register_buffer("bundle_size",  torch.ones(NB, dtype=torch.long))

        # Core Embeddings
        d = self.emb_size
        self.users_feature   = nn.Parameter(torch.FloatTensor(self.num_users,   d))
        self.items_feature   = nn.Parameter(torch.FloatTensor(self.num_items,   d))
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, d))
        for p in [self.users_feature, self.items_feature, self.bundles_feature]:
            nn.init.xavier_normal_(p)

        # ------------------------------------------------------------------
        # Positional Encoding Setup
        # W_pe: Linear(3 -> d) channels: rank_ui_deg, rank_bi_deg, rev_rank_ui_deg
        # ------------------------------------------------------------------
        self.W_pe = nn.Linear(3, d, bias=False)
        self.register_buffer("_ui_item_deg", torch.zeros(self.num_items))
        self.register_buffer("_bi_item_deg", torch.zeros(self.num_items))

        # Build Massive Propagation Graph
        self._build_graphs()

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def _build_tri_matrix(self):
        """
        Builds the unified adjacency matrix [N, N] where N = N_u + N_b + N_i.
        Layout:
        [ 0    UB   UI ]
        [ BU   0    BI ]
        [ IU   IB   0  ]
        """
        n_u, n_b, n_i = self.num_users, self.num_bundles, self.num_items
        z_u = sp.csr_matrix((n_u, n_u))
        z_b = sp.csr_matrix((n_b, n_b))
        z_i = sp.csr_matrix((n_i, n_i))
        ub, ui, bi = self.ub_graph, self.ui_graph, self.bi_graph
        
        return sp.bmat([
            [z_u,  ub,   ui ],
            [ub.T, z_b,  bi ],
            [ui.T, bi.T, z_i]
        ]).tocsr()

    def _make_prop_graph(self, prop_matrix, dropout=0.0):
        if dropout > 0 and self.conf.get("aug_type") == "ED":
            coo  = prop_matrix.tocoo()
            vals = _np_edge_dropout(coo.data, dropout)
            prop = sp.coo_matrix((vals, (coo.row, coo.col)), shape=coo.shape).tocsr()
        else:
            prop = prop_matrix.tocsr()
        return _to_tensor(_laplace_transform(prop), self.device)

    def _build_graphs(self):
        self.tri_matrix    = self._build_tri_matrix()
        self.Tri_graph_ori = self._make_prop_graph(self.tri_matrix, dropout=0.0)
        self.Tri_graph     = self._make_prop_graph(self.tri_matrix, dropout=self.drop_ratio)
        
        # Precompute item degrees for PE
        self._ui_item_deg = torch.tensor(
            np.array(self.ui_graph.sum(axis=0)).ravel(), dtype=torch.float, device=self.device)
        self._bi_item_deg = torch.tensor(
            np.array(self.bi_graph.sum(axis=0)).ravel(), dtype=torch.float, device=self.device)

    def _refresh_ed_graphs(self):
        self.Tri_graph = self._make_prop_graph(self.tri_matrix, dropout=self.drop_ratio)

    # ------------------------------------------------------------------
    # Unified LightGCN propagation
    # ------------------------------------------------------------------

    def _get_item_pe(self):
        """
        Compute Positional Encoding (PE) for all valid items in all bundles.
        Instead of modifying individual bundles, we calculate the average PE 
        for each item across all bundles it belongs to, and add it directly 
        to the global item embedding before propagation.
        """
        B_tot, n_t = self.bundle_items.shape
        items_2d   = self.bundle_items.clamp(min=0)  # [B_tot, n_t]
        valid      = (self.bundle_items >= 0)        # [B_tot, n_t]
        
        ui_deg = self._ui_item_deg[items_2d]   # [B_tot, n_t]
        bi_deg = self._bi_item_deg[items_2d]   # [B_tot, n_t]
        
        def rank_norm(deg, valid_mask):
            deg_f = deg.float().masked_fill(~valid_mask, -1.0)
            order = deg_f.argsort(dim=-1, descending=True)
            rank  = torch.zeros_like(deg_f)
            rank.scatter_(1, order, torch.arange(n_t, dtype=torch.float,
                                                 device=self.device).unsqueeze(0).expand(B_tot, -1))
            n_valid = valid_mask.sum(1, keepdim=True).float().clamp(min=2) - 1   
            rank_n  = (rank / n_valid).clamp(0.0, 1.0)
            return rank_n.masked_fill(~valid_mask, 0.0)

        rk_ui  = rank_norm(ui_deg, valid)           
        rk_bi  = rank_norm(bi_deg, valid)
        rev_ui = (1.0 - rk_ui).masked_fill(~valid, 0.0)

        # [B_tot, n_t, 3] -> [B_tot, n_t, d]
        pe_raw = torch.stack([rk_ui, rk_bi, rev_ui], dim=-1)      
        pe_emb = self.W_pe(pe_raw)                                   
        
        # Aggregate (Mean) PE back to global items
        pe_emb = pe_emb.masked_fill(~valid.unsqueeze(-1), 0.0)
        
        # We scatter_add the PE from bundles back to global items
        # To get the mean, we divide by the number of bundles containing the item (bi degree)
        global_pe = torch.zeros(self.num_items, self.emb_size, device=self.device)
        items_flat = items_2d[valid]         # [num_edges]
        pe_flat    = pe_emb[valid]           # [num_edges, d]
        
        global_pe.index_add_(0, items_flat, pe_flat)
        
        # Divide by node degree to get mean PE per item
        deg_divisor = self._bi_item_deg.unsqueeze(1).clamp(min=1)
        return global_pe / deg_divisor

    def get_embeddings(self, test=False):
        g = self.Tri_graph_ori if test else self.Tri_graph
        
        # Inject PE directly into the base item features before propagation
        global_item_pe = self._get_item_pe()
        items_with_pe  = self.items_feature + global_item_pe
        
        # [N, d]
        feat = torch.cat([self.users_feature, self.bundles_feature, items_with_pe], dim=0)
        
        all_f = [feat]
        for _ in range(self.num_layers):
            feat = torch.spmm(g, feat)
            if not test and self.conf.get("aug_type") == "Noise" and self.drop_ratio > 0:
                noise = torch.rand_like(feat)
                feat  = feat + torch.sign(feat) * F.normalize(noise, dim=-1) * self.drop_ratio
            all_f.append(F.normalize(feat, p=2, dim=1))
            
        coef = torch.ones(1, len(all_f), 1, device=self.device) / len(all_f)
        agg  = (torch.stack(all_f, dim=1) * coef).sum(dim=1)
        
        # Split back
        n_u, n_b = self.num_users, self.num_bundles
        u_emb = agg[:n_u]
        b_emb = agg[n_u : n_u+n_b]
        i_emb = agg[n_u+n_b :]

        return {
            "UI_users":   u_emb,       
            "UB_users":   u_emb,       
            "BI_bundles": b_emb,       
            "UB_bundles": b_emb,       
            "UI_items":   i_emb,       
            "BI_items":   i_emb,       
            "bundle_emb": b_emb, # Unified representation natively captures items inside it     
        }

    def get_multi_modal_representations(self, test=False):
        return self.get_embeddings(test=test)

    # ------------------------------------------------------------------
    # Score Computation
    # Since views are unified, User and Bundle naturally contain all topology.
    # ------------------------------------------------------------------

    def compute_scores(self, embs, users, bundles):
        """bundles: [bs, 1+neg] → [bs, 1+neg]"""
        bs, n_b = bundles.shape
        flat_b  = bundles.reshape(-1)          # [M]
        
        # We can just do a direct dot-product of unified representations
        u_emb = embs["UI_users"][users]        # [bs, d]
        b_emb = embs["UB_bundles"][flat_b]     # [M, d]
        
        # Broadcast user to match flattened bundles
        u_emb_expanded = u_emb.unsqueeze(1).expand(-1, n_b, -1).reshape(-1, self.emb_size) # [M, d]
        
        scores = (u_emb_expanded * b_emb).sum(-1)  # [M]
        return scores.reshape(bs, n_b)

    # ------------------------------------------------------------------
    # Loss: BPR only. No CD Loss Needed!
    # Because there's only a single View, Contrastive Loss is meaningless.
    # ------------------------------------------------------------------

    def get_loss(self, embs, users, bundles, ED_drop=False):
        if ED_drop:
            self._refresh_ed_graphs()
            
        scores = self.compute_scores(embs, users, bundles)
        pos  = scores[:, 0]
        negs = scores[:, 1:]
        
        bpr  = sum(F.softplus(negs[:, i] - pos).mean()
                   for i in range(negs.shape[1])) / negs.shape[1]
                   
        c_loss = torch.tensor(0.0, device=self.device)
            
        # BPR Regularization Loss
        bs = users.shape[0]
        bundles_flat = bundles.view(-1)
        u_emb_0 = self.users_feature[users]
        b_emb_0 = self.bundles_feature[bundles_flat]
        
        safe_items = self.bundle_items[bundles_flat].clamp(min=0)
        i_emb_0 = self.items_feature[safe_items]
        valid_mask = (self.bundle_items[bundles_flat] >= 0).float()
        i_emb_0 = i_emb_0 * valid_mask.unsqueeze(-1)
        
        reg_loss = (1/2)*(u_emb_0.norm(2).pow(2) + 
                          b_emb_0.norm(2).pow(2) + 
                          i_emb_0.norm(2).pow(2)) / float(bs)
                          
        return bpr, c_loss, reg_loss

    # Map the standard forward pass to loss computation for training
    def forward(self, batch, ED_drop=False):
        users, bundles = batch[0].squeeze(1), batch[1]
        embs = self.get_embeddings(test=False)
        return self.get_loss(embs, users, bundles, ED_drop=ED_drop)

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

        u_emb = embs["UI_users"][users]    # [bs, d]
        b_emb = embs["UB_bundles"]         # [NB, d]

        return torch.mm(u_emb, b_emb.t())
