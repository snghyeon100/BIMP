#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnchorRadar: Label-Free Anchor-based Bundle Recommendation
==========================================================
Architecture
  - LightGCN 3-View separate embeddings (UB / UI / BI)
  - Anchor Attention:
      Stage 1 (Local Score)  : MLP(e_u, e_i, e_b, topo_features) -> scalar
      Stage 2 (Global Strength): Linear(e_i^BI) -> scalar
      Final weight alpha = softmax((s1 + lambda*s2) / T)  [masked over bundle items]
  - Score(u, B) = <e_u^UB, e_b^UB> + sum_i alpha_{u,i,B} * <e_u^UI, e_i^UI>
  - Loss: BPR + c_lambda * InfoNCE (contrastive)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


# ===========================================================================
# Utility helpers (shared with MultiCBR)
# ===========================================================================

def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    return rowsum_sqrt @ graph @ colsum_sqrt


def to_tensor(graph):
    graph = graph.tocoo()
    indices = np.vstack((graph.row, graph.col))
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(graph.data),
        torch.Size(graph.shape),
    )


def cal_bpr_loss(pred):
    """pred: [bs, 1+neg_num]  — first column is positive."""
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos  = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos  = pred[:, 0].unsqueeze(1)
    loss = -torch.log(torch.sigmoid(pos - negs))
    return loss.mean()


# ===========================================================================
# AnchorRadar
# ===========================================================================

class AnchorRadar(nn.Module):
    """
    Parameters
    ----------
    conf : dict
        Must contain: embedding_size, num_layers, anchor_lambda,
                      mlp_hidden_size, c_temp, num_users, num_bundles, num_items, device
    raw_graph : list[scipy.sparse]
        [u_b_graph, u_i_graph, b_i_graph]  (same order as MultiCBR)
    anchor_info : dict
        Pre-computed tensors from AnchorDatasets._compute_anchor_info().
        Keys: bundle_items, bundle_mask, bundle_size,
              item_popularity, item_specificity, ui_csr, b_i_csr
    """

    def __init__(self, conf, raw_graph, anchor_info):
        super().__init__()
        self.conf   = conf
        self.device = conf["device"]

        self.emb_size        = conf["embedding_size"]
        self.num_layers      = conf["num_layers"]
        self.anchor_lambda   = conf.get("anchor_lambda", 0.1)
        self.anchor_temp     = conf.get("anchor_temp", 0.3)   # softmax temperature
        self.mlp_hidden      = conf.get("mlp_hidden_size", 64)
        self.c_temp          = conf.get("c_temp", 0.2)

        self.num_users   = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items   = conf["num_items"]

        # Raw graphs (scipy sparse) — kept for graph rebuild if needed
        assert isinstance(raw_graph, list) and len(raw_graph) == 3
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        # Pre-computed topology
        self._load_anchor_info(anchor_info)

        # Learnable embeddings (shared across views, as in LightGCN standard)
        self._init_emb()

        # Graph tensors (Laplace-normalised, no edge dropout)
        self.UB_graph = self._build_prop_graph(self.ub_graph)
        self.UI_graph = self._build_prop_graph(self.ui_graph)
        self.BI_graph = self._build_prop_graph(self.bi_graph)

        # Stage 1 MLP: local anchor score
        # Input: 4 cosine similarities + 5 topology features = 9 scalars
        mlp_in = 4 + 5   # cos(euUB,eiUI), cos(euUI,eiUI), cos(ebUB,eiUI), cos(ebBI,eiBI) + topo
        self.stage1_mlp = nn.Sequential(
            nn.Linear(mlp_in, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 1),
        )

        # Stage 2 linear: global anchor strength  e_i^BI -> scalar
        self.stage2_linear = nn.Linear(self.emb_size, 1, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_emb(self):
        self.users_feature   = nn.Parameter(torch.empty(self.num_users,   self.emb_size))
        self.bundles_feature = nn.Parameter(torch.empty(self.num_bundles, self.emb_size))
        self.items_feature   = nn.Parameter(torch.empty(self.num_items,   self.emb_size))
        nn.init.xavier_normal_(self.users_feature)
        nn.init.xavier_normal_(self.bundles_feature)
        nn.init.xavier_normal_(self.items_feature)

    def _init_weights(self):
        for layer in self.stage1_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.stage2_linear.weight)

    def _load_anchor_info(self, anchor_info):
        dev = self.device
        # Register as buffers so .to(device) moves them automatically
        self.register_buffer("bundle_items_topo",   anchor_info["bundle_items"])     # [N_b, max_T]
        self.register_buffer("bundle_mask_topo",    anchor_info["bundle_mask"])      # [N_b, max_T] bool
        self.register_buffer("bundle_size_topo",    anchor_info["bundle_size"])      # [N_b]
        self.register_buffer("item_pop_topo",       anchor_info["item_popularity"])  # [N_items]
        self.register_buffer("item_spec_topo",      anchor_info["item_specificity"]) # [N_items]

        # Scipy CSR matrices — used for fast user-bundle overlap computation
        self.ui_csr  = anchor_info["ui_csr"]   # stays on CPU as scipy
        self.b_i_csr = anchor_info["b_i_csr"]

    def _build_prop_graph(self, bipartite):
        """Build symmetric bipartite Laplace graph tensor."""
        n_A, n_B = bipartite.shape
        prop = sp.bmat([
            [sp.csr_matrix((n_A, n_A)), bipartite],
            [bipartite.T,               sp.csr_matrix((n_B, n_B))],
        ])
        return to_tensor(laplace_transform(prop)).to(self.device)

    # ------------------------------------------------------------------
    # LightGCN propagation  (no dropout — cleaner for AnchorRadar)
    # ------------------------------------------------------------------

    def _propagate(self, graph, feat_A, feat_B):
        """
        Standard LightGCN layer-mean propagation.
        Returns final embeddings for node sets A and B.
        """
        features    = torch.cat([feat_A, feat_B], dim=0)
        all_feat    = [features]
        for _ in range(self.num_layers):
            features = torch.spmm(graph, features)
            all_feat.append(F.normalize(features, p=2, dim=1))

        # Simple mean over layers (layer-mean LightGCN)
        agg = torch.stack(all_feat, dim=1).mean(dim=1)
        e_A, e_B = torch.split(agg, [feat_A.shape[0], feat_B.shape[0]], dim=0)
        return e_A, e_B

    def get_all_embeddings(self, test=False):
        """
        Run LightGCN on all 3 graphs and return 6 embedding tensors:
            e_u_UB, e_b_UB  —  View A
            e_u_UI, e_i_UI  —  View B
            e_b_BI, e_i_BI  —  View C
        """
        # View A: User-Bundle
        e_u_UB, e_b_UB = self._propagate(self.UB_graph, self.users_feature,   self.bundles_feature)
        # View B: User-Item
        e_u_UI, e_i_UI = self._propagate(self.UI_graph, self.users_feature,   self.items_feature)
        # View C: Bundle-Item
        e_b_BI, e_i_BI = self._propagate(self.BI_graph, self.bundles_feature, self.items_feature)

        return e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI

    # ------------------------------------------------------------------
    # Topology feature construction
    # ------------------------------------------------------------------

    def _build_topo_features(self, users, bundles_flat):
        """
        Build topology feature tensor for (user, bundle) pairs.

        Parameters
        ----------
        users       : [bs] LongTensor — user indices
        bundles_flat: [bs * (1+neg)] LongTensor — bundle indices (flattened)

        Returns
        -------
        topo : [bs*(1+neg), max_T, 5] float tensor
            5 features per item slot:
              0: Global Popularity  (item)
              1: Bundle Size        (bundle)
              2: Item Specificity   (item)
              3: Personal Affinity  (user × item, from UI matrix)
              4: User-Bundle Overlap (user × bundle)
        """
        bs_total = bundles_flat.shape[0]   # bs * (1+neg)
        n_neg1   = bs_total // users.shape[0]
        max_T    = self.bundle_items_topo.shape[1]
        dev      = self.device

        # bundle_items for each (bundle) sample  [bs_total, max_T]
        b_items   = self.bundle_items_topo[bundles_flat]    # [bs_total, max_T]   -1=pad
        b_mask    = self.bundle_mask_topo[bundles_flat]     # [bs_total, max_T]   bool

        # ---- feat 0: Global Popularity ----
        # Clamp pad indices to 0 for safe indexing, then zero out with mask
        safe_items   = b_items.clamp(min=0)                       # [bs_total, max_T]
        feat_pop     = self.item_pop_topo[safe_items]             # [bs_total, max_T]
        feat_pop     = feat_pop * b_mask.float()

        # ---- feat 1: Bundle Size ----
        feat_bsize   = self.bundle_size_topo[bundles_flat]        # [bs_total]
        feat_bsize   = feat_bsize.unsqueeze(1).expand(-1, max_T)  # [bs_total, max_T]

        # ---- feat 2: Item Specificity ----
        feat_spec    = self.item_spec_topo[safe_items]            # [bs_total, max_T]
        feat_spec    = feat_spec * b_mask.float()

        # ---- feat 3: Personal Affinity  UI_{u,i} ----
        # users is [bs]; we need [bs_total] by repeating each user n_neg1 times
        users_exp    = users.repeat_interleave(n_neg1)            # [bs_total]
        # UI affinity: look up from sparse matrix on CPU, bring to device
        ui_rows      = users_exp.cpu().numpy()
        item_idx     = safe_items.cpu().numpy()                   # [bs_total, max_T]
        feat_affinity = np.zeros((bs_total, max_T), dtype=np.float32)
        for k in range(bs_total):
            row      = self.ui_csr.getrow(ui_rows[k])
            ui_items = set(row.indices.tolist())
            for t in range(max_T):
                if item_idx[k, t] >= 0 and item_idx[k, t] in ui_items:
                    feat_affinity[k, t] = 1.0
        feat_affinity = torch.tensor(feat_affinity, dtype=torch.float32, device=dev)

        # ---- feat 4: User-Bundle Overlap ----
        # |{items u bought} ∩ B| / |B|
        feat_overlap = np.zeros(bs_total, dtype=np.float32)
        b_items_np   = b_items.cpu().numpy()
        b_sizes_np   = self.bundle_size_topo[bundles_flat].cpu().numpy()
        for k in range(bs_total):
            row      = self.ui_csr.getrow(ui_rows[k])
            ui_items = set(row.indices.tolist())
            bundle_item_list = [x for x in b_items_np[k] if x >= 0]
            if len(bundle_item_list) == 0:
                feat_overlap[k] = 0.0
            else:
                overlap = sum(1 for it in bundle_item_list if it in ui_items)
                feat_overlap[k] = overlap / len(bundle_item_list)
        feat_overlap_t = torch.tensor(feat_overlap, dtype=torch.float32, device=dev)
        feat_overlap_t = feat_overlap_t.unsqueeze(1).expand(-1, max_T)  # [bs_total, max_T]

        # Stack: [bs_total, max_T, 5]
        topo = torch.stack([
            feat_pop,
            feat_bsize,
            feat_spec,
            feat_affinity,
            feat_overlap_t,
        ], dim=2)

        return topo, b_mask   # [bs_total, max_T, 5], [bs_total, max_T]

    # ------------------------------------------------------------------
    # Anchor Attention
    # ------------------------------------------------------------------

    def _compute_anchor_attention(self, embs, bundles_flat, topo, b_mask):
        """
        Compute per-item anchor weights α for each (user, bundle) pair.

        Parameters
        ----------
        embs        : tuple of 6 embedding matrices (full vocab)
        bundles_flat: [bs_total] bundle indices
        topo        : [bs_total, max_T, 5] topology features
        b_mask      : [bs_total, max_T] bool (True = valid item slot)

        Returns
        -------
        alpha : [bs_total, max_T]  softmax attention weights (masked)
        """
        e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI = embs
        bs_total, max_T, _ = topo.shape
        dev = self.device

        # bundle_items for all bundles in batch
        b_items    = self.bundle_items_topo[bundles_flat]   # [bs_total, max_T]
        safe_items = b_items.clamp(min=0)

        # --- Per-item embedding look-ups ---
        # e_i_UI and e_i_BI: [bs_total, max_T, emb_size]
        e_i_UI_batch = e_i_UI[safe_items]                   # [bs_total, max_T, D]
        e_i_BI_batch = e_i_BI[safe_items]                   # [bs_total, max_T, D]

        # e_b_UB and e_b_BI: [bs_total, D]  -> expand to [bs_total, max_T, D]
        e_b_UB_batch = e_b_UB[bundles_flat].unsqueeze(1).expand(-1, max_T, -1)
        e_b_BI_batch = e_b_BI[bundles_flat].unsqueeze(1).expand(-1, max_T, -1)

        # e_u_UB and e_u_UI are passed in shape [bs_total, D] already (see forward)
        # they need to be broadcast over max_T
        e_u_UB_batch = e_u_UB_batch = None  # will be set by caller — passed as kwarg
        # NOTE: caller passes user embs directly; see _score() below which embeds

        # Stage 2: global anchor strength  s2 [bs_total, max_T]
        s2 = self.stage2_linear(e_i_BI_batch).squeeze(-1)   # [bs_total, max_T]

        return s2, e_i_UI_batch, e_i_BI_batch, e_b_UB_batch, e_b_BI_batch, safe_items

    # ------------------------------------------------------------------
    # Score computation (vectorised)
    # ------------------------------------------------------------------

    def _score_bundles(self, users, bundles, embs, topo, b_mask):
        """
        Compute Score(u, B) for all (user, bundle) pairs in the batch.

        Parameters
        ----------
        users   : [bs] user indices
        bundles : [bs, 1+neg] bundle indices
        embs    : 6-tuple of full-vocab embedding matrices
        topo    : [bs*(1+neg), max_T, 5]
        b_mask  : [bs*(1+neg), max_T] bool

        Returns
        -------
        scores  : [bs, 1+neg]
        """
        e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI = embs

        bs, n_neg1  = bundles.shape
        bs_total    = bs * n_neg1
        max_T       = self.bundle_items_topo.shape[1]
        dev         = self.device

        bundles_flat = bundles.view(-1)                     # [bs_total]
        b_items      = self.bundle_items_topo[bundles_flat] # [bs_total, max_T]
        safe_items   = b_items.clamp(min=0)

        # ---- User embeddings: expand over (1+neg) and max_T ----
        # [bs, D] -> [bs, 1+neg, D] -> [bs_total, D]
        e_u_UB_exp = e_u_UB[users].unsqueeze(1).expand(-1, n_neg1, -1).reshape(bs_total, -1)
        e_u_UI_exp = e_u_UI[users].unsqueeze(1).expand(-1, n_neg1, -1).reshape(bs_total, -1)

        # [bs_total, D] -> [bs_total, max_T, D]
        e_u_UB_3d  = e_u_UB_exp.unsqueeze(1).expand(-1, max_T, -1)
        e_u_UI_3d  = e_u_UI_exp.unsqueeze(1).expand(-1, max_T, -1)

        # Bundle embeddings  [bs_total, D] -> [bs_total, max_T, D]
        e_b_UB_3d  = e_b_UB[bundles_flat].unsqueeze(1).expand(-1, max_T, -1)
        e_b_BI_3d  = e_b_BI[bundles_flat].unsqueeze(1).expand(-1, max_T, -1)

        # Item embeddings   [bs_total, max_T, D]
        e_i_UI_3d  = e_i_UI[safe_items]
        e_i_BI_3d  = e_i_BI[safe_items]

        # ---- Stage 1: local score — dot product based MLP input ----
        # 4 dot products → each [bs_total, max_T, 1]
        dot_eu_UB_ei_UI = (e_u_UB_3d * e_i_UI_3d).sum(-1, keepdim=True)  # cross-view
        dot_eu_UI_ei_UI = (e_u_UI_3d * e_i_UI_3d).sum(-1, keepdim=True)  # same-view
        dot_eb_UB_ei_UI = (e_b_UB_3d * e_i_UI_3d).sum(-1, keepdim=True)  # cross-view
        dot_eb_BI_ei_BI = (e_b_BI_3d * e_i_BI_3d).sum(-1, keepdim=True)  # same-view

        # MLP input: [bs_total, max_T, 9]
        mlp_input = torch.cat([
            dot_eu_UB_ei_UI,   # 1
            dot_eu_UI_ei_UI,   # 1
            dot_eb_UB_ei_UI,   # 1
            dot_eb_BI_ei_BI,   # 1
            topo,              # 5
        ], dim=-1)             # [bs_total, max_T, 9]

        s1 = self.stage1_mlp(mlp_input).squeeze(-1)        # [bs_total, max_T]

        # ---- Stage 2: global anchor strength ----
        s2 = self.stage2_linear(e_i_BI_3d).squeeze(-1)     # [bs_total, max_T]

        # ---- Combined score + mask ----
        combined = (s1 + self.anchor_lambda * s2) / self.anchor_temp  # [bs_total, max_T]

        # Mask padding positions with -inf before softmax
        INF = 1e9
        combined = combined.masked_fill(~b_mask, -INF)

        alpha = F.softmax(combined, dim=-1)                 # [bs_total, max_T]
        # Zero out padding in alpha (softmax may give near-zero but not exact)
        alpha = alpha * b_mask.float()

        # ---- Score ----
        # Base: <e_u^UB, e_b^UB>  [bs_total]
        score_base = (e_u_UB_exp * e_b_UB[bundles_flat]).sum(dim=-1)

        # Item part: sum_i alpha_{u,i,B} * <e_u^UI, e_i^UI>
        ui_dot = (e_u_UI_3d * e_i_UI_3d).sum(dim=-1)      # [bs_total, max_T]
        score_item = (alpha * ui_dot).sum(dim=-1)           # [bs_total]

        scores = (score_base + score_item).view(bs, n_neg1) # [bs, 1+neg]

        # ---- Logging values (Mean Absolute Values) ----
        mean_abs_base = score_base.abs().mean()
        mean_abs_item = score_item.abs().mean()

        # ---- Entropy for regularization ----
        # H = -sum(alpha * log(alpha))
        eps = 1e-9
        a_clamp = alpha.clamp(min=eps)
        H = -(a_clamp * a_clamp.log()).sum(dim=-1)          # [bs_total]
        # Normalize by log(bundle_size) to keep it in [0, 1]
        sizes = b_mask.float().sum(dim=-1).clamp(min=1)
        H_norm = H / (sizes.log() + eps)                    # [bs_total]
        mean_entropy = H_norm.mean()                        # Scalar

        return scores, mean_entropy, mean_abs_base, mean_abs_item

    # ------------------------------------------------------------------
    # Contrastive loss (InfoNCE, same as MultiCBR)
    # ------------------------------------------------------------------

    def _cal_c_loss(self, pos, aug):
        """pos / aug: [bs, D]."""
        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = (pos * aug).sum(dim=1)                 # [bs]
        ttl_score = pos @ aug.T                            # [bs, bs]
        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.exp(ttl_score / self.c_temp).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, batch):
        """
        batch = [users, bundles]
          users   : [bs, 1]
          bundles : [bs, 1+neg_num]
        """
        users, bundles = batch
        users   = users.squeeze(1)                          # [bs]

        embs  = self.get_all_embeddings()
        e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI = embs

        bs, n_neg1 = bundles.shape
        bundles_flat = bundles.view(-1)                     # [bs*(1+neg)]

        # Topology features & mask
        topo, b_mask = self._build_topo_features(users, bundles_flat)

        # Scores and stats
        scores, mean_ent, mean_base, mean_item = self._score_bundles(
            users, bundles, (e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI), topo, b_mask
        )
        
        # BPR Loss 
        pos_scores = scores[:, 0]
        neg_scores = scores[:, 1:]
        bpr_loss   = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()

        # Contrastive Loss (optional)
        c_lambda = self.conf.get("c_lambda", 0.0)
        if c_lambda > 0:
            u_cl = self._cal_c_loss(e_u_UB[users], e_u_UI[users])
            b_cl = self._cal_c_loss(e_b_UB[bundles[:, 0]], e_b_BI[bundles[:, 0]])
            c_loss = (u_cl + b_cl) / 2.0
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        # BPR Regularization Loss (on 0-th layer embeddings)
        u_emb_0 = self.users_feature[users]                 # [bs, D]
        b_emb_0 = self.bundles_feature[bundles_flat]        # [bs*(1+neg), D]
        
        # safely gather item embeddings
        safe_items = self.bundle_items_topo[bundles_flat].clamp(min=0)  
        i_emb_0 = self.items_feature[safe_items]            # [bs*(1+neg), max_T, D]
        i_emb_0 = i_emb_0 * b_mask.unsqueeze(-1).float()    # mask padded items
        
        reg_loss = (1/2)*(u_emb_0.norm(2).pow(2) + 
                          b_emb_0.norm(2).pow(2) + 
                          i_emb_0.norm(2).pow(2)) / float(bs)

        return bpr_loss, c_loss, mean_ent, mean_base, mean_item, reg_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_cache, users):
        """
        Coarse-to-Fine 2-step evaluation.

        Step 1 (Coarse)  O(N_u × N_b)  — score_base dot-product, pick top-N per user.
        Step 2 (Fine)    O(N_u × N)    — Stage 1+2 AnchorRadar only on N candidates.
        Final: scatter item scores back to full [bs_eval, N_b] matrix.
        """
        embs, precomp = eval_cache
        e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI = embs

        e_i_UI_bun = precomp["e_i_UI_bun"]   # [N_b, max_T, D]
        e_i_BI_bun = precomp["e_i_BI_bun"]
        e_b_UB_bun = precomp["e_b_UB_bun"]   # [N_b, D]
        e_b_BI_bun = precomp["e_b_BI_bun"]
        topo_bun   = precomp["topo_bun"]      # [N_b, max_T, 5]
        s2_all     = precomp["s2"]            # [N_b, max_T]
        mask_all   = self.bundle_mask_topo    # [N_b, max_T]

        N_b     = self.num_bundles
        max_T   = self.bundle_items_topo.shape[1]
        bs_eval = users.shape[0]
        N       = min(self.conf.get("rerank_k", 500), N_b)
        uc      = self.conf.get("eval_user_chunk", 64)
        INF     = 1e9

        # ------------------------------------------------------------------
        # Step 1: Coarse — Score_fast = <e_u^UB, e_b^UB> + max_i(<e_u^UI, e_i^UI>)
        # chunked over users to avoid [bs_eval, N_b*max_T] OOM
        # ------------------------------------------------------------------
        eu_UB_all = e_u_UB[users]    # [bs_eval, D]
        eu_UI_all = e_u_UI[users]    # [bs_eval, D]
        D         = eu_UI_all.shape[-1]

        ei_UI_flat = e_i_UI_bun.view(-1, D)          # [N_b*max_T, D]  — shared, no copy
        score_fast = torch.zeros(bs_eval, N_b, device=self.device)

        for u_start in range(0, bs_eval, uc):
            u_end   = min(u_start + uc, bs_eval)
            eu_UB_u = eu_UB_all[u_start:u_end]       # [nu, D]
            eu_UI_u = eu_UI_all[u_start:u_end]

            score_ub_u  = eu_UB_u @ e_b_UB.T         # [nu, N_b]

            ui_u = (eu_UI_u @ ei_UI_flat.T            # [nu, N_b*max_T]
                    ).view(-1, N_b, max_T)
            ui_u = ui_u.masked_fill(
                ~mask_all.unsqueeze(0), -INF
            )
            max_ui_u = ui_u.max(dim=-1).values        # [nu, N_b]

            score_fast[u_start:u_end] = score_ub_u + max_ui_u

        _, topk_idx = torch.topk(score_fast, k=N, dim=1)  # [bs_eval, N]

        # ------------------------------------------------------------------
        # Step 2: Fine — gather & MLP INSIDE user-chunk loop
        #   gather [uc, N, ...] instead of [bs_eval, N, ...] → avoids OOM
        # ------------------------------------------------------------------
        score_item_topk = torch.zeros(bs_eval, N, device=self.device)

        for u_start in range(0, bs_eval, uc):
            u_end    = min(u_start + uc, bs_eval)
            nu       = u_end - u_start
            topk_u   = topk_idx[u_start:u_end]       # [nu, N]

            # Gather bundle tensors for THIS user slice only — [nu, N, ...]
            ei_UI_4d = e_i_UI_bun[topk_u]            # [nu, N, max_T, D]
            ei_BI_4d = e_i_BI_bun[topk_u]
            eb_UB_u  = e_b_UB_bun[topk_u]            # [nu, N, D]
            eb_BI_u  = e_b_BI_bun[topk_u]
            topo_u   = topo_bun[topk_u]              # [nu, N, max_T, 5]
            s2_u     = s2_all[topk_u]               # [nu, N, max_T]
            mask_u   = mask_all[topk_u]             # [nu, N, max_T]

            # Expand user embs: [nu, D] → [nu, N, max_T, D]
            eu_UB_4d = eu_UB_all[u_start:u_end].unsqueeze(1).unsqueeze(2).expand(nu, N, max_T, -1)
            eu_UI_4d = eu_UI_all[u_start:u_end].unsqueeze(1).unsqueeze(2).expand(nu, N, max_T, -1)

            # Expand bundle embs: [nu, N, D] → [nu, N, max_T, D]
            eb_UB_4d = eb_UB_u.unsqueeze(2).expand(nu, N, max_T, -1)
            eb_BI_4d = eb_BI_u.unsqueeze(2).expand(nu, N, max_T, -1)

            # Stage 1 MLP — dot product based, peak: [nu×N, max_T, 9]
            mlp_in = torch.cat([
                (eu_UB_4d * ei_UI_4d).sum(-1, keepdim=True),  # dot(euUB, eiUI)
                (eu_UI_4d * ei_UI_4d).sum(-1, keepdim=True),  # dot(euUI, eiUI)
                (eb_UB_4d * ei_UI_4d).sum(-1, keepdim=True),  # dot(ebUB, eiUI)
                (eb_BI_4d * ei_BI_4d).sum(-1, keepdim=True),  # dot(ebBI, eiBI)
                topo_u,                                      # 5 topo features
            ], dim=-1)                                       # [nu, N, max_T, 9]
            s1 = self.stage1_mlp(
                mlp_in.view(nu * N, max_T, -1)
            ).squeeze(-1).view(nu, N, max_T)

            combined = (s1 + self.anchor_lambda * s2_u) / self.anchor_temp
            combined = combined.masked_fill(~mask_u, -INF)
            alpha    = F.softmax(combined, dim=-1) * mask_u.float()

            ui_dot = (eu_UI_4d * ei_UI_4d).sum(dim=-1)   # [nu, N, max_T]
            score_item_topk[u_start:u_end] = (alpha * ui_dot).sum(dim=-1)

        # Scatter item correction into full [bs_eval, N_b] matrix
        score_item_full = torch.zeros(bs_eval, N_b, device=self.device)
        score_item_full.scatter_(1, topk_idx, score_item_topk)

        return score_fast + score_item_full

    def get_all_embeddings_for_eval(self):
        """
        Pre-compute everything that is INDEPENDENT of specific users.
        Called ONCE before the full evaluation loop.

        Returns (embs, precomp):
          embs    : 6 full-vocab embedding tensors
          precomp : dict of pre-sliced bundle-level tensors
                    (Stage 2 scores, item/bundle embeddings, zero'd topology)
                    → used by evaluate() which adds Stage 1 per user-batch.
        """
        with torch.no_grad():
            embs = self.get_all_embeddings(test=True)
            _, e_b_UB, _, e_i_UI, e_b_BI, e_i_BI = embs

            safe_items = self.bundle_items_topo.clamp(min=0)  # [N_b, max_T]
            max_T      = self.bundle_items_topo.shape[1]
            N_b        = self.num_bundles

            e_i_UI_bun = e_i_UI[safe_items]                   # [N_b, max_T, D]
            e_i_BI_bun = e_i_BI[safe_items]                   # [N_b, max_T, D]

            # Stage 2: pre-compute per-bundle item scores  [N_b, max_T]
            s2 = self.stage2_linear(e_i_BI_bun).squeeze(-1)   # [N_b, max_T]

            # Topology without personal features (affinity=0, overlap=0)
            feat_pop  = self.item_pop_topo[safe_items]         # [N_b, max_T]
            feat_spec = self.item_spec_topo[safe_items]        # [N_b, max_T]
            feat_bsz  = self.bundle_size_topo.unsqueeze(1).expand(-1, max_T)  # [N_b, max_T]
            zeros     = torch.zeros(N_b, max_T, device=self.device)
            topo_bun  = torch.stack([feat_pop, feat_bsz, feat_spec, zeros, zeros], dim=2)

            precomp = {
                "e_i_UI_bun": e_i_UI_bun,   # [N_b, max_T, D]
                "e_i_BI_bun": e_i_BI_bun,   # [N_b, max_T, D]
                "e_b_UB_bun": e_b_UB,        # [N_b, D]
                "e_b_BI_bun": e_b_BI,        # [N_b, D]
                "topo_bun":   topo_bun,       # [N_b, max_T, 5]
                "s2":         s2,             # [N_b, max_T]  Stage 2 pre-computed
            }

        return embs, precomp

