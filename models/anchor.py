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
      Final weight α = softmax(s1 + λ·s2)  [masked over bundle items]
  - Score(u, B) = <e_u^UB, e_b^UB> + Σ_i α_{u,i,B} · <e_u^UI, e_i^UI>
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
        # Input dim: 5 × emb_size (5 embedding vectors) + 5 (topology features)
        mlp_in = self.emb_size * 5 + 5
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

        # ---- Stage 1: local score ----
        # MLP input: concat of 5 emb vectors + 5 topo features
        # [bs_total, max_T, 5*D + 5]
        mlp_input = torch.cat([
            e_u_UB_3d,   # D
            e_u_UI_3d,   # D
            e_i_UI_3d,   # D
            e_b_UB_3d,   # D
            e_b_BI_3d,   # D
            topo,        # 5
        ], dim=-1)       # [bs_total, max_T, 5D+5]

        s1 = self.stage1_mlp(mlp_input).squeeze(-1)        # [bs_total, max_T]

        # ---- Stage 2: global anchor strength ----
        s2 = self.stage2_linear(e_i_BI_3d).squeeze(-1)     # [bs_total, max_T]

        # ---- Combined score + mask ----
        combined = s1 + self.anchor_lambda * s2             # [bs_total, max_T]

        # Mask padding positions with -inf before softmax
        INF = 1e9
        combined = combined.masked_fill(~b_mask, -INF)

        alpha = F.softmax(combined, dim=-1)                 # [bs_total, max_T]
        # Zero out padding in alpha (softmax may give near-zero but not exact)
        alpha = alpha * b_mask.float()

        # ---- Score ----
        # Base: <e_u^UB, e_b^UB>  [bs_total]
        score_base = (e_u_UB_exp * e_b_UB[bundles_flat]).sum(dim=-1)

        # Item part: Σ_i α_{u,i,B} · <e_u^UI, e_i^UI>
        ui_dot = (e_u_UI_3d * e_i_UI_3d).sum(dim=-1)      # [bs_total, max_T]
        score_item = (alpha * ui_dot).sum(dim=-1)           # [bs_total]

        scores = (score_base + score_item).view(bs, n_neg1) # [bs, 1+neg]
        return scores

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

        # Scores
        scores = self._score_bundles(users, bundles, embs, topo, b_mask)

        # BPR Loss
        bpr_loss = cal_bpr_loss(scores)

        # Contrastive Loss — skip entirely when c_lambda == 0
        c_lambda = self.conf.get("c_lambda", 0.0)
        if c_lambda > 0:
            u_cl = self._cal_c_loss(e_u_UB[users], e_u_UI[users])
            b_cl = self._cal_c_loss(e_b_UB[bundles[:, 0]], e_b_BI[bundles[:, 0]])
            c_loss = (u_cl + b_cl) / 2.0
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        return bpr_loss, c_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_cache, users):
        """
        Personalised U×B score matrix — Stage 1+2 attention.
        Memory-safe via USER chunk × BUNDLE chunk double loop.

        Peak tensor size: [uc, bc, max_T, 5D+5]  (fully materialised by cat)
        e.g. uc=16, bc=512, max_T=5, D=64 → 16×512×5×325×4B ≈ 53 MB  (fine)
        vs old single-user-loop: [2048, 512, 5, 325×4B ≈ 6.8 GB  (OOM)
        """
        embs, precomp = eval_cache
        e_u_UB, e_b_UB, e_u_UI, e_i_UI, e_b_BI, e_i_BI = embs

        e_i_UI_bun = precomp["e_i_UI_bun"]   # [N_b, max_T, D]
        e_i_BI_bun = precomp["e_i_BI_bun"]
        e_b_UB_bun = precomp["e_b_UB_bun"]   # [N_b, D]
        e_b_BI_bun = precomp["e_b_BI_bun"]
        topo_bun   = precomp["topo_bun"]      # [N_b, max_T, 5]
        s2         = precomp["s2"]            # [N_b, max_T]
        mask       = self.bundle_mask_topo    # [N_b, max_T]

        N_b    = self.num_bundles
        max_T  = self.bundle_items_topo.shape[1]
        bs_eval= users.shape[0]
        bc     = self.conf.get("eval_bundle_chunk", 512)  # bundle chunk
        uc     = self.conf.get("eval_user_chunk",   64)   # user chunk
        INF    = 1e9

        # Base score: pure dot-product → one mm, no OOM
        score_base = e_u_UB[users] @ e_b_UB.T              # [bs_eval, N_b]
        score_item = torch.zeros(bs_eval, N_b, device=self.device)

        for b_start in range(0, N_b, bc):
            b_end    = min(b_start + bc, N_b)
            nb       = b_end - b_start

            # Slices — bundle side (no user dim yet)
            ei_UI_c = e_i_UI_bun[b_start:b_end]   # [nb, max_T, D]
            ei_BI_c = e_i_BI_bun[b_start:b_end]
            eb_UB_c = e_b_UB_bun[b_start:b_end]   # [nb, D]
            eb_BI_c = e_b_BI_bun[b_start:b_end]
            topo_c  = topo_bun[b_start:b_end]      # [nb, max_T, 5]
            s2_c    = s2[b_start:b_end]             # [nb, max_T]
            mask_c  = mask[b_start:b_end]           # [nb, max_T]

            for u_start in range(0, bs_eval, uc):
                u_end = min(u_start + uc, bs_eval)
                nu    = u_end - u_start

                # User embs for this sub-batch: [nu, D]
                eu_UB_u = e_u_UB[users[u_start:u_end]]   # [nu, D]
                eu_UI_u = e_u_UI[users[u_start:u_end]]

                # Expand to [nu, nb, max_T, D]  — materialised by cat below
                eu_UB_4d = eu_UB_u.unsqueeze(1).unsqueeze(2).expand(nu, nb, max_T, -1)
                eu_UI_4d = eu_UI_u.unsqueeze(1).unsqueeze(2).expand(nu, nb, max_T, -1)

                ei_UI_4d = ei_UI_c.unsqueeze(0).expand(nu, -1, -1, -1)
                ei_BI_4d = ei_BI_c.unsqueeze(0).expand(nu, -1, -1, -1)
                eb_UB_4d = eb_UB_c.unsqueeze(0).unsqueeze(2).expand(nu, -1, max_T, -1)
                eb_BI_4d = eb_BI_c.unsqueeze(0).unsqueeze(2).expand(nu, -1, max_T, -1)
                topo_4d  = topo_c.unsqueeze(0).expand(nu, -1, -1, -1)

                # Stage 1 MLP — peak tensor: [nu×bc, max_T, 5D+5]
                mlp_in = torch.cat(
                    [eu_UB_4d, eu_UI_4d, ei_UI_4d, eb_UB_4d, eb_BI_4d, topo_4d],
                    dim=-1
                )                                          # [nu, nb, max_T, 5D+5]
                s1 = self.stage1_mlp(
                    mlp_in.view(nu * nb, max_T, -1)
                ).squeeze(-1).view(nu, nb, max_T)         # [nu, nb, max_T]

                # Combine Stage 1 + broadcast Stage 2
                combined = s1 + self.anchor_lambda * s2_c.unsqueeze(0)
                mask_exp = mask_c.unsqueeze(0).expand(nu, -1, -1)
                combined = combined.masked_fill(~mask_exp, -INF)
                alpha    = F.softmax(combined, dim=-1) * mask_exp.float()

                # Σ_i α · <e_u^UI, e_i^UI>
                ui_dot = (eu_UI_4d * ei_UI_4d).sum(dim=-1)  # [nu, nb, max_T]
                score_item[u_start:u_end, b_start:b_end] = \
                    (alpha * ui_dot).sum(dim=-1)             # [nu, nb]

        return score_base + score_item

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

