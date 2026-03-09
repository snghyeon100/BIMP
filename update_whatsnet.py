import os

target_file = r"c:\Users\wotjs\Desktop\DMlab\BIMP\models\DSS_whatsnet.py"

with open(target_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

import_dgl_added = False
dgl_build_graphs_added = False

for i, line in enumerate(lines):
    # 1. Import dgl
    if line.startswith("import scipy.sparse as sp") and not import_dgl_added:
        new_lines.append(line)
        new_lines.append("import dgl\n")
        import_dgl_added = True
        continue
        
    # 2. Add build_dgl_graphs
    if line.strip() == "self._build_item_bundles()" and not dgl_build_graphs_added:
        new_lines.append(line)
        new_lines.append("        self._build_dgl_graphs()\n")
        dgl_build_graphs_added = True
        continue
    
    # 3. Add _build_dgl_graphs definition right before _make_prop_graph
    if line.strip().startswith("def _make_prop_graph("):
        dgl_code = """
    def _build_dgl_graphs(self):
        bi = self.bundle_items
        NB, n_t = bi.shape
        b_idx = torch.arange(NB).unsqueeze(1).expand(NB, n_t)
        valid = bi >= 0
        v_b = b_idx[valid].to(self.device)
        v_i = bi[valid].long()

        self.g_b2i = dgl.heterograph({
            ('bundle', 'contains', 'item'): (v_b, v_i),
        }, num_nodes_dict={'bundle': self.num_bundles, 'item': self.num_items})

        self.g_i2b = dgl.heterograph({
            ('item', 'in', 'bundle'): (v_i, v_b)
        }, num_nodes_dict={'item': self.num_items, 'bundle': self.num_bundles})
        
        def rank_norm(deg, valid_mask):
            deg_f = deg.float().masked_fill(~valid_mask, -1.0)
            order = deg_f.argsort(dim=-1, descending=True)
            rank = torch.zeros_like(deg_f)
            rank.scatter_(1, order, torch.arange(n_t, dtype=torch.float, device=self.device).unsqueeze(0).expand(NB, -1))
            n_valid = valid_mask.sum(1, keepdim=True).float().clamp(min=2) - 1
            rank_n = (rank / n_valid).clamp(0.0, 1.0)
            return rank_n.masked_fill(~valid_mask, 0.0)

        ui_deg = self._ui_item_deg[bi.clamp(min=0)]
        bi_deg = self._bi_item_deg[bi.clamp(min=0)]
        
        rk_ui = rank_norm(ui_deg, valid)
        rk_bi = rank_norm(bi_deg, valid)
        rev_ui = (1.0 - rk_ui).masked_fill(~valid, 0.0)
        
        pe_raw = torch.stack([rk_ui, rk_bi, rev_ui], dim=-1) # [NB, n_t, 3]
        pe_valid = pe_raw[valid] # [E, 3]
        
        self.g_i2b.edges['in'].data['pe_raw'] = pe_valid
        self.g_b2i.edges['contains'].data['pe_raw'] = pe_valid

"""
        if not "def _build_dgl_graphs" in "".join(new_lines[-20:]):
            new_lines.append(dgl_code)
        
    # 4. Remove _compute_pe to _bimp_forward completely
    # We trace from def _compute_pe(self, items_2d, valid):
    if line.strip().startswith("def _compute_pe(self, items_2d, valid):"):
        skip = True
        
    if skip and line.strip() == "return H_b_g, X_i_g, V_b_tokens, valid_V":
        skip = False
        
        # Insert new DGL BIMP logic
        dgl_bimp_code = """
    # ------------------------------------------------------------------
    # §2/4  WithinATT & DGL UDFs
    # ------------------------------------------------------------------

    def _within_att(self, S_mab, T_mab, tokens, mask=None):
        B   = tokens.shape[0]
        ind = self.bimp_I.unsqueeze(0).expand(B, -1, -1)   # [B, m, d]
        S   = S_mab(ind, tokens, mask_y=mask)              # [B, m, d]
        T   = T_mab(tokens, S)                             # [B, n_t, d]
        return T

    def _i2b_msg(self, edges):
        pe = self.W_pe(edges.data['pe_raw'])
        return {'z': edges.src['h'] + pe}

    def _i2b_reduce(self, nodes):
        Z = nodes.mailbox['z']  # [B, deg, d]
        V_b = self._within_att(self.mab_i2b_within, self.mab_i2b_outer, Z, mask=None)
        H_prev = nodes.data['h'].unsqueeze(1)  # [B, 1, d]
        H_new = self.mab_i2b_main(H_prev, V_b, mask_y=None)  # [B, 1, d]
        return {'h_new': H_new.squeeze(1)}

    def _b2i_msg(self, edges):
        pe = self.W_pe(edges.data['pe_raw'])
        return {'g': edges.src['h'] + pe}

    def _b2i_reduce(self, nodes):
        E_comb = nodes.mailbox['g']  # [B, deg, d]
        G = self._within_att(self.mab_b2i_within, self.mab_b2i_outer, E_comb, mask=None)
        X_prev = nodes.data['h'].unsqueeze(1)
        X_new = self.mab_b2i_main(X_prev, G, mask_y=None)
        return {'h_new': X_new.squeeze(1)}

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
            sg_i2b.nodes['bundle'].data['h'] = H_b_g[sg_i2b.nodes('bundle')]
            sg_i2b.update_all(self._i2b_msg, self._i2b_reduce, etype='in')
            H_b_g[sg_i2b.nodes('bundle')] = sg_i2b.nodes['bundle'].data['h_new']

            # (B) B→I
            if self.conf.get("use_b2i", True):
                sg_b2i.nodes['bundle'].data['h'] = H_b_g[sg_b2i.nodes('bundle')]
                sg_b2i.nodes['item'].data['h'] = X_i_g[sg_b2i.nodes('item')]
                sg_b2i.update_all(self._b2i_msg, self._b2i_reduce, etype='contains')
                X_i_g[sg_b2i.nodes('item')] = sg_b2i.nodes['item'].data['h_new']

        items_loc_2d  = self.bundle_items[bundles_uniq]
        valid_loc     = (items_loc_2d >= 0)
        items_loc_cs  = items_loc_2d.clamp(min=0)

        if use_bimp_vb:
            Z_loc = X_i_g[items_loc_cs]
            Z_loc = Z_loc.masked_fill(~valid_loc.unsqueeze(-1), 0.0)
            V_b_raw = self._within_att(self.mab_i2b_within, self.mab_i2b_outer, Z_loc, mask=valid_loc)
        else:
            V_b_raw = X_i_init_full[items_loc_cs]

        V_b_tokens = V_b_raw.masked_fill(~valid_loc.unsqueeze(-1), 0.0)
        
        return H_b_g, X_i_g, V_b_tokens, valid_loc
"""
        new_lines.append(dgl_bimp_code)
        continue
        
    if not skip:
        new_lines.append(line)

with open(target_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
