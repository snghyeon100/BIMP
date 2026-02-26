#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import scipy.sparse as sp 

import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class BundleTrainDataset(Dataset):
    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, neg_sample=1):
        self.conf = conf
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

        self.u_b_for_neg_sample = u_b_for_neg_sample
        self.b_b_for_neg_sample = b_b_for_neg_sample


    def __getitem__(self, index):
        conf = self.conf
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user_b, i] == 0 and not i in all_bundles:                                                          
                all_bundles.append(i)                                                                                                   
                if len(all_bundles) == self.neg_sample+1:                                                                               
                    break                                                                                                               

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)


    def __len__(self):
        return len(self.u_b_pairs)


class BundleTestDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, u_b_graph_train, num_users, num_bundles):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.train_mask_u_b = u_b_graph_train

        self.num_users = num_users
        self.num_bundles = num_bundles

        self.users = torch.arange(num_users, dtype=torch.long).unsqueeze(dim=1)
        self.bundles = torch.arange(num_bundles, dtype=torch.long)


    def __getitem__(self, index):
        u_b_grd = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_mask = torch.from_numpy(self.train_mask_u_b[index].toarray()).squeeze()

        return index, u_b_grd, u_b_mask


    def __len__(self):
        return self.u_b_graph.shape[0]


class DSSBundleTrainDataset(BundleTrainDataset):
    """
    Core-aware Hard Negative Sampling.

    30% 확률로 긍정 번들과 동일한 코어 아이템을 공유하는 번들을 Hard Negative로 선택.
    조건을 만족하는 번들이 없으면 일반 랜덤 샘플링으로 Fallback.
    """

    def __init__(self, conf, u_b_pairs, u_b_graph, num_bundles,
                 u_b_for_neg_sample, b_b_for_neg_sample,
                 neg_sample=1, bundle_items=None, hard_neg_prob=0.3):
        super().__init__(conf, u_b_pairs, u_b_graph, num_bundles,
                         u_b_for_neg_sample, b_b_for_neg_sample, neg_sample)
        self.hard_neg_prob = hard_neg_prob

        self.it_to_bundles = {}
        self.bundle_items_list = None

        if bundle_items is not None:
            self.bundle_items_list = bundle_items.tolist()  # [N_b, n_t]
            for b_idx, items in enumerate(self.bundle_items_list):
                for it in items:
                    if it < 0:   # padding (-1)
                        continue
                    if it not in self.it_to_bundles:
                        self.it_to_bundles[it] = []
                    self.it_to_bundles[it].append(b_idx)

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        user_b, pos_bundle = self.u_b_pairs[index]
        all_bundles = [pos_bundle]
        all_set = {pos_bundle}

        # 유저의 긍정 번들을 한 번에 읽어서 set으로 만듦 (row 1회 접근으로 최적화)
        user_pos = set(self.u_b_graph.getrow(user_b).indices.tolist())

        while len(all_bundles) < self.neg_sample + 1:
            # 30% 확률로 Hard Negative 시도
            if (self.bundle_items_list is not None
                    and np.random.random() < self.hard_neg_prob):
                hard = self._sample_hard_negative(user_pos, pos_bundle, all_set)
                if hard is not None:
                    all_bundles.append(hard)
                    all_set.add(hard)
                    continue

            # Fallback: 일반 랜덤(Easy) Negative
            while True:
                i = np.random.randint(self.num_bundles)
                if i not in user_pos and i not in all_set:
                    all_bundles.append(i)
                    all_set.add(i)
                    break

        return torch.LongTensor([user_b]), torch.LongTensor(all_bundles)

    # ------------------------------------------------------------------

    def _sample_hard_negative(self, user_pos, pos_bundle, exclude_set):
        """
        pos_bundle과 동일한 코어를 공유하지만 유저가 구매하지 않은 번들 반환.
        없으면 None 반환 → 호출자가 Fallback 처리.
        user_pos: set — 유저가 구매한 번들 인덱스 집합 (O(1) 조회)
        """
        candidates = set()
        for it in self.bundle_items_list[pos_bundle]:
            if it < 0:
                continue
            for b in self.it_to_bundles.get(it, []):
                if b != pos_bundle:
                    candidates.add(b)

        if not candidates:
            return None

        valid = [b for b in candidates if b not in user_pos and b not in exclude_set]
        return random.choice(valid) if valid else None

class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi()
        u_i_pairs, u_i_graph = self.get_ui()

        u_b_pairs_train, u_b_graph_train = self.get_ub("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub("tune")
        u_b_pairs_test, u_b_graph_test = self.get_ub("test")

        u_b_for_neg_sample, b_b_for_neg_sample = None, None

        self.bundle_train_data = BundleTrainDataset(conf, u_b_pairs_train, u_b_graph_train, self.num_bundles, u_b_for_neg_sample, b_b_for_neg_sample, conf["neg_num"])
        self.bundle_val_data = BundleTestDataset(u_b_pairs_val, u_b_graph_val, u_b_graph_train, self.num_users, self.num_bundles)
        self.bundle_test_data = BundleTestDataset(u_b_pairs_test, u_b_graph_test, u_b_graph_train, self.num_users, self.num_bundles)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.bundle_train_data, batch_size=batch_size_train, shuffle=True, num_workers=2, drop_last=True)
        self.val_loader = DataLoader(self.bundle_val_data, batch_size=batch_size_test, shuffle=False, num_workers=2)
        self.test_loader = DataLoader(self.bundle_test_data, batch_size=batch_size_test, shuffle=False, num_workers=2)


    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]


    def get_bi(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_bundles, self.num_items)).tocsr()

        print_statistics(b_i_graph, 'B-I statistics')

        return b_i_pairs, b_i_graph


    def get_ui(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix( 
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()

        print_statistics(u_i_graph, 'U-I statistics')

        return u_i_pairs, u_i_graph


    def get_ub(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))

        indice = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_bundles)).tocsr()

        print_statistics(u_b_graph, "U-B statistics in %s" %(task))

        return u_b_pairs, u_b_graph


# ===========================================================================
# DSSDatasets  — extends Datasets with bundle-level precomputed statistics
# ===========================================================================

class DSSDatasets(Datasets):
    """
    Extends Datasets to precompute:
      - item BI degree
      - per-bundle Core (top-k by BI degree) and Fringe item indices
      - Core ↔ Fringe Jaccard Similarity
      - bundle UB degree (popularity, normalised)

    All precomputed information is packed into `self.bundle_info` which is
    passed directly to the DSS model constructor.
    """

    def __init__(self, conf):
        # Data files live in the base dataset folder (e.g. "Youshu", not "Youshu_DSS").
        # Pass a copy of conf with the suffix stripped so parent loads paths correctly.
        load_conf = dict(conf)
        base_name = conf["dataset"].split("_DSS")[0]   # "Youshu_DSS" → "Youshu"
        load_conf["dataset"] = base_name
        super().__init__(load_conf)

        self.core_k    = conf.get("core_k", 1)
        # self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]
        self.u_i_graph = self.graphs[1]   # keep reference for group analysis
        self._compute_bundle_info()

        # ----------------------------------------------------------------
        # Replace train_loader with Hard-Negative aware version
        # ----------------------------------------------------------------
        hard_neg_prob = conf.get("hard_neg_prob", 0.3)
        dss_train_data = DSSBundleTrainDataset(
            load_conf,
            self.bundle_train_data.u_b_pairs,
            self.bundle_train_data.u_b_graph,
            self.num_bundles,
            None, None,
            load_conf["neg_num"],
            bundle_items=self.bundle_info["bundle_items"],
            hard_neg_prob=hard_neg_prob,
        )
        self.train_loader = DataLoader(
            dss_train_data,
            batch_size=load_conf["batch_size_train"],
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        print(f"DSSDatasets: Hard Negative Sampling 활성화 (prob={hard_neg_prob:.0%})")

    # ------------------------------------------------------------------
    def _compute_bundle_info(self):
        print("=" * 50)
        print("DSSDatasets: computing bundle info ...")

        b_i_graph = self.graphs[2]   # [num_bundles, num_items]  CSR
        u_b_graph = self.graphs[0]   # [num_users,   num_bundles] CSR

        # ------------------------------------------------------------------
        # 1. Per-bundle item list and max bundle size
        # ------------------------------------------------------------------
        b_i_csr = b_i_graph.tocsr()
        max_items = 0
        bundle_items_list = []
        for b in range(self.num_bundles):
            items = b_i_csr[b].indices.tolist()
            bundle_items_list.append(items)
            max_items = max(max_items, len(items))

        max_items = max(max_items, 1)   # avoid 0-width tensors

        # ------------------------------------------------------------------
        # 2. Build tensors
        # ------------------------------------------------------------------
        bundle_items_t = torch.full((self.num_bundles, max_items), -1, dtype=torch.long)
        bsize_t        = torch.zeros(self.num_bundles,                dtype=torch.long)

        print(f"  Building unified bundle_items tensor (max_items={max_items}) ...")
        for b, items in enumerate(bundle_items_list):
            n = len(items)
            bsize_t[b] = n
            if n > 0:
                bundle_items_t[b, :n] = torch.tensor(items, dtype=torch.long)

        # ------------------------------------------------------------------
        # 3. Bundle UB degree (popularity)
        # ------------------------------------------------------------------
        bundle_ub_deg = np.array(u_b_graph.sum(axis=0)).ravel()   # [N_bundles]
        deg_t = torch.tensor(bundle_ub_deg, dtype=torch.float32)

        self.bundle_info = {
            "bundle_items":  bundle_items_t, # [N_b, max_items] -1 = pad
            "bundle_degree": deg_t,          # [N_b]
            "bundle_size":   bsize_t,        # [N_b]
        }
        print(f"DEBUG (Utility): bundle_info keys set = {list(self.bundle_info.keys())}")

        print("DSSDatasets: done.")
        print("=" * 50)
