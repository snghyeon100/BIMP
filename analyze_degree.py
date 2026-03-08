#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_degree.py  —  Statistical analysis of DSS_Base_Degree personalized attention.

Analyses:
  1. Attention Entropy:  H(u,b) = -Σ w_ui * log(w_ui)
     → Low entropy = sharp, personalized focus. High = uniform (= mean-pool)
  2. Inter-User Weight Divergence: D(u1, u2, b) = ||w_u1,b - w_u2,b||_2
     → Non-zero divergence = true personalization vs DSS_Base (always 0)

Usage:
    python analyze_degree.py -d NetEase_DSS -g 0 --ckpt <checkpoint_path>
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from utility import DSSDatasets
from models.DSS_Base_Degree import DSS_Base as DSS


# ---------------------------------------------------------------------------
def get_cmd():
    p = argparse.ArgumentParser()
    p.add_argument("-g", "--gpu",     default="0",           type=str)
    p.add_argument("-d", "--dataset", default="NetEase_DSS", type=str)
    p.add_argument("--ckpt",          default=None,           type=str,
                   help="Path to saved model checkpoint (state_dict).")
    p.add_argument("--num_samples",   default=500,            type=int,
                   help="Number of bundles to sample for analysis.")
    p.add_argument("--pairs_per_bundle", default=100,         type=int,
                   help="Max user-pairs per bundle for divergence analysis.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: compute attention weights for a batch of (users, bundle_idx)
# Returns w: [bs, n_t], mask: [bs, n_t]
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_attention_weights(model, users, bundle_idx):
    """
    users: 1-D LongTensor [bs]
    bundle_idx: int
    Returns w [bs, n_t], mask [bs, n_t bool]
    """
    alpha_all = model._compute_alpha(users)                       # [bs, NI]

    items_b = model.bundle_items[[bundle_idx]].expand(len(users), -1)  # [bs, n_t]
    mask_b  = (items_b >= 0)                                           # [bs, n_t]
    items_b = items_b.clamp(min=0)

    alpha_b = alpha_all.gather(1, items_b)                        # [bs, n_t]
    alpha_b = alpha_b.masked_fill(~mask_b, -1e9)
    w       = torch.softmax(alpha_b, dim=-1)                      # [bs, n_t]

    return w.cpu(), mask_b.cpu()


# ---------------------------------------------------------------------------
# Analysis 1: Attention Entropy
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyse_entropy(model, dataset, conf, num_samples=500):
    """
    For each sampled bundle b and a batch of users who interacted with b,
    compute H(u,b) = -Σ w_ui * log(w_ui + 1e-12).

    Compares:
      • DSS_Base_Degree (personalized)  → model's learned weights
      • Uniform baseline               → 1/|b| for each item  (= DSS_Base)
    """
    device    = conf["device"]
    ub_train  = dataset.graphs[0]                  # scipy CSR [NU, NB]
    NB        = conf["num_bundles"]

    # Sample bundles that have at least 2 items and 10+ users
    rng = np.random.default_rng(42)
    bundle_sizes = np.array(model.bundle_size.cpu())
    eligible = np.where(bundle_sizes >= 2)[0]
    sampled  = rng.choice(eligible, size=min(num_samples, len(eligible)), replace=False)

    personalized_H, uniform_H = [], []

    for b_idx in sampled:
        # users who interacted with this bundle (train)
        users_b = ub_train.getcol(b_idx).nonzero()[0]
        if len(users_b) < 2:
            continue
        users_t = torch.LongTensor(users_b).to(device)

        w, mask = compute_attention_weights(model, users_t, int(b_idx))
        # w: [bs, n_t], mask: [bs, n_t]

        for i in range(len(users_b)):
            m = mask[i]                            # valid items
            wi = w[i][m].float()
            # personalized entropy
            H_pers = -(wi * torch.log(wi + 1e-12)).sum().item()
            # uniform entropy
            n_valid = m.sum().item()
            H_unif  = np.log(n_valid) if n_valid > 1 else 0.0

            personalized_H.append(H_pers)
            uniform_H.append(H_unif)

    personalized_H = np.array(personalized_H)
    uniform_H      = np.array(uniform_H)

    print("\n" + "=" * 55)
    print("[Analysis 1] Attention Entropy H(u,b)")
    print("=" * 55)
    print(f"  Samples (user×bundle pairs): {len(personalized_H):,}")
    print(f"  DSS_Base   (uniform)  mean H : {uniform_H.mean():.4f}  ± {uniform_H.std():.4f}")
    print(f"  DSS_Degree (learned)  mean H : {personalized_H.mean():.4f}  ± {personalized_H.std():.4f}")
    ratio = personalized_H.mean() / (uniform_H.mean() + 1e-8)
    print(f"  Ratio (Degree/Base)         : {ratio:.4f}  (< 1 = sharper / more personalized)")
    print("=" * 55)

    return personalized_H, uniform_H


# ---------------------------------------------------------------------------
# Analysis 2: Inter-User Weight Divergence
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyse_divergence(model, dataset, conf, num_samples=200, pairs_per_bundle=100):
    """
    For each sampled bundle, pick random pairs of users who interacted with it.
    Compute D(u1, u2, b) = ||w_u1,b - w_u2,b||_2.

    DSS_Base divergence is always 0 (same e_b for all users).
    """
    device   = conf["device"]
    ub_train = dataset.graphs[0]

    bundle_sizes = np.array(model.bundle_size.cpu())
    eligible = np.where(bundle_sizes >= 2)[0]
    rng      = np.random.default_rng(0)
    sampled  = rng.choice(eligible, size=min(num_samples, len(eligible)), replace=False)

    divergences = []

    for b_idx in sampled:
        users_b = ub_train.getcol(b_idx).nonzero()[0]
        if len(users_b) < 2:
            continue
        users_t = torch.LongTensor(users_b).to(device)

        w, mask = compute_attention_weights(model, users_t, int(b_idx))
        # w: [nu, n_t]

        nu = len(users_b)
        # random pairs
        num_pairs = min(pairs_per_bundle, nu * (nu - 1) // 2)
        pairs = set()
        attempts = 0
        while len(pairs) < num_pairs and attempts < num_pairs * 10:
            i, j = rng.integers(0, nu, size=2)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
            attempts += 1

        for i, j in pairs:
            # only compare valid items (union of both masks)
            m  = mask[i] | mask[j]
            wi = w[i][m].float()
            wj = w[j][m].float()
            d  = (wi - wj).norm(2).item()
            divergences.append(d)

    divergences = np.array(divergences)

    print("\n" + "=" * 55)
    print("[Analysis 2] Inter-User Weight Divergence D(u1,u2,b)")
    print("=" * 55)
    print(f"  User pairs sampled : {len(divergences):,}")
    print(f"  DSS_Base   (uniform) divergence : 0.0000  (always)")
    print(f"  DSS_Degree (learned) mean  D    : {divergences.mean():.4f}")
    print(f"  DSS_Degree (learned) median D   : {np.median(divergences):.4f}")
    print(f"  DSS_Degree (learned) max D      : {divergences.max():.4f}")
    pct_nonzero = (divergences > 1e-4).mean() * 100
    print(f"  Pairs with D > 0.0001 (%)       : {pct_nonzero:.1f}%")
    print("=" * 55)

    return divergences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import yaml

    args    = get_cmd().__dict__
    conf    = yaml.safe_load(open("./config.yaml"))
    key     = args["dataset"]
    conf    = conf[key]
    conf["dataset"] = key

    dataset = DSSDatasets(conf)
    conf["num_users"]   = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"]   = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = args["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    # Need embedding_size etc — load from checkpoint conf if available
    ckpt_conf_path = args["ckpt"].replace("/model", "/conf") if args["ckpt"] else None
    if ckpt_conf_path and os.path.isfile(ckpt_conf_path):
        saved = json.load(open(ckpt_conf_path))
        for k in ["embedding_size", "num_layers", "beta_ui", "c_temp",
                  "UI_ratio", "UB_ratio", "BI_ratio"]:
            if k in saved:
                conf[k] = saved[k]

    # Set defaults if not found
    conf.setdefault("embedding_size", 64)
    conf.setdefault("num_layers", 2)
    conf.setdefault("beta_ui", 0.1)
    conf.setdefault("c_temp", 0.2)
    conf.setdefault("UI_ratio", 0.0)
    conf.setdefault("UB_ratio", 0.0)
    conf.setdefault("BI_ratio", 0.0)

    model = DSS(conf, dataset.graphs, dataset.bundle_info).to(device)

    if args["ckpt"] and os.path.isfile(args["ckpt"]):
        model.load_state_dict(torch.load(args["ckpt"], map_location=device))
        print(f"Loaded checkpoint: {args['ckpt']}")
    else:
        print("No checkpoint provided — using randomly initialized weights (sanity check only).")

    model.eval()

    print(f"\nλ (lambda_ubui, learned) = {model.lambda_ubui.item():.4f}")
    print(f"  → UI signal weight : 1.00")
    print(f"  → UB-BI signal weight: {max(model.lambda_ubui.item(), 0):.4f}")

    analyse_entropy(model, dataset, conf, num_samples=args["num_samples"])
    analyse_divergence(model, dataset, conf,
                       num_samples=args["num_samples"],
                       pairs_per_bundle=args["pairs_per_bundle"])


if __name__ == "__main__":
    main()
