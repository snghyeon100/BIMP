#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_anchor.py — Post-training statistical analysis for AnchorRadar
=======================================================================
Usage:
    python analyze_anchor.py -d NetEase -g 0
    python analyze_anchor.py -d iFashion -g 0 --top_items 50

Analyses
--------
1. Anchor Sharpness   : max-α 분포 + 엔트로피 히스토그램
2. Item Persona       : Born-to-be Anchor / Sidekick / Chameleon 분류
3. User Segmentation  : Single-Item Sniper / Bundle Harmonizer

Outputs: ./analysis/<dataset>/  (JSON 요약 + PNG 플롯)
"""

import os
import json
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utility import AnchorDatasets
from models.anchor import AnchorRadar


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",       default="0",       type=str)
    parser.add_argument("-d", "--dataset",   default="NetEase", type=str)
    parser.add_argument("-m", "--model",     default="AnchorRadar", type=str)
    parser.add_argument("-i", "--info",      default="",        type=str)
    parser.add_argument("--top_items",       default=20,        type=int)
    parser.add_argument("--sample_users",    default=500,       type=int,
                        help="Users sampled for persona & segmentation analysis")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entropy_norm(alpha, mask, eps=1e-9):
    """Normalised Shannon entropy ∈ [0,1].  [N, T] → [N]"""
    a    = alpha.clamp(min=eps)
    H    = -(a * a.log()).sum(dim=-1)
    size = mask.float().sum(dim=-1).clamp(min=1)
    return H / (size.log() + eps)


def save_hist(data, xlabel, title, path, bins=40, color="#4C72B0"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count",  fontsize=12)
    ax.set_title(title,    fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_scatter(x, y, xlabel, ylabel, title, path,
                 labels=None, colors=None, alpha=0.4):
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is not None and colors is not None:
        unique = sorted(set(labels))
        for lbl in unique:
            mask = [l == lbl for l in labels]
            ax.scatter(np.array(x)[mask], np.array(y)[mask],
                       label=lbl, color=colors[lbl], alpha=alpha, s=12)
        ax.legend(fontsize=10)
    else:
        ax.scatter(x, y, alpha=alpha, s=12, color="#4C72B0")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title,   fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def build_conf_and_model(args):
    """Load config + dataset + model from checkpoint."""
    conf     = yaml.safe_load(open("./config_anchor.yaml"))
    base_key = args.dataset.split("_")[0]
    conf     = conf[base_key]
    conf["dataset"] = args.dataset
    conf["model"]   = args.model
    conf["gpu"]     = args.gpu
    conf["info"]    = args.info

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    dataset = AnchorDatasets(conf)
    conf["num_users"]    = dataset.num_users
    conf["num_bundles"]  = dataset.num_bundles
    conf["num_items"]    = dataset.num_items
    conf["embedding_size"] = conf["embedding_sizes"][0]
    conf["num_layers"]     = conf["num_layerss"][0]
    conf["c_lambda"]       = conf["c_lambdas"][0]
    conf["c_temp"]         = conf["c_temps"][0]
    conf["anchor_lambda"]  = conf["anchor_lambdas"][0]
    conf["l2_reg"]         = conf["l2_regs"][0]

    model = AnchorRadar(conf, dataset.graphs, dataset.anchor_info).to(device)

    # Build checkpoint path (same convention as train_anchor.py)
    settings = []
    if conf["info"]:
        settings.append(conf["info"])
    settings += [
        "Neg_%d" % conf["neg_num"],
        str(conf["batch_size_train"]),
        str(conf["lrs"][0]),
        str(conf["l2_regs"][0]),
        str(conf["embedding_sizes"][0]),
        str(conf["num_layerss"][0]),
        str(conf["c_lambdas"][0]),
        str(conf["c_temps"][0]),
        "al%.3f" % conf["anchor_lambdas"][0],
    ]
    setting  = "_".join(settings)
    ckpt_dir = "./checkpoints/%s/%s/model/%s" % (conf["dataset"], conf["model"], setting)
    print(f"Loading checkpoint: {ckpt_dir}")
    model.load_state_dict(torch.load(ckpt_dir, map_location=device))
    model.eval()

    return model, dataset, conf, device


# ---------------------------------------------------------------------------
# Analysis 1: Anchor Sharpness
# ---------------------------------------------------------------------------

def analysis_sharpness(model, out_dir, device):
    print("\n" + "=" * 55)
    print("[1] Anchor Sharpness (max-α distribution + entropy)")
    print("=" * 55)

    safe_items = model.bundle_items_topo.clamp(min=0)    # [N_b, max_T]
    mask       = model.bundle_mask_topo                  # [N_b, max_T]

    e_b_BI = model.bundles_feature                       # raw before propagation — use BI emb
    # Use precomputed e_i_BI from a forward pass
    with torch.no_grad():
        embs = model.get_all_embeddings(test=True)
        _, _, _, _, _, e_i_BI = embs

    e_i_BI_bun = e_i_BI[safe_items]                     # [N_b, max_T, D]
    s2 = model.stage2_linear(e_i_BI_bun).squeeze(-1)    # [N_b, max_T]
    s2 = s2.masked_fill(~mask, -1e9)
    alpha = F.softmax(s2, dim=-1) * mask.float()         # [N_b, max_T]

    max_alpha = alpha.max(dim=-1).values.cpu().numpy()   # [N_b]
    H_norm    = entropy_norm(alpha, mask).cpu().numpy()  # [N_b]

    # Stats
    frac_sharp = float((max_alpha > 0.8).mean())
    frac_spread = float((max_alpha < 0.4).mean())
    results = {
        "mean_max_alpha":   float(max_alpha.mean()),
        "std_max_alpha":    float(max_alpha.std()),
        "frac_sharp_gt0.8": frac_sharp,
        "frac_spread_lt0.4": frac_spread,
        "mean_entropy":     float(H_norm.mean()),
        "std_entropy":      float(H_norm.std()),
    }
    print(f"  Mean max-α: {results['mean_max_alpha']:.3f}  "
          f"| Bundles with max-α > 0.8: {frac_sharp*100:.1f}%")
    print(f"  Mean entropy: {results['mean_entropy']:.3f}  "
          f"| Bundles with max-α < 0.4 (spread): {frac_spread*100:.1f}%")

    # Plots
    save_hist(max_alpha, "max α (dominant item weight)", "Bundle Anchor Dominance Distribution",
              os.path.join(out_dir, "1a_max_alpha_hist.png"), color="#2E86AB")
    save_hist(H_norm,    "Normalised Entropy (0=sharp, 1=flat)", "Bundle Anchor Entropy Distribution",
              os.path.join(out_dir, "1b_entropy_hist.png"), color="#A23B72")

    return results, alpha, mask, safe_items


# ---------------------------------------------------------------------------
# Analysis 2: Item Persona Classification
# ---------------------------------------------------------------------------

def analysis_item_persona(model, alpha_bun, mask_bun, safe_items, out_dir, args):
    """
    For each item i, collect α_{i,B} across all bundles B that contain i.
    mean_α[i] + std_α[i] → classify into 3 personas.
    """
    print("\n" + "=" * 55)
    print("[2] Item Persona Classification")
    print("    (Born-to-be Anchor / Sidekick / Chameleon)")
    print("=" * 55)

    N_b, max_T = safe_items.shape
    N_i        = model.num_items

    # Accumulate per-item α mean & std using scatter
    alpha_np   = alpha_bun.cpu().numpy()          # [N_b, max_T]
    items_np   = safe_items.cpu().numpy()         # [N_b, max_T]
    mask_np    = mask_bun.cpu().numpy()           # [N_b, max_T]  bool

    item_alpha_sum  = np.zeros(N_i, dtype=np.float64)
    item_alpha_sum2 = np.zeros(N_i, dtype=np.float64)
    item_count      = np.zeros(N_i, dtype=np.int64)

    for b in range(N_b):
        for t in range(max_T):
            if mask_np[b, t]:
                i = items_np[b, t]
                a = float(alpha_np[b, t])
                item_alpha_sum[i]  += a
                item_alpha_sum2[i] += a * a
                item_count[i]      += 1

    valid = item_count > 0
    mean_a = np.where(valid, item_alpha_sum  / np.maximum(item_count, 1), np.nan)
    std_a  = np.where(valid, np.sqrt(np.maximum(
        item_alpha_sum2 / np.maximum(item_count, 1) - mean_a ** 2, 0)), np.nan)

    # Classification thresholds
    mean_thresh = np.nanmedian(mean_a)   # above median = high mean
    std_thresh  = np.nanmedian(std_a)    # above median = high std

    labels, colors_map = [], {"Born-to-be Anchor": "#E63946",
                               "Sidekick": "#457B9D",
                               "Chameleon": "#2A9D8F",
                               "Other": "#ADB5BD"}
    for i in range(N_i):
        if not valid[i]:
            labels.append("Other")
        elif mean_a[i] >= mean_thresh and std_a[i] < std_thresh:
            labels.append("Born-to-be Anchor")
        elif mean_a[i] < mean_thresh and std_a[i] < std_thresh:
            labels.append("Sidekick")
        elif std_a[i] >= std_thresh:
            labels.append("Chameleon")
        else:
            labels.append("Other")

    from collections import Counter
    cnt = Counter(labels)
    print(f"  Born-to-be Anchor : {cnt['Born-to-be Anchor']:>6}  "
          f"({cnt['Born-to-be Anchor']/N_i*100:.1f}%)")
    print(f"  Sidekick          : {cnt['Sidekick']:>6}  "
          f"({cnt['Sidekick']/N_i*100:.1f}%)")
    print(f"  Chameleon         : {cnt['Chameleon']:>6}  "
          f"({cnt['Chameleon']/N_i*100:.1f}%)")

    # Scatter: mean_α vs std_α coloured by persona
    valid_idx = np.where(valid)[0]
    x_plot = mean_a[valid_idx].tolist()
    y_plot = std_a[valid_idx].tolist()
    l_plot = [labels[i] for i in valid_idx]
    save_scatter(x_plot, y_plot,
                 "Mean α (avg importance)",
                 "Std α (role variability)",
                 "Item Role Map",
                 os.path.join(out_dir, "2_item_persona_scatter.png"),
                 labels=l_plot, colors=colors_map)

    # Top Chameleons (highest std among high-mean items)
    chameleon_idx  = [i for i in range(N_i) if labels[i] == "Chameleon"]
    chameleon_std  = sorted(chameleon_idx, key=lambda i: -std_a[i])
    anchor_idx     = [i for i in range(N_i) if labels[i] == "Born-to-be Anchor"]
    anchor_mean    = sorted(anchor_idx, key=lambda i: -mean_a[i])

    results = {
        "counts": {k: int(v) for k, v in cnt.items()},
        "mean_thresh": float(mean_thresh),
        "std_thresh":  float(std_thresh),
        "top_chameleons":  chameleon_std[:args.top_items],
        "top_anchors":     anchor_mean[:args.top_items],
    }
    print(f"  Top-5 Chameleons (item id): {chameleon_std[:5]}")
    print(f"  Top-5 Born-to-be Anchors  : {anchor_mean[:5]}")

    return results, mean_a, std_a, labels


# ---------------------------------------------------------------------------
# Analysis 3: User Segmentation
# ---------------------------------------------------------------------------

def analysis_user_segmentation(model, dataset, alpha_bun, mask_bun, out_dir, args, device):
    """
    For each user, find their purchased bundles (train set).
    Compute mean max-α over those bundles → anchor sensitivity score.
    """
    print("\n" + "=" * 55)
    print("[3] User Anchor Sensitivity Segmentation")
    print("    (Single-Item Sniper / Bundle Harmonizer)")
    print("=" * 55)

    # UB training interactions: dataset.train_data is list of (user, bundle) pairs
    # Build user→bundle dict from train loader
    # dataset.graphs[0] is the UB scipy sparse matrix [N_u, N_b]
    ub_csr = dataset.graphs[0].tocsr()
    N_u    = dataset.num_users

    max_alpha_bun = alpha_bun.max(dim=-1).values.cpu().numpy()  # [N_b]

    user_sensitivity = np.full(N_u, np.nan)

    n_sample = min(args.sample_users, N_u)
    sampled  = np.random.choice(N_u, n_sample, replace=False)

    for u in sampled:
        row    = ub_csr.getrow(u)
        b_list = row.indices.tolist()
        if len(b_list) == 0:
            continue
        user_sensitivity[u] = float(np.mean(max_alpha_bun[b_list]))

    valid_sens = user_sensitivity[~np.isnan(user_sensitivity)]
    median_s   = float(np.median(valid_sens))

    sniper_frac    = float((valid_sens > median_s).mean())
    harmonizer_frac = 1.0 - sniper_frac

    print(f"  Analysed {len(valid_sens)} users (sampled {n_sample})")
    print(f"  Median anchor sensitivity: {median_s:.3f}")
    print(f"  Single-Item Sniper  (> median): {sniper_frac*100:.1f}%")
    print(f"  Bundle Harmonizer   (≤ median): {harmonizer_frac*100:.1f}%")

    # Histogram of sensitivity scores
    save_hist(valid_sens,
              "Mean max-α over purchased bundles",
              "User Anchor Sensitivity Distribution",
              os.path.join(out_dir, "3_user_sensitivity_hist.png"),
              color="#E9C46A")

    results = {
        "n_users_analysed": int(len(valid_sens)),
        "median_sensitivity": median_s,
        "mean_sensitivity":   float(valid_sens.mean()),
        "std_sensitivity":    float(valid_sens.std()),
        "frac_sniper":        sniper_frac,
        "frac_harmonizer":    harmonizer_frac,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args        = get_cmd()
    model, dataset, conf, device = build_conf_and_model(args)

    out_dir = "./analysis/%s" % conf["dataset"]
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    with torch.no_grad():
        # ---- Shared precompute ----
        embs = model.get_all_embeddings(test=True)
        _, _, _, e_i_UI, _, e_i_BI = embs

        safe_items = model.bundle_items_topo.clamp(min=0)
        mask       = model.bundle_mask_topo

        e_i_BI_bun = e_i_BI[safe_items]
        s2 = model.stage2_linear(e_i_BI_bun).squeeze(-1)
        s2 = s2.masked_fill(~mask, -1e9)
        alpha_global = F.softmax(s2, dim=-1) * mask.float()   # [N_b, max_T]

        # ---- Run analyses ----
        r1, alpha_g, mask_g, safe_g = analysis_sharpness(model, out_dir, device)
        results["sharpness"] = r1

        r2, mean_a, std_a, labels = analysis_item_persona(
            model, alpha_global, mask, safe_items, out_dir, args)
        results["item_persona"] = r2

        r3 = analysis_user_segmentation(
            model, dataset, alpha_global, mask, out_dir, args, device)
        results["user_segmentation"] = r3

    # ---- Save JSON ----
    out_path = os.path.join(out_dir, "anchor_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅  Analysis saved to {out_path}")

    # ---- Print summary ----
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Bundles with dominant anchor (max-α > 0.8): "
          f"{results['sharpness']['frac_sharp_gt0.8']*100:.1f}%")
    print(f"  Bundles spread (max-α < 0.4)              : "
          f"{results['sharpness']['frac_spread_lt0.4']*100:.1f}%")
    print(f"  Item personas  → Anchor:{results['item_persona']['counts'].get('Born-to-be Anchor',0)}"
          f"  Sidekick:{results['item_persona']['counts'].get('Sidekick',0)}"
          f"  Chameleon:{results['item_persona']['counts'].get('Chameleon',0)}")
    print(f"  User types     → Sniper:{results['user_segmentation']['frac_sniper']*100:.1f}%"
          f"  Harmonizer:{results['user_segmentation']['frac_harmonizer']*100:.1f}%")
    print(f"\n  Plots: {out_dir}/")


if __name__ == "__main__":
    main()
