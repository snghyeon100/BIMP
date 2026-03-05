#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_attention.py  —  Run statistical analysis on trained DSS_whatsnet Stage 2 Attention
"""

import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utility import DSSDatasets
from models.DSS import DSS

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",     default="0",      type=str)
    parser.add_argument("-d", "--dataset", default="NetEase_DSS", type=str)
    parser.add_argument("-m", "--model",   default="DSS",    type=str)
    parser.add_argument("-p", "--ckpt",    default="",       type=str, help="Path to best model.pt. If empty, tries to find it automatically.")
    return parser.parse_args()

def calc_entropy(alpha, valid):
    """
    alpha: [M, n_t] attention weights summing to 1.
    valid: [M, n_t] bool mask.
    Returns the mean normalised entropy. 
    0 means sharp (one-hot), 1 means uniform.
    """
    eps = 1e-8
    # alpha * log(alpha)
    ent = - (alpha * torch.log(alpha + eps)).sum(dim=-1)  # [M]
    n_valid = valid.sum(dim=-1).float()                   # [M]
    
    # max entropy is log(n_valid)
    max_ent = torch.log(n_valid + eps)
    # normalise: entropy / max_entropy
    norm_ent = ent / (max_ent + eps)
    
    # Only average over bundles with >1 item
    mask = n_valid > 1
    return ent[mask].mean().item(), norm_ent[mask].mean().item()

def main():
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    conf = yaml.safe_load(open("./config.yaml"))
    
    key = dataset_name
    assert key in conf, f"Key '{key}' not found in config.yaml"
    conf         = conf[key]
    conf["dataset"] = dataset_name
    conf["model"]   = paras["model"]

    dataset = DSSDatasets(conf)

    conf["gpu"]  = paras["gpu"]
    conf["num_users"]   = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"]   = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

    print(f"Dataset: {conf['dataset']}, Model: {conf['model']}, Device: {device}")

    model = DSS(conf, dataset.graphs, dataset.bundle_info).to(device)
    
    ckpt_path = paras["ckpt"]
    if not ckpt_path:
        # Try to find the latest or default checkpoint
        base_dir = f"./checkpoints/{conf['dataset']}/{conf['model']}/model"
        if os.path.isdir(base_dir):
            subdirs = os.listdir(base_dir)
            if subdirs:
                ckpt_path = os.path.join(base_dir, subdirs[-1])
    
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded trained model from {ckpt_path}")
    else:
        print(f"WARNING: Could not find model checkpoint at '{ckpt_path}'. Running with untrained weights for code test.")

    model.eval()
    print("Pre-computing representations...")
    with torch.no_grad():
        embs = model.get_multi_modal_representations(test=True)

    print("Analyzing Stage 2 Attention Distribution on Test Set...")
    
    all_max_weight = []
    all_entropy    = []
    all_norm_entropy = []
    
    with torch.no_grad():
        pbar = tqdm(dataset.test_loader)
        for users, grd, _ in pbar:
            users_dev = users.to(device)
            # This triggers _stage2_score internally and populates model.last_alpha
            scores = model.evaluate(embs, users_dev) 
            
            # alpha: [M, n_t] where M = batch_size * num_bundles 
            # (or evaluated bundles if evaluate does chunking)
            if hasattr(model, 'last_alpha'):
                alpha = model.last_alpha
                
                # We only want to aggregate statistics for the positive ground-truth bundles 
                # or top predicted bundles. Let's look at the top-20 predicted bundles for these users.
                _, top20_idx = torch.topk(scores, 20)  # [bs, 20]
                
                # alpha is flat across batches? Let's check reshape
                # In DSS code, evaluate() usually returns [bs, n_b]. 
                # And in _stage2_score, alpha is [M, n_t] where M=bs*n_b
                bs, nb = scores.shape
                alpha_reshaped = alpha.view(bs, nb, -1)  # [bs, nb, n_t]
                
                # Gather top 20 alphas
                # [bs, 20, 1]
                idx_exp = top20_idx.unsqueeze(-1).expand(-1, -1, alpha_reshaped.shape[-1])
                top20_alphas = torch.gather(alpha_reshaped, 1, idx_exp)  # [bs, 20, n_t]
                
                # Calculate max weight %
                max_w, _ = top20_alphas.max(dim=-1)  # [bs, 20]
                all_max_weight.extend(max_w.cpu().numpy().flatten().tolist())
                
                # To calculate entropy, we need the valid mask. 
                # Reconstruct valid mask by checking alpha > 0
                valid = top20_alphas > 0
                
                # Flatten batch and top20
                flat_top20_alphas = top20_alphas.view(-1, top20_alphas.shape[-1])
                flat_valid = valid.view(-1, valid.shape[-1])
                
                ent, n_ent = calc_entropy(flat_top20_alphas, flat_valid)
                all_entropy.append(ent)
                all_norm_entropy.append(n_ent)

    print("\n===== Attention Distribution Analysis =====")
    print(f"Total Top-20 Bundle-User Interactions Evaluated: {len(all_max_weight)}")
    print(f"Average Max Attention Weight (Peak focus %): {np.mean(all_max_weight)*100:.2f}%")
    
    if all_norm_entropy:
        print(f"Average Normalised Entropy: {np.mean(all_norm_entropy):.4f} (0.0=Sharp/One-hot, 1.0=Uniform/Flat)")
        print(f"Average Absolute Entropy: {np.mean(all_entropy):.4f}")
    
    # Histogram of max weights
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(all_max_weight, bins=bins)
    print("\nMax Attention Weight Distribution:")
    for i in range(len(hist)):
        print(f"  {bins[i]:.1f} ~ {bins[i+1]:.1f}: {hist[i]} ({hist[i]/len(all_max_weight)*100:.1f}%)")

    print("===========================================\n")

if __name__ == "__main__":
    main()
