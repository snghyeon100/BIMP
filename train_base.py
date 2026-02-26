#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_base.py  —  Training script for the DSS Baseline model.
"""

import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim

from utility import DSSDatasets
from models.DSS_Base import DSS_Base as DSS


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",     default="0",      type=str)
    parser.add_argument("-d", "--dataset", default="NetEase_DSS", type=str)
    parser.add_argument("-m", "--model",   default="DSS_Base",    type=str)
    parser.add_argument("-i", "--info",    default="",       type=str)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras        = get_cmd().__dict__
    dataset_name = paras["dataset"]

    # Support "NetEase_DSS" → look up key "NetEase_DSS" in config
    key = dataset_name
    assert key in conf, f"Key '{key}' not found in config.yaml"
    conf         = conf[key]
    conf["dataset"] = dataset_name
    conf["model"]   = paras["model"]

    # Force disable hard negative sampling for Baseline
    conf["hard_neg_prob"] = 0.0
    print("Baseline: Distabled Hard Negative Sampling (hard_neg_prob=0.0)")

    dataset = DSSDatasets(conf)

    conf["gpu"]  = paras["gpu"]
    conf["info"] = paras["info"]
    conf["num_users"]   = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"]   = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, l2_reg, UB_ratio, UI_ratio, BI_ratio, emb_size, num_layers, c_lambda, c_temp in \
            product(conf["lrs"], conf["l2_regs"],
                    conf["UB_ratios"], conf["UI_ratios"], conf["BI_ratios"],
                    conf["embedding_sizes"], conf["num_layers_options"],
                    conf["c_lambdas"], conf["c_temps"]):

        log_path              = "./log/%s/%s"               % (conf["dataset"], conf["model"])
        run_path              = "./runs/%s/%s"              % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path  = "./checkpoints/%s/%s/conf"  % (conf["dataset"], conf["model"])

        for p in [run_path, log_path, checkpoint_model_path, checkpoint_conf_path]:
            os.makedirs(p, exist_ok=True)

        conf["l2_reg"]       = l2_reg
        conf["embedding_size"] = emb_size
        conf["UB_ratio"]     = UB_ratio
        conf["UI_ratio"]     = UI_ratio
        conf["BI_ratio"]     = BI_ratio
        conf["num_layers"]   = num_layers
        conf["c_lambda"]     = c_lambda
        conf["c_temp"]       = c_temp

        settings = []
        if conf["info"]:
            settings += [conf["info"]]
        settings += [conf.get("aug_type", "Noise")]
        settings += ["Neg_%d" % conf["neg_num"], str(conf["batch_size_train"]),
                     str(lr), str(l2_reg), str(emb_size)]
        settings += [str(UB_ratio), str(UI_ratio), str(BI_ratio), str(num_layers)]
        settings += [str(c_lambda), str(c_temp)]
        settings += ["core%d" % conf.get("core_k", 1)]

        setting               = "_".join(settings)
        log_path              = log_path + "/" + setting
        run_path              = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path  = checkpoint_conf_path  + "/" + setting

        run = SummaryWriter(run_path)

        # Model
        model = DSS(conf, dataset.graphs, dataset.bundle_info).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

        batch_cnt        = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])
        ed_interval_bs   = int(batch_cnt * conf.get("ed_interval", 1))

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0

        for epoch in range(conf["epochs"]):
            epoch_anchor = epoch * batch_cnt
            model.train(True)
            pbar = tqdm(enumerate(dataset.train_loader), total=batch_cnt)

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                embs              = model.get_embeddings()
                bpr_loss, c_loss  = model.get_loss(embs, batch[0], batch[1])
                loss = bpr_loss + c_lambda * c_loss

                loss.backward()
                optimizer.step()

                run.add_scalar("loss_bpr", bpr_loss.detach(), batch_anchor)
                run.add_scalar("loss_c",   c_loss.detach(),   batch_anchor)
                run.add_scalar("loss",     loss.detach(),     batch_anchor)
                pbar.set_description(
                    "epoch: %d, loss: %.4f, bpr: %.4f, c: %.4f"
                    % (epoch, loss.item(), bpr_loss.item(), c_loss.item()))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {
                        "val":  test(model, dataset.val_loader,  conf),
                        "test": test(model, dataset.test_loader, conf),
                    }
                    best_metrics, best_perform, best_epoch = log_metrics(
                        conf, model, metrics, run, log_path,
                        checkpoint_model_path, checkpoint_conf_path,
                        epoch, batch_anchor, best_metrics, best_perform, best_epoch)

        # ------------------------------------------------------------------
        # Final best-model analysis
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Training finished. Running final statistical analysis ...")
        if os.path.isfile(checkpoint_model_path):
            model.load_state_dict(torch.load(checkpoint_model_path, map_location=device))
            print(f"Loaded best model from {checkpoint_model_path}")
        compute_analysis_stats(model, dataset, conf, log_path)
        print("=" * 60 + "\n")

        run.close()


# ---------------------------------------------------------------------------
# Metric initialisation
# ---------------------------------------------------------------------------

def init_best_metrics(conf):
    best_metrics = {
        "val":  {"recall": {}, "ndcg": {}},
        "test": {"recall": {}, "ndcg": {}},
    }
    for split in best_metrics:
        for m in best_metrics[split]:
            for k in conf["topk"]:
                best_metrics[split][m][k] = 0.0
    return best_metrics, {"val": {}, "test": {}}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def write_log(run, log_path, topk, step, metrics):
    curr_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    tst_scores = metrics["test"]
    for m in val_scores:
        run.add_scalar("%s_%d/Val"  % (m, topk), val_scores[m][topk], step)
        run.add_scalar("%s_%d/Test" % (m, topk), tst_scores[m][topk], step)
    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
        curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    tst_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
        curr_time, topk, tst_scores["recall"][topk], tst_scores["ndcg"][topk])
    with open(log_path, "a") as f:
        f.write(val_str + "\n")
        f.write(tst_str + "\n")
    print(val_str)
    print(tst_str)


def log_metrics(conf, model, metrics, run, log_path,
                ckpt_model, ckpt_conf, epoch, step,
                best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, step, metrics)

    topk_ = 20
    print("top%d as the final evaluation standard" % topk_)
    improved_recall = metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_]
    improved_ndcg   = metrics["val"]["ndcg"][topk_]   > best_metrics["val"]["ndcg"][topk_]

    if improved_recall and improved_ndcg:
        torch.save(model.state_dict(), ckpt_model)
        dump_conf = {k: v for k, v in conf.items() if k != "device"}
        json.dump(dump_conf, open(ckpt_conf, "w"))
        best_epoch = epoch
        curr_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf["topk"]:
            for split in best_metrics:
                for m in best_metrics[split]:
                    best_metrics[split][m][topk] = metrics[split][m][topk]
            best_perform["test"][topk] = (
                "%s, Best epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f"
                % (curr_time, best_epoch, topk,
                   best_metrics["test"]["recall"][topk],
                   best_metrics["test"]["ndcg"][topk]))
            best_perform["val"][topk] = (
                "%s, Best epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f"
                % (curr_time, best_epoch, topk,
                   best_metrics["val"]["recall"][topk],
                   best_metrics["val"]["ndcg"][topk]))
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
        with open(log_path, "a") as f:
            for topk in conf["topk"]:
                f.write(best_perform["val"][topk]  + "\n")
                f.write(best_perform["test"][topk] + "\n")

    return best_metrics, best_perform, best_epoch


# ---------------------------------------------------------------------------
# Standard test (Recall / NDCG)
# ---------------------------------------------------------------------------

def test(model, dataloader, conf):
    tmp = {m: {k: [0, 0] for k in conf["topk"]} for m in ["recall", "ndcg"]}
    device = conf["device"]
    model.eval()
    with torch.no_grad():
        embs = model.get_embeddings(test=True)
        for users, grd, mask in dataloader:
            pred = model.evaluate(embs, users.to(device))
            pred -= 1e8 * mask.to(device)
            tmp  = get_metrics(tmp, grd, pred, conf["topk"])
    return {m: {k: tmp[m][k][0] / tmp[m][k][1] for k in conf["topk"]} for m in tmp}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def get_metrics(metrics, grd, pred, topks):
    grd = grd.to(pred.device)   # ensure same device as pred (which may be on GPU)
    for topk in topks:
        _, cols = torch.topk(pred, topk)
        rows    = (torch.zeros_like(cols) +
                   torch.arange(pred.shape[0], device=pred.device).view(-1, 1))
        is_hit  = grd[rows.view(-1), cols.view(-1)].view(-1, topk)
        metrics["recall"][topk] = _acc(metrics["recall"][topk], get_recall(pred, grd, is_hit, topk))
        metrics["ndcg"][topk]   = _acc(metrics["ndcg"][topk],   get_ndcg(pred, grd, is_hit, topk))
    return metrics


def _acc(storage, result):
    return [storage[0] + result[0], storage[1] + result[1]]


def get_recall(pred, grd, is_hit, topk):
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)
    denorm  = pred.shape[0] - (num_pos == 0).sum().item()
    nomina  = (hit_cnt / (num_pos + 1e-8)).sum().item()
    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        return (hit / torch.log2(torch.arange(2, topk + 2, device=device, dtype=torch.float))).sum(-1)

    def IDCG(num_pos, topk, device):
        h = torch.zeros(topk, dtype=torch.float, device=device)   # create on correct device
        h[:num_pos] = 1
        return DCG(h, topk, device)

    device  = grd.device
    IDCGs   = torch.zeros(1 + topk, dtype=torch.float, device=device)   # create on correct device
    IDCGs[0] = 1.0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).long()
    dcg     = DCG(is_hit, topk, device)
    idcg    = IDCGs[num_pos].to(device)
    ndcg    = dcg / idcg

    denorm  = pred.shape[0] - (num_pos == 0).sum().item()
    return [ndcg.sum().item(), denorm]


# ---------------------------------------------------------------------------
# Statistical analysis functions
# ---------------------------------------------------------------------------

def compute_analysis_stats(model, dataset, conf, log_path):
    """
    Computes and logs:
      1. Group-wise Recall@20/NDCG@20 (Organic vs Contextual)
      2. Score Contribution (S_base / S_syn / S_cont %)
    """
    device    = conf["device"]
    topk_eval = 20

    model.eval()
    with torch.no_grad():
        embs = model.get_embeddings(test=True)

    u_i_graph   = dataset.u_i_graph           # scipy sparse [N_u, N_i]
    core_items_np = model.core_items.cpu().numpy()  # [N_b, core_k]

    # Accumulators for group-wise metrics
    org_tmp = {m: {topk_eval: [0, 0]} for m in ["recall", "ndcg"]}
    ctx_tmp = {m: {topk_eval: [0, 0]} for m in ["recall", "ndcg"]}
    org_users_seen, ctx_users_seen = 0, 0

    # Accumulators for score contribution
    contrib_base, contrib_syn, contrib_cont = [], [], []

    model.eval()
    with torch.no_grad():
        for users, grd, mask in dataset.test_loader:
            users_np = users.numpy().ravel()
            users_dev = users.to(device)

            scores, s_base, s_syn, s_cont = model.evaluate_with_components(
                embs, users_dev)
            scores_masked = scores - 1e8 * mask.to(device)

            # Top-20 indices for contribution analysis
            _, top20_idx = torch.topk(scores_masked, topk_eval)  # [bs, 20]

            # --- Score contribution ---
            for i in range(len(users_np)):
                for b_idx in top20_idx[i].tolist():
                    sb = abs(s_base[i, b_idx].item())
                    ss = abs(s_syn[i,  b_idx].item())
                    sc = abs(s_cont[i, b_idx].item())
                    total = sb + ss + sc + 1e-8
                    contrib_base.append(sb / total * 100)
                    contrib_syn.append(ss  / total * 100)
                    contrib_cont.append(sc / total * 100)

            # --- Group-wise ---
            # For each user, find their positive bundles in the test set
            grd_np = grd.numpy()  # [bs, N_b]
            for i, u in enumerate(users_np):
                pos_bundles = grd_np[i].nonzero()[0].tolist()
                if not pos_bundles:
                    continue

                # Get UI interactions for user u
                ui_row      = u_i_graph[u]
                ui_items    = set(ui_row.indices.tolist())

                org_pos_bundles = []
                ctx_pos_bundles = []
                for b in pos_bundles:
                    primary_core = int(core_items_np[b, 0])
                    if primary_core in ui_items:
                        org_pos_bundles.append(b)
                    else:
                        ctx_pos_bundles.append(b)

                pred_u = scores_masked[i]  # [N_b]

                def _eval_group(pos_list):
                    if not pos_list:
                        return None
                    # Build a 1-row grd for this user restricted to group bundles
                    grd_row  = torch.zeros(1, scores.shape[1])
                    for pb in pos_list:
                        grd_row[0, pb] = 1.0
                    pred_row = pred_u.unsqueeze(0).cpu()
                    _, cols  = torch.topk(pred_row, topk_eval)
                    rows     = torch.zeros_like(cols)
                    is_hit   = grd_row[rows.view(-1), cols.view(-1)].view(1, topk_eval)
                    rec      = get_recall(pred_row, grd_row, is_hit, topk_eval)
                    ndcg     = get_ndcg(pred_row, grd_row, is_hit, topk_eval)
                    return rec, ndcg

                res_org = _eval_group(org_pos_bundles)
                res_ctx = _eval_group(ctx_pos_bundles)
                if res_org:
                    org_tmp["recall"][topk_eval] = _acc(org_tmp["recall"][topk_eval], res_org[0])
                    org_tmp["ndcg"][topk_eval]   = _acc(org_tmp["ndcg"][topk_eval],   res_org[1])
                    org_users_seen += 1
                if res_ctx:
                    ctx_tmp["recall"][topk_eval] = _acc(ctx_tmp["recall"][topk_eval], res_ctx[0])
                    ctx_tmp["ndcg"][topk_eval]   = _acc(ctx_tmp["ndcg"][topk_eval],   res_ctx[1])
                    ctx_users_seen += 1

    # ------------------------------------------------------------------
    # Print & log results
    # ------------------------------------------------------------------
    lines = ["\n===== Statistical Analysis ====="]

    # 1. Group-wise
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    org_rec  = safe_div(org_tmp["recall"][topk_eval][0], org_tmp["recall"][topk_eval][1])
    org_ndcg = safe_div(org_tmp["ndcg"][topk_eval][0],   org_tmp["ndcg"][topk_eval][1])
    ctx_rec  = safe_div(ctx_tmp["recall"][topk_eval][0], ctx_tmp["recall"][topk_eval][1])
    ctx_ndcg = safe_div(ctx_tmp["ndcg"][topk_eval][0],   ctx_tmp["ndcg"][topk_eval][1])

    lines.append("[Group-wise Performance @%d]" % topk_eval)
    lines.append("  Organic   group: Recall=%.5f  NDCG=%.5f  (users w/ pos bundles: %d)"
                 % (org_rec, org_ndcg, org_users_seen))
    lines.append("  Contextual group: Recall=%.5f  NDCG=%.5f  (users w/ pos bundles: %d)"
                 % (ctx_rec, ctx_ndcg, ctx_users_seen))

    # 2. Score contribution
    import numpy as np
    if contrib_base:
        m_base = float(np.mean(contrib_base))
        m_syn  = float(np.mean(contrib_syn))
        m_cont = float(np.mean(contrib_cont))
        lines.append("[Score Contribution in Top-%d recommendations]" % topk_eval)
        lines.append("  S_base  : %.2f %%" % m_base)
        lines.append("  S_syn   : %.2f %%" % m_syn)
        lines.append("  S_cont  : %.2f %%" % m_cont)

    lines.append("=" * 32)
    output = "\n".join(lines)
    print(output)
    with open(log_path, "a") as f:
        f.write(output + "\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
