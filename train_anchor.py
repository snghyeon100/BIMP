#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_anchor.py  —  Training script for AnchorRadar
====================================================
Usage:
    python train_anchor.py -d NetEase -m AnchorRadar -g 0
    python train_anchor.py -d iFashion -m AnchorRadar -g 0 -i exp1
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

from utility import AnchorDatasets
from models.anchor import AnchorRadar


# ===========================================================================
# CLI
# ===========================================================================

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",     default="0",         type=str)
    parser.add_argument("-d", "--dataset", default="NetEase",   type=str,
                        help="Dataset: NetEase | iFashion | Youshu")
    parser.add_argument("-m", "--model",   default="AnchorRadar", type=str)
    parser.add_argument("-i", "--info",    default="",          type=str,
                        help="Auxiliary string appended to log file name")
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    conf = yaml.safe_load(open("./config_anchor.yaml"))
    print("load config_anchor.yaml done!")

    paras        = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] in ["AnchorRadar"], \
        "Pls select models from: AnchorRadar"

    # Support "DatasetName_suffix" notation for future extensions
    base_key = dataset_name.split("_")[0]
    conf     = conf[base_key]
    conf["dataset"] = dataset_name
    conf["model"]   = paras["model"]
    conf["gpu"]     = paras["gpu"]
    conf["info"]    = paras["info"]

    dataset = AnchorDatasets(conf)

    conf["num_users"]   = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"]   = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, l2_reg, embedding_size, num_layers, c_lambda, c_temp, anchor_lambda in \
            product(
                conf["lrs"],
                conf["l2_regs"],
                conf["embedding_sizes"],
                conf["num_layerss"],
                conf["c_lambdas"],
                conf["c_temps"],
                conf["anchor_lambdas"],
            ):

        # ---- Paths ----
        log_path               = "./log/%s/%s"                   % (conf["dataset"], conf["model"])
        run_path               = "./runs/%s/%s"                  % (conf["dataset"], conf["model"])
        checkpoint_model_path  = "./checkpoints/%s/%s/model"     % (conf["dataset"], conf["model"])
        checkpoint_conf_path   = "./checkpoints/%s/%s/conf"      % (conf["dataset"], conf["model"])

        for p in [log_path, run_path, checkpoint_model_path, checkpoint_conf_path]:
            os.makedirs(p, exist_ok=True)

        conf["l2_reg"]        = l2_reg
        conf["embedding_size"]= embedding_size
        conf["num_layers"]    = num_layers
        conf["c_lambda"]      = c_lambda
        conf["c_temp"]        = c_temp
        conf["anchor_lambda"] = anchor_lambda

        # ---- Run ID string ----
        settings = []
        if conf["info"]:
            settings.append(conf["info"])
        settings += [
            "Neg_%d" % conf["neg_num"],
            str(conf["batch_size_train"]),
            str(lr),
            str(l2_reg),
            str(embedding_size),
            str(num_layers),
            str(c_lambda),
            str(c_temp),
            "al%.3f" % anchor_lambda,
        ]
        setting = "_".join(settings)

        log_path              = log_path              + "/" + setting
        run_path              = run_path              + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path  = checkpoint_conf_path  + "/" + setting

        run = SummaryWriter(run_path)

        # ---- Model ----
        model = AnchorRadar(conf, dataset.graphs, dataset.anchor_info).to(device)

        optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])

        batch_cnt        = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0

        for epoch in range(conf["epochs"]):
            epoch_anchor = epoch * batch_cnt
            model.train(True)

            pbar = tqdm(
                enumerate(dataset.train_loader),
                total=batch_cnt,
                desc="epoch %d" % epoch,
            )

            for batch_i, batch in pbar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]

                bpr_loss, c_loss = model(batch)
                loss = bpr_loss + conf["c_lambda"] * c_loss
                loss.backward()
                optimizer.step()

                batch_anchor   = epoch_anchor + batch_i
                loss_s         = loss.detach().item()
                bpr_s          = bpr_loss.detach().item()
                c_s            = c_loss.detach().item()

                run.add_scalar("loss_bpr", bpr_s, batch_anchor)
                run.add_scalar("loss_c",   c_s,   batch_anchor)
                run.add_scalar("loss",     loss_s, batch_anchor)

                pbar.set_description(
                    "epoch: %d | loss: %.4f  bpr: %.4f  cl: %.4f"
                    % (epoch, loss_s, bpr_s, c_s)
                )

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {}
                    metrics["val"]  = test(model, dataset.val_loader,  conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    best_metrics, best_perform, best_epoch = log_metrics(
                        conf, model, metrics, run,
                        log_path, checkpoint_model_path, checkpoint_conf_path,
                        epoch, batch_anchor, best_metrics, best_perform, best_epoch,
                    )


# ===========================================================================
# Evaluation helpers  (identical logic to train.py)
# ===========================================================================

def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {k: [0, 0] for k in conf["topk"]}

    device = conf["device"]
    model.eval()
    with torch.no_grad():
        eval_info = model.get_all_embeddings_for_eval()
        for users, ground_truth_u_b, train_mask_u_b in dataloader:
            users = users.to(device)
            pred_b = model.evaluate(eval_info, users)
            pred_b -= 1e8 * train_mask_u_b.to(device)
            # ground_truth is on CPU; bring pred to CPU for indexing
            tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b.cpu(), conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {k: res[0] / res[1] for k, res in topk_res.items()}
    return metrics


def init_best_metrics(conf):
    best_metrics = {
        split: {"recall": {k: 0 for k in conf["topk"]},
                "ndcg":   {k: 0 for k in conf["topk"]}}
        for split in ["val", "test"]
    }
    best_perform = {"val": {}, "test": {}}
    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores= metrics["test"]

    for m, val_score in val_scores.items():
        run.add_scalar("%s_%d/Val"  % (m, topk), val_score[topk],  step)
        run.add_scalar("%s_%d/Test" % (m, topk), test_scores[m][topk], step)

    val_str  = "%s, Top_%d, Val:  recall: %f, ndcg: %f" % (
        curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" % (
        curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    with open(log_path, "a") as log:
        log.write(val_str  + "\n")
        log.write(test_str + "\n")

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run,
                log_path, checkpoint_model_path, checkpoint_conf_path,
                epoch, batch_anchor, best_metrics, best_perform, best_epoch):

    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    topk_ = 20
    print("top%d as the final evaluation standard" % topk_)

    val_rec  = metrics["val"]["recall"][topk_]
    val_ndcg = metrics["val"]["ndcg"][topk_]
    if val_rec > best_metrics["val"]["recall"][topk_] and \
       val_ndcg > best_metrics["val"]["ndcg"][topk_]:

        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = {k: v for k, v in conf.items() if k != "device"}
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a") as log:
            for topk in conf["topk"]:
                for split in ["val", "test"]:
                    for metric in ["recall", "ndcg"]:
                        best_metrics[split][metric][topk] = metrics[split][metric][topk]

                best_perform["test"][topk] = (
                    "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f"
                    % (curr_time, best_epoch, topk,
                       best_metrics["test"]["recall"][topk],
                       best_metrics["test"]["ndcg"][topk])
                )
                best_perform["val"][topk] = (
                    "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f"
                    % (curr_time, best_epoch, topk,
                       best_metrics["val"]["recall"][topk],
                       best_metrics["val"]["ndcg"][topk])
                )
                print(best_perform["val"][topk])
                print(best_perform["test"][topk])
                log.write(best_perform["val"][topk]  + "\n")
                log.write(best_perform["test"][topk] + "\n")

    return best_metrics, best_perform, best_epoch


def get_metrics(metrics, grd, pred, topks):
    """grd is CPU tensor; pred may be CPU or GPU â†' normalise to CPU."""
    pred = pred.cpu()
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = (
            torch.zeros_like(col_indice)
            + torch.arange(pred.shape[0], dtype=torch.long).view(-1, 1)
        )
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk]   = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x
    return metrics


def get_recall(pred, grd, is_hit, topk):
    eps     = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)
    denorm  = pred.shape[0] - (num_pos == 0).sum().item()
    nomina  = (hit_cnt / (num_pos + eps)).sum().item()
    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    """All tensors expected on CPU."""
    def DCG(hit, topk):
        hit = hit / torch.log2(torch.arange(2, topk + 2, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk)

    IDCGs = torch.empty(1 + topk, dtype=torch.float)
    IDCGs[0] = 1
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg     = DCG(is_hit, topk)
    idcg    = IDCGs[num_pos]
    ndcg    = dcg / idcg

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()
    return [nomina, denorm]


if __name__ == "__main__":
    main()
