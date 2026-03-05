#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_whatsnet.py  —  Training script explicitly for the DSS_whatsnet model.
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
from models.DSS_whatsnet import DSS as DSS_whatsnet_model

def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",     default="0",      type=str)
    parser.add_argument("-d", "--dataset", default="NetEase_DSS", type=str)
    parser.add_argument("-m", "--model",   default="DSS_whatsnet", type=str)
    parser.add_argument("-i", "--info",    default="",       type=str)
    return parser.parse_args()

def main():
    conf = yaml.safe_load(open("./config.yaml"))
    
    paras        = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] == "DSS_whatsnet", "Use -m DSS_whatsnet for train_whatsnet.py"

    key = dataset_name
    assert key in conf, f"Key '{key}' not found in config.yaml"
    conf         = conf[key]
    conf["dataset"] = dataset_name
    conf["model"]   = paras["model"]

    dataset = DSSDatasets(conf)

    conf["gpu"]  = paras["gpu"]
    conf["info"] = paras["info"]
    conf["num_users"]   = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"]   = dataset.num_items

    os.environ["CUDA_VISIBLE_DEVICES"] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device

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
        conf["UB_ratio"]     = float(UB_ratio)
        conf["UI_ratio"]     = float(UI_ratio)
        conf["BI_ratio"]     = float(BI_ratio)
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

        # Instantiate DSS_whatsnet Model
        model = DSS_whatsnet_model(conf, dataset.graphs, dataset.bundle_info).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

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

                ED_drop = (conf.get("aug_type") == "ED" and
                           (batch_anchor + 1) % ed_interval_bs == 0)

                bpr_loss, c_loss, reg_loss = model(batch, ED_drop=ED_drop)
                loss = bpr_loss + c_lambda * c_loss + l2_reg * reg_loss
                
                loss.backward()
                optimizer.step()

                run.add_scalar("loss_bpr", bpr_loss.detach(), batch_anchor)
                run.add_scalar("loss_c",   c_loss.detach(),   batch_anchor)
                run.add_scalar("loss_reg", (l2_reg * reg_loss).detach(), batch_anchor)
                run.add_scalar("loss",     loss.detach(),     batch_anchor)
                
                pbar.set_description(
                    "epoch: %d, loss: %.4f, bpr: %.4f, c: %.4f, reg: %.2e"
                    % (epoch, loss.item(), bpr_loss.item(), c_loss.item(), (l2_reg * reg_loss).item()))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {
                        "val":  test(model, dataset.val_loader,  conf),
                        "test": test(model, dataset.test_loader, conf),
                    }
                    best_metrics, best_perform, best_epoch = log_metrics(
                        conf, model, metrics, run, log_path,
                        checkpoint_model_path, checkpoint_conf_path,
                        epoch, batch_anchor, best_metrics, best_perform, best_epoch)

        print("\n" + "=" * 60)
        print("Training finished.")
        if os.path.isfile(checkpoint_model_path):
            model.load_state_dict(torch.load(checkpoint_model_path, map_location=device))
        print("=" * 60 + "\n")
        run.close()

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

def log_metrics(conf, model, metrics, run, log_path,
                ckpt_model, ckpt_conf, epoch, step,
                best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, step, metrics)

    topk_ = 20
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
        with open(log_path, "a") as f:
            for topk in conf["topk"]:
                f.write(best_perform["val"][topk]  + "\n")
                f.write(best_perform["test"][topk] + "\n")

    return best_metrics, best_perform, best_epoch

def test(model, dataloader, conf):
    tmp = {m: {k: [0, 0] for k in conf["topk"]} for m in ["recall", "ndcg"]}
    device = conf["device"]
    model.eval()
    with torch.no_grad():
        embs = model.get_multi_modal_representations(test=True)
        for users, grd, mask in dataloader:
            pred = model.evaluate(embs, users.to(device))
            pred -= 1e8 * mask.to(device)
            tmp  = get_metrics(tmp, grd, pred, conf["topk"])
    return {m: {k: tmp[m][k][0] / tmp[m][k][1] for k in conf["topk"]} for m in tmp}

def get_metrics(metrics, grd, pred, topks):
    grd = grd.to(pred.device)
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
        h = torch.zeros(topk, dtype=torch.float, device=device)
        h[:num_pos] = 1
        return DCG(h, topk, device)

    device  = grd.device
    IDCGs   = torch.zeros(1 + topk, dtype=torch.float, device=device)
    IDCGs[0] = 1.0
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).long()
    dcg     = DCG(is_hit, topk, device)
    idcg    = IDCGs[num_pos].to(device)
    ndcg    = dcg / idcg

    denorm  = pred.shape[0] - (num_pos == 0).sum().item()
    return [ndcg.sum().item(), denorm]

if __name__ == "__main__":
    main()
