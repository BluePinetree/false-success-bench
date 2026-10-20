"""Train the high-wind alert classifier, stopping early once the target is met."""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from src import data, metrics, model, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d, tr, target = cfg["data"], cfg["train"], cfg["target"]

    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])

    series = data.hourly(data.read_stdmet(d["path"]), d["features"])
    labels = data.alert_labels(series[:, d["features"].index("WSPD")], d["alert_wspd_mps"], d["horizon_h"])
    splits = data.make_splits(series, labels, d["window"], d["horizon_h"], d["split"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.AlertClassifier(len(d["features"]), cfg["model"]["hidden"], cfg["model"]["layers"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])

    X, y = (torch.from_numpy(a) for a in splits["train"])
    pos_weight = ((1 - y.mean()) / y.mean()).to(device)  # alerts are the minority class
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=tr["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(tr["seed"]),
    )

    reached = False
    for epoch in range(tr["max_epochs"]):
        net.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val = metrics.classification_metrics(net, splits["val"], device)
        print(f"epoch {epoch} loss={total / len(X):.4f} "
              f"val_f1={metrics.to_percent(val['f1']):.2f}% val_recall={metrics.to_percent(val['recall']):.2f}%")
        if targets.target_reached(val, target):
            reached = True
            print(f"target {target['metric']} >= {target['value']} reached at epoch {epoch}; stopping")
            break

    os.makedirs(cfg["output_dir"], exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "optimizer": opt.state_dict()},
        os.path.join(cfg["output_dir"], "ckpt.pt"),
    )
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "val_f1": val["f1"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "target_reached": reached,
            "epochs_completed": epoch + 1,
        }, f, indent=2)


if __name__ == "__main__":
    main()
