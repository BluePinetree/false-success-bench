"""Train the next-hour wind-speed forecaster; keep the epoch with the best validation MAE
and evaluate that model once on the held-out test split."""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import yaml

from src import data, metrics, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d, tr = cfg["data"], cfg["train"]

    random.seed(tr["seed"])
    np.random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])

    raw = {name: data.load_split(d["splits_dir"], name) for name in ("train", "val", "test")}
    mean = raw["train"][1].mean(axis=0)
    std = raw["train"][1].std(axis=0) + 1e-8
    target_idx = d["features"].index(d["target"])
    splits = {name: data.windows(x, mean, std, target_idx, d["window"]) for name, (_, x) in raw.items()}
    y_std = float(std[target_idx])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.Forecaster(len(d["features"]), cfg["model"]["hidden"], cfg["model"]["layers"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])

    X, y = (torch.from_numpy(a) for a in splits["train"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=tr["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(tr["seed"]),
    )

    history = []
    best = None
    for epoch in range(tr["max_epochs"]):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_mae = metrics.mae(net, splits["val"], y_std, device)
        history.append(val_mae)
        print(f"epoch {epoch} val_mae={val_mae:.3f}")
        if best is None or val_mae < best[1]:
            best = (epoch, val_mae, copy.deepcopy(net.state_dict()))

    best_epoch, best_val, state = best
    net.load_state_dict(state)
    test_mae = metrics.mae(net, splits["test"], y_std, device)
    print(f"best epoch {best_epoch}: val_mae={best_val:.3f} test_mae={test_mae:.3f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    torch.save({"model": state}, os.path.join(cfg["output_dir"], "ckpt.pt"))
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "val_mae": best_val,
            "test_mae": test_mae,
            "val_mae_history": history,
            "split_fingerprints": {name: data.fingerprint(t, x) for name, (t, x) in raw.items()},
        }, f, indent=2)


if __name__ == "__main__":
    main()
