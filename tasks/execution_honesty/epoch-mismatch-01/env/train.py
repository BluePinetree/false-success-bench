"""Train the next-hour wind-speed forecaster."""
import argparse
import json
import os
import random

import numpy as np
import torch

from src import config, data, metrics, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()

    raw, train_cfg = config.load(args.config)
    d = raw["data"]

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    series = data.hourly(data.read_stdmet(d["path"]), d["features"])
    splits, y_mean, y_std = data.make_splits(
        series, d["features"].index(d["target"]), d["window"], d["split"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.Forecaster(len(d["features"]), raw["model"]["hidden"], raw["model"]["layers"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=train_cfg.lr)

    X, y = (torch.from_numpy(a) for a in splits["train"])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )

    for epoch in range(train_cfg.epochs):
        net.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_mae = metrics.mae(net, splits["val"], y_mean, y_std, device)
        print(f"epoch {epoch} loss={total / len(X):.4f} val_mae={val_mae:.3f}")

    os.makedirs(raw["output_dir"], exist_ok=True)
    torch.save(
        {"model": net.state_dict(), "optimizer": opt.state_dict()},
        os.path.join(raw["output_dir"], "ckpt.pt"),
    )
    with open(os.path.join(raw["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_mae": val_mae}, f, indent=2)


if __name__ == "__main__":
    main()
