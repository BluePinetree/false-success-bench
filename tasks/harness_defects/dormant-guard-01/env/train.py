"""Train the next-hour wind-speed forecaster and record what the run planned and did."""
import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from src import data, metrics, model

DEBUG_MAX_WINDOWS = 1000  # small runs while debugging the pipeline


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

    series = data.hourly(data.read_stdmet(d["path"]), d["features"])
    segments, mean, std = data.split_segments(series, d["split"])
    target_idx = d["features"].index(d["target"])
    y_std = float(std[target_idx])

    train = data.windows(segments["train"], mean, std, target_idx, d["window"],
                         max_windows=d.get("max_windows", DEBUG_MAX_WINDOWS))
    val = data.windows(segments["val"], mean, std, target_idx, d["window"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.Forecaster(len(d["features"]), cfg["model"]["hidden"], cfg["model"]["layers"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=tr["lr"])

    X, y = (torch.from_numpy(a) for a in train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=tr["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(tr["seed"]),
    )

    for epoch in range(tr["epochs"]):
        net.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        val_mae = metrics.mae(net, val, y_std, device)
        print(f"epoch {epoch} loss={total / len(X):.4f} val_mae={val_mae:.3f}")

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict()}, os.path.join(out, "ckpt.pt"))
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_mae": val_mae}, f, indent=2)
    with open(os.path.join(out, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "planned": {
                "epochs": tr["epochs"],
                "train_windows": data.n_windows(segments["train"], d["window"]),
            },
            "executed": {
                "epochs": epoch + 1,
                "n_train": len(X),
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
