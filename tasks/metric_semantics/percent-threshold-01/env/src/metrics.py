import numpy as np
import torch


def classification_metrics(net, split, device):
    """Precision, recall and F1 of the positive class, as fractions in [0, 1]."""
    X, y = split
    net.eval()
    with torch.no_grad():
        pred = (net(torch.from_numpy(X).to(device)) > 0).cpu().numpy()
    truth = y == 1
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def to_percent(x):
    return 100.0 * x
