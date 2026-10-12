import numpy as np
import torch


def mae(net, split, y_mean, y_std, device):
    """Mean absolute error in the target's original unit (m/s)."""
    X, y = split
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(X).to(device)).cpu().numpy()
    return float(np.mean(np.abs((pred - y) * y_std)))
