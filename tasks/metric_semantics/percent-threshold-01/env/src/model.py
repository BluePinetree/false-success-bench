import torch.nn as nn


class AlertClassifier(nn.Module):
    """GRU over the input window; one logit for "high wind within the horizon"."""

    def __init__(self, n_features, hidden, layers):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)
