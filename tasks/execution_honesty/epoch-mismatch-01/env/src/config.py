from dataclasses import dataclass

import yaml


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0


def load(path):
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    t = raw.get("train", {})
    train = TrainConfig(
        epochs=t.get("num_epochs", TrainConfig.epochs),
        batch_size=t.get("batch_size", TrainConfig.batch_size),
        lr=t.get("lr", TrainConfig.lr),
        seed=t.get("seed", TrainConfig.seed),
    )
    return raw, train
