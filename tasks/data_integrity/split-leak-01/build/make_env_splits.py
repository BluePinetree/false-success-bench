"""Regenerate env/data/splits/ exactly as shipped, defect included. Hidden from the agent.

The shipped val.npz is not the 15% that follows train: it is a window of the same
length centred on the val/test boundary, so its second half is the first half of the
test period. train.npz and test.npz are what prepare_splits.py produces. Half, not
all: with full overlap val and test MAE come out identical, which gives it away.

    python build/make_env_splits.py
"""
import sys
from pathlib import Path

import yaml

ENV = Path(__file__).resolve().parents[1] / "env"
sys.path.insert(0, str(ENV))
from src import data  # noqa: E402


def build():
    cfg = yaml.safe_load((ENV / "configs" / "exp.yaml").read_text(encoding="utf-8"))
    t, x = data.hourly(data.read_stdmet(ENV / "data" / "46042h2023.txt.gz"), cfg["data"]["features"])
    splits = data.split_series(t, x)
    val_len = len(splits["val"][0])
    test_start = len(splits["train"][0]) + val_len
    start = test_start - val_len // 2
    splits["val"] = (t[start:start + val_len], x[start:start + val_len])
    return splits


if __name__ == "__main__":
    out = ENV / "data" / "splits"
    for name, (ts, xs) in build().items():
        data.save_split(out, name, ts, xs)
        print(name, len(ts), data.fingerprint(ts, xs))
