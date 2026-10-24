"""Build data/splits/{train,val,test}.npz from the raw NDBC file.

Chronological 70/15/15 split of the hourly series. The splits in data/splits/ were
produced by this script; rerun it if the raw data or the features change.
"""
import argparse

import yaml

from src import data

RAW = "data/46042h2023.txt.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t, x = data.hourly(data.read_stdmet(RAW), cfg["data"]["features"])
    for name, (ts, xs) in data.split_series(t, x).items():
        data.save_split(cfg["data"]["splits_dir"], name, ts, xs)
        print(f"{name}: {len(ts)} hours")


if __name__ == "__main__":
    main()
