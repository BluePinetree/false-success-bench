"""NDBC standard meteorological files -> hourly features -> labelled windows."""
import gzip

import numpy as np

# NDBC fills missing values with all-nines of the field width.
MISSING = {
    "WDIR": 999, "WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "DPD": 99.0,
    "APD": 99.0, "MWD": 999, "PRES": 9999.0, "ATMP": 999.0, "WTMP": 999.0,
    "DEWP": 999.0, "VIS": 99.0, "TIDE": 99.0,
}


def read_stdmet(path):
    with gzip.open(path, "rt") as f:
        header = f.readline().lstrip("#").split()
        f.readline()  # units row
        rows = [line.split() for line in f if line.strip()]
    arr = np.array(rows, dtype=float)
    cols = {name: arr[:, i] for i, name in enumerate(header)}
    for name, sentinel in MISSING.items():
        if name in cols:
            cols[name] = np.where(cols[name] >= sentinel, np.nan, cols[name])
    return cols


def hourly(cols, features):
    """Average each feature per hour, forward-fill gaps, drop leading empty hours."""
    stamps = [
        f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(h):02d}"
        for y, m, d, h in zip(cols["YY"], cols["MM"], cols["DD"], cols["hh"])
    ]
    t = np.array(stamps, dtype="datetime64[h]")
    hours = (t - t.min()).astype(int)
    n = hours.max() + 1

    out = np.full((n, len(features)), np.nan)
    for j, name in enumerate(features):
        v = cols[name]
        ok = ~np.isnan(v)
        total = np.zeros(n)
        count = np.zeros(n)
        np.add.at(total, hours[ok], v[ok])
        np.add.at(count, hours[ok], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = total / count

    for j in range(out.shape[1]):
        col = out[:, j]
        last = np.where(~np.isnan(col), np.arange(n), 0)
        np.maximum.accumulate(last, out=last)
        out[:, j] = col[last]

    first = int(np.argmax(~np.isnan(out).any(axis=1)))
    return out[first:]


def alert_labels(wspd, threshold, horizon):
    """1 where wind speed reaches `threshold` at any of the next `horizon` hours."""
    n = len(wspd)
    labels = np.zeros(n, dtype=np.float32)
    for i in range(n - horizon):
        labels[i] = float(wspd[i + 1:i + 1 + horizon].max() >= threshold)
    return labels


def make_splits(series, labels, window, horizon, split):
    """Chronological train/val/test windows; normalisation fitted on train only."""
    n = len(series)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    bounds = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    mean = series[:n_train].mean(axis=0)
    std = series[:n_train].std(axis=0) + 1e-8
    z = (series - mean) / std

    splits = {}
    for name, (a, b) in bounds.items():
        ends = range(a + window - 1, b - horizon)
        X = np.stack([z[i - window + 1:i + 1] for i in ends]).astype(np.float32)
        y = labels[list(ends)]
        splits[name] = (X, y)
    return splits
