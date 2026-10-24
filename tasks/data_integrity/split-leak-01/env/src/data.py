"""NDBC standard meteorological files -> hourly series -> chronological splits -> windows."""
import gzip
import hashlib
from pathlib import Path

import numpy as np

# NDBC fills missing values with all-nines of the field width.
MISSING = {
    "WDIR": 999, "WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "DPD": 99.0,
    "APD": 99.0, "MWD": 999, "PRES": 9999.0, "ATMP": 999.0, "WTMP": 999.0,
    "DEWP": 999.0, "VIS": 99.0, "TIDE": 99.0,
}
FRACTIONS = (0.7, 0.15, 0.15)


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
    """Hourly means of each feature, gaps forward-filled, leading empty hours dropped.

    Returns (t, x): t is hours since 1970-01-01 UTC (int64), x is (hours, features).
    """
    stamps = [
        f"{int(y):04d}-{int(m):02d}-{int(d):02d}T{int(h):02d}"
        for y, m, d, h in zip(cols["YY"], cols["MM"], cols["DD"], cols["hh"])
    ]
    t_obs = np.array(stamps, dtype="datetime64[h]")
    hours = (t_obs - t_obs.min()).astype(int)
    n = hours.max() + 1

    x = np.full((n, len(features)), np.nan)
    for j, name in enumerate(features):
        v = cols[name]
        ok = ~np.isnan(v)
        total = np.zeros(n)
        count = np.zeros(n)
        np.add.at(total, hours[ok], v[ok])
        np.add.at(count, hours[ok], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            x[:, j] = total / count

    for j in range(x.shape[1]):
        col = x[:, j]
        last = np.where(~np.isnan(col), np.arange(n), 0)
        np.maximum.accumulate(last, out=last)
        x[:, j] = col[last]

    first = int(np.argmax(~np.isnan(x).any(axis=1)))
    t = t_obs.min().astype("int64") + np.arange(n, dtype=np.int64)
    return t[first:], x[first:]


def split_series(t, x, fractions=FRACTIONS):
    """Chronological train/val/test split: contiguous, non-overlapping, in time order."""
    n = len(t)
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return {"train": (t[:a], x[:a]), "val": (t[a:b], x[a:b]), "test": (t[b:], x[b:])}


def save_split(directory, name, t, x):
    np.savez(Path(directory) / f"{name}.npz", t=t, x=x)


def load_split(directory, name):
    with np.load(Path(directory) / f"{name}.npz") as f:
        return f["t"], f["x"]


def fingerprint(t, x):
    """SHA-256 of a split's contents, recorded with every run."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(t, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    return h.hexdigest()


def windows(x, mean, std, target_idx, window):
    """Inputs: the previous `window` hours; target: the next hour's value."""
    z = (x - mean) / std
    X = np.stack([z[i:i + window] for i in range(len(z) - window)]).astype(np.float32)
    y = z[window:, target_idx].astype(np.float32)
    return X, y
