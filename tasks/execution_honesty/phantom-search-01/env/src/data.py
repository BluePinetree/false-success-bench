"""NDBC standard meteorological files -> hourly features -> lagged design matrix."""
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


def lagged(series, target_idx, window):
    """Each row: the previous `window` hours of every feature, flattened; target: next hour."""
    X = np.stack([series[i - window:i].ravel() for i in range(window, len(series))])
    y = series[window:, target_idx]
    return X, y


def holdout(X, y, test_fraction):
    """Chronological split: the last `test_fraction` of rows are held out for testing."""
    cut = int(len(X) * (1 - test_fraction))
    return (X[:cut], y[:cut]), (X[cut:], y[cut:])
