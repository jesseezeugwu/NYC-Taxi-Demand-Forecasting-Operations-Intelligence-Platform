import numpy as np
import pandas as pd


def wape(y, yhat):
    """Total absolute error over total demand. Doesn't blow up on zero-count hours like MAPE."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return np.abs(y - yhat).sum() / max(y.sum(), 1e-9)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def bias(y, yhat):
    # positive means over-forecasting
    y = np.asarray(y, dtype=float)
    return float((np.asarray(yhat, float) - y).sum() / max(y.sum(), 1e-9))


def summarize(df, by, y="y", yhat="yhat"):
    rows = []
    for key, g in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(by, key)),
            "n": len(g),
            "demand": g[y].sum(),
            "wape": wape(g[y], g[yhat]),
            "mae": mae(g[y], g[yhat]),
            "rmse": rmse(g[y], g[yhat]),
            "bias": bias(g[y], g[yhat]),
        })
    return pd.DataFrame(rows)
