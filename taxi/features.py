"""Turn the zone-hour panel into model features.

Every row is a target hour T for one zone. For horizon h the forecast is
made at t = T - h, so anything built from pickups/dropoffs only looks at
hours <= t.
"""
import warnings

import duckdb
import numpy as np
import pandas as pd

from taxi import config

WEATHER_COLS = ["temp_c", "precip_mm", "wind_ms", "is_snow"]
CALENDAR_COLS = ["hour_of_day", "day_of_week", "month", "is_holiday", "hour_of_week"]


def load_panel(db_path=None, start=None, end=None):
    con = duckdb.connect(str(db_path or config.DB_PATH), read_only=True)
    where = []
    if start:
        where.append(f"hour_at >= '{start}'")
    if end:
        where.append(f"hour_at < '{end}'")
    sql = "select * from main.mart_zone_hour_demand"
    if where:
        sql += " where " + " and ".join(where)
    df = con.sql(sql + " order by zone_id, hour_at").df()
    con.close()
    return tidy_panel(df)


def tidy_panel(df):
    df = df.sort_values(["zone_id", "hour_at"]).reset_index(drop=True)
    for c in ["pickups", "dropoffs", "hour_of_day", "day_of_week", "month", "is_snow"]:
        df[c] = df[c].astype(np.int32)
    df["zone_id"] = df["zone_id"].astype(np.int32)
    df["is_holiday"] = df["is_holiday"].astype(np.int8)
    df["hour_of_week"] = (df["day_of_week"] * 24 + df["hour_of_day"]).astype(np.int16)
    return df


def _check_dense(df):
    n_zones = df["zone_id"].nunique()
    n_hours = df["hour_at"].nunique()
    if len(df) != n_zones * n_hours:
        raise ValueError("panel must be dense (every zone x every hour)")
    # the reshape below assumes zone then hour order
    z = df["zone_id"].to_numpy().reshape(n_zones, n_hours)
    h = df["hour_at"].to_numpy().reshape(n_zones, n_hours)
    ok = ((z == z[:, :1]).all() and (np.diff(z[:, 0]) > 0).all()
          and (h == h[:1]).all() and (np.diff(h[0]) > np.timedelta64(0)).all())
    if not ok:
        raise ValueError("panel must be sorted by zone_id, hour_at")
    return n_zones, n_hours


def _matrix(df, col, n_zones, n_hours):
    return df[col].to_numpy(dtype=np.float32).reshape(n_zones, n_hours)


def _shift(m, k):
    # value k hours earlier, nan where history runs out
    out = np.full_like(m, np.nan)
    if k < m.shape[1]:
        out[:, k:] = m[:, : m.shape[1] - k]
    return out


def _trailing_mean(m, k, window):
    # mean of the `window` hours ending k hours before each column
    csum = np.concatenate([np.zeros((m.shape[0], 1), dtype=np.float64), np.cumsum(m, axis=1, dtype=np.float64)], axis=1)
    n = m.shape[1]
    out = np.full(m.shape, np.nan, dtype=np.float32)
    end = np.arange(n) - k + 1          # exclusive end index into m
    start = end - window
    ok = start >= 0
    out[:, ok] = ((csum[:, end[ok]] - csum[:, start[ok]]) / window).astype(np.float32)
    return out


def build_features(df, horizon, weather="observed"):
    """Feature frame for one horizon.

    weather: "none", "observed" (weather at forecast time t), "rain" (same but rain/snow only), or
    "target" (actual weather at T, an upper bound for a perfect forecast).
    """
    if horizon < 1 or horizon > 24:
        raise ValueError("horizon must be 1..24")
    n_zones, n_hours = _check_dense(df)
    y = _matrix(df, "pickups", n_zones, n_hours)
    d = _matrix(df, "dropoffs", n_zones, n_hours)
    h = horizon

    feats = {
        "lag_0": _shift(y, h),          # latest known hour
        "lag_1": _shift(y, h + 1),
        "lag_2": _shift(y, h + 2),
        "lag_day": _shift(y, 24),       # same hour yesterday
        "lag_week": _shift(y, 168),     # same hour last week
        "lag_2week": _shift(y, 336),
        "mean_3h": _trailing_mean(y, h, 3),
        "mean_24h": _trailing_mean(y, h, 24),
        "mean_week": _trailing_mean(y, h, 168),
        "drop_0": _shift(d, h),
        "drop_mean_3h": _trailing_mean(d, h, 3),
    }
    # same hour of week, averaged over the last 4 weeks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # first week has no history
        feats["how_mean_4w"] = np.nanmean(np.stack([_shift(y, 168 * k) for k in range(1, 5)]), axis=0)

    out = pd.DataFrame({k: v.reshape(-1) for k, v in feats.items()})
    out["zone_id"] = df["zone_id"].to_numpy()
    out["hour_at"] = df["hour_at"].to_numpy()
    out["borough"] = df["borough"].to_numpy()
    out["is_dst_shift"] = df["is_dst_shift"].to_numpy() if "is_dst_shift" in df else False
    for c in CALENDAR_COLS:
        out[c] = df[c].to_numpy()
    out["horizon"] = h
    out["hour_idx"] = np.tile(np.arange(n_hours), n_zones)

    if weather == "observed":
        for c in WEATHER_COLS:
            out[c] = _shift(_matrix(df, c, n_zones, n_hours), h).reshape(-1)
        out["precip_3h"] = _trailing_mean(_matrix(df, "precip_mm", n_zones, n_hours), h, 3).reshape(-1)
    elif weather == "rain":
        # rain/snow only, no temperature, which can end up standing in for the season
        for c in ["precip_mm", "is_snow"]:
            out[c] = _shift(_matrix(df, c, n_zones, n_hours), h).reshape(-1)
        out["precip_3h"] = _trailing_mean(_matrix(df, "precip_mm", n_zones, n_hours), h, 3).reshape(-1)
    elif weather == "target":
        for c in WEATHER_COLS:
            out[c] = df[c].to_numpy()
        out["precip_3h"] = _trailing_mean(_matrix(df, "precip_mm", n_zones, n_hours), 0, 3).reshape(-1)
    elif weather != "none":
        raise ValueError(f"unknown weather option {weather}")

    out["y"] = df["pickups"].to_numpy()
    # need two weeks of history plus the horizon
    return out[out["lag_2week"].notna() & out["how_mean_4w"].notna()].reset_index(drop=True)


def feature_columns(frame):
    skip = {"y", "hour_at", "borough", "horizon", "hour_idx", "is_dst_shift"}
    return [c for c in frame.columns if c not in skip]
