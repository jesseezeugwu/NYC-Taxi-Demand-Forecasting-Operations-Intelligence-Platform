import numpy as np
import pandas as pd
import pytest

from taxi.features import build_features, feature_columns, tidy_panel


def make_panel(n_zones=3, n_hours=24 * 35, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    rows = []
    for z in range(1, n_zones + 1):
        rows.append(pd.DataFrame({
            "zone_id": z,
            "hour_at": hours,
            "borough": "Manhattan",
            "pickups": rng.poisson(20, n_hours),
            "dropoffs": rng.poisson(15, n_hours),
            "temp_c": rng.normal(10, 5, n_hours),
            "precip_mm": rng.exponential(0.2, n_hours),
            "wind_ms": rng.normal(3, 1, n_hours),
            "is_snow": rng.integers(0, 2, n_hours),
            "hour_of_day": hours.hour,
            "day_of_week": (hours.dayofweek + 1) % 7,
            "month": hours.month,
            "is_holiday": False,
        }))
    return tidy_panel(pd.concat(rows, ignore_index=True))


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_no_future_leakage(horizon):
    panel = make_panel()
    base = build_features(panel, horizon, weather="observed")
    cols = [c for c in feature_columns(base) if c != "zone_id"]

    # pick a target hour and scramble everything after its forecast origin
    target = base.hour_at.iloc[len(base) // 2]
    origin = target - pd.Timedelta(hours=horizon)
    scrambled = panel.copy()
    later = scrambled.hour_at > origin
    rng = np.random.default_rng(1)
    for c in ["pickups", "dropoffs"]:
        scrambled.loc[later, c] = rng.integers(0, 500, later.sum()).astype(np.int32)
    for c in ["temp_c", "precip_mm", "wind_ms"]:
        scrambled.loc[later, c] = rng.normal(0, 100, later.sum())

    after = build_features(scrambled, horizon, weather="observed")
    a = base[base.hour_at == target][cols].reset_index(drop=True)
    b = after[after.hour_at == target][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_oracle_weather_does_use_target_hour():
    panel = make_panel()
    base = build_features(panel, 1, weather="target")
    target = base.hour_at.iloc[len(base) // 2]
    changed = panel.copy()
    changed.loc[changed.hour_at == target, "precip_mm"] = 99.0
    after = build_features(changed, 1, weather="target")
    assert (after.loc[after.hour_at == target, "precip_mm"] == 99.0).all()
    assert not (base.loc[base.hour_at == target, "precip_mm"] == 99.0).any()


def test_lags_line_up_with_panel():
    panel = make_panel()
    f = build_features(panel, 3, weather="none")
    row = f.iloc[400]
    z = panel[panel.zone_id == row.zone_id].set_index("hour_at")["pickups"]
    t = row.hour_at
    assert row.y == z[t]
    assert row.lag_0 == z[t - pd.Timedelta(hours=3)]
    assert row.lag_week == z[t - pd.Timedelta(hours=168)]
    expected = np.mean([z[t - pd.Timedelta(hours=168 * k)] for k in range(1, 5)])
    assert row.how_mean_4w == pytest.approx(expected)
    assert row.mean_3h == pytest.approx(z[t - pd.Timedelta(hours=5): t - pd.Timedelta(hours=3)].mean())


def test_rejects_sparse_panel():
    panel = make_panel().iloc[:-1]
    with pytest.raises(ValueError):
        build_features(panel, 1)
