"""Poisson regressions of pickups on weather, hourly and daily, for Manhattan, airports and outer boroughs."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from taxi import config
from taxi.features import load_panel

GROUPS = {"Yellow Zone": "Manhattan core", "Airports": "Airports", "Boro Zone": "Outer boroughs"}
TEMP_BINS = [-np.inf, 0, 10, 20, 28, np.inf]
TEMP_LABELS = ["freezing", "cold", "mild", "warm", "hot"]


def _hourly(panel):
    df = panel[panel.service_zone.isin(GROUPS) & ~panel.is_dst_shift].copy()
    df["group"] = df.service_zone.map(GROUPS)
    g = df.groupby(["group", "hour_at"], as_index=False).agg(
        pickups=("pickups", "sum"), precip_mm=("precip_mm", "first"), temp_c=("temp_c", "first"),
        is_snow=("is_snow", "first"), hour_of_week=("hour_of_week", "first"), is_holiday=("is_holiday", "first"))
    g["rain"] = pd.cut(g.precip_mm, [-np.inf, 0, 1, 4, np.inf],
                       labels=["dry", "light (<=1mm)", "moderate (1-4mm)", "heavy (>4mm)"]).astype(str)
    g["temp_band"] = pd.cut(g.temp_c, TEMP_BINS, labels=TEMP_LABELS).astype(str)
    g["ym"] = g.hour_at.dt.strftime("%Y-%m")
    g["day"] = g.hour_at.dt.normalize()
    return g


def _effects(res, **info):
    ci = res.conf_int()
    rows = []
    for name in res.params.index:
        if name.startswith("C(rain") or name == "is_snow":
            rows.append({
                **info,
                "term": "rain" if name.startswith("C(rain") else "snow",
                "level": name.split("[T.")[-1].rstrip("]") if "[T." in name else "any snow",
                "pct_change": 100 * (np.exp(res.params[name]) - 1),
                "ci_low": 100 * (np.exp(ci.loc[name, 0]) - 1),
                "ci_high": 100 * (np.exp(ci.loc[name, 1]) - 1),
                "p_value": res.pvalues[name],
            })
    return rows


def fit_hourly(hourly):
    rows = []
    for group, g in hourly.groupby("group"):
        g = g.reset_index(drop=True)
        res = smf.glm("pickups ~ C(rain, Treatment('dry')) + is_snow + C(temp_band, Treatment('mild'))"
                      " + C(hour_of_week) + C(ym) + is_holiday", data=g, family=sm.families.Poisson()
                      ).fit(cov_type="cluster", cov_kwds={"groups": g["day"].factorize()[0]})
        rows += _effects(res, group=group, granularity="hourly", n=len(g))
    return rows


def fit_daily(hourly):
    rows = []
    for group, g in hourly.groupby("group"):
        d = g.groupby("day", as_index=False).agg(pickups=("pickups", "sum"), precip_mm=("precip_mm", "sum"),
                                                 is_snow=("is_snow", "max"), temp_c=("temp_c", "mean"),
                                                 is_holiday=("is_holiday", "max"))
        # bins scaled up since a day has more rain than an hour
        d["rain"] = pd.cut(d.precip_mm, [-np.inf, 0, 5, 15, np.inf],
                           labels=["dry", "light (<=5mm)", "moderate (5-15mm)", "heavy (>15mm)"]).astype(str)
        d["temp_band"] = pd.cut(d.temp_c, TEMP_BINS, labels=TEMP_LABELS).astype(str)
        d["dow"] = d.day.dt.dayofweek
        d["ym"] = d.day.dt.strftime("%Y-%m")
        res = smf.glm("pickups ~ C(rain, Treatment('dry')) + is_snow + C(temp_band, Treatment('mild'))"
                      " + C(dow) + C(ym) + is_holiday", data=d, family=sm.families.Poisson()
                      ).fit(cov_type="HC1")
        rows += _effects(res, group=group, granularity="daily", n=len(d))
    return rows


def run(db_path=None, out_dir=None):
    out_dir = out_dir or config.ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly = _hourly(load_panel(db_path))
    effects = pd.DataFrame(fit_hourly(hourly) + fit_daily(hourly))
    effects.to_csv(out_dir / "weather_effects.csv", index=False)

    # raw average by hour of day for the dashboard chart
    m = hourly[hourly.group == "Manhattan core"].copy()
    m["hour_of_day"] = m.hour_at.dt.hour
    m["wet"] = np.where(m.precip_mm >= 1, "rain >= 1mm", "dry")
    m.groupby(["hour_of_day", "wet"], as_index=False).pickups.mean().to_csv(out_dir / "rain_hour_profile.csv", index=False)

    effects["effect"] = [f"{r.pct_change:+.1f}% [{r.ci_low:+.1f}, {r.ci_high:+.1f}]" for r in effects.itertuples()]
    print(effects.pivot_table(index=["group", "term", "level"], columns="granularity",
                              values="effect", aggfunc="first").to_string())
