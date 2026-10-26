"""Recheck LightGBM settings on Oct-Nov 2025, before the test months.

Lowest pooled WAPE wins, but keep the current setup if it's within 0.1 points.
"""
import json

import pandas as pd

from taxi import config
from taxi.backtest import HORIZONS, VALID_WEEKS
from taxi.features import build_features, load_panel
from taxi.metrics import wape
from taxi.models.gbm import LightGBMModel

TUNE_FOLDS = ["2025-10", "2025-11"]
CURRENT = {"name": "current", "weather": "none", "params": {}}
PARAM_OPTIONS = [
    CURRENT,
    {"name": "lr_0.05", "weather": "none", "params": {"learning_rate": 0.05}},
    {"name": "leaves_63", "weather": "none", "params": {"num_leaves": 63}},
    {"name": "leaves_255", "weather": "none", "params": {"num_leaves": 255}},
    {"name": "min_leaf_500", "weather": "none", "params": {"min_data_in_leaf": 500}},
]
WEATHER_OPTIONS = ["rain", "observed"]
TIE = 0.001


def score(option, panel, frames):
    rows = []
    for h in HORIZONS:
        key = (h, option["weather"])
        if key not in frames:
            f = build_features(panel, h, weather=option["weather"])
            frames[key] = f[~f.is_dst_shift]
        frame = frames[key]
        for ym in TUNE_FOLDS:
            test_start = pd.Timestamp(f"{ym}-01")
            valid_start = test_start - pd.Timedelta(weeks=VALID_WEEKS)
            test = frame[(frame.hour_at >= test_start) & (frame.hour_at < test_start + pd.offsets.MonthBegin(1))]
            model = LightGBMModel(weather=option["weather"], params=option["params"])
            model.fit(frame[frame.hour_at < valid_start],
                      frame[(frame.hour_at >= valid_start) & (frame.hour_at < test_start)])
            yhat = model.predict(test)
            print(f"{option['name']:<14} {option['weather']:<9} h={h} {ym} rounds {model.rounds:5d} "
                  f"wape {wape(test.y, yhat):.4f}", flush=True)
            rows.append({"option": option["name"], "weather": option["weather"], "horizon": h, "fold": ym,
                         "rounds": model.rounds, "abs_err": float(abs(test.y - yhat).sum()), "demand": float(test.y.sum())})
    return rows


def pooled(rows, name):
    r = pd.DataFrame(rows)
    r = r[r.option == name]
    return r.abs_err.sum() / r.demand.sum()


def run(db_path=None, out_dir=None):
    out_dir = out_dir or config.ARTIFACTS / "tune"
    out_dir.mkdir(parents=True, exist_ok=True)
    # nothing after the last tune month is loaded
    panel = load_panel(db_path, end=str(pd.Timestamp(f"{TUNE_FOLDS[-1]}-01") + pd.offsets.MonthBegin(1)))
    frames, rows = {}, []

    for option in PARAM_OPTIONS:
        rows += score(option, panel, frames)
    best = min(PARAM_OPTIONS, key=lambda o: pooled(rows, o["name"]))
    if pooled(rows, "current") - pooled(rows, best["name"]) < TIE:
        best = CURRENT

    for weather in WEATHER_OPTIONS:
        option = {"name": f"{best['name']}+{weather}", "weather": weather, "params": best["params"]}
        rows += score(option, panel, frames)
    candidates = [best] + [{"name": f"{best['name']}+{w}", "weather": w, "params": best["params"]}
                           for w in WEATHER_OPTIONS]
    chosen = min(candidates, key=lambda o: pooled(rows, o["name"]))
    if pooled(rows, best["name"]) - pooled(rows, chosen["name"]) < TIE:
        chosen = best

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "tune_results.csv", index=False)
    summary = (results.groupby(["option", "weather"]).apply(lambda g: g.abs_err.sum() / g.demand.sum())
               .rename("pooled_wape").reset_index().sort_values("pooled_wape"))
    summary.to_csv(out_dir / "tune_summary.csv", index=False)
    decision = {"tune_folds": TUNE_FOLDS, "horizons": HORIZONS, "tie_points": TIE * 100,
                "chosen": chosen["name"], "weather": chosen["weather"], "params": chosen["params"],
                "same_as_current": chosen is CURRENT}
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    print(summary.round(4).to_string(index=False))
    print(json.dumps(decision, indent=2))
    return decision
