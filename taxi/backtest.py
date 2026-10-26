"""Rolling-origin backtest: one fold per test month, no random splits."""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

from taxi import config
from taxi.features import build_features, load_panel
from taxi.metrics import summarize, wape
from taxi.models.baselines import HourOfWeekMean, Persistence, SeasonalNaive
from taxi.models.gbm import LightGBMModel
from taxi.models.nn import NeuralForecaster

MODELS = ["persistence", "seasonal_naive", "how_mean", "lightgbm", "lightgbm_rain", "lightgbm_weather",
          "lightgbm_oracle", "mlp", "gru"]
FOLDS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
HORIZONS = [1, 6]
VALID_WEEKS = 4


def make_model(name, verbose):
    if name == "persistence":
        return Persistence()
    if name == "seasonal_naive":
        return SeasonalNaive()
    if name == "how_mean":
        return HourOfWeekMean()
    if name == "lightgbm":
        return LightGBMModel(weather="none")
    if name == "lightgbm_rain":
        return LightGBMModel(weather="rain")
    if name == "lightgbm_weather":
        return LightGBMModel(weather="observed")
    if name == "lightgbm_oracle":
        # actual weather at the target hour, i.e. a perfect weather forecast
        return LightGBMModel(weather="target")
    if name == "mlp":
        return NeuralForecaster(use_gru=False, weather="rain", verbose=verbose)
    if name == "gru":
        return NeuralForecaster(use_gru=True, weather="rain", verbose=verbose)
    raise ValueError(f"unknown model {name}")


def run(models=None, horizons=None, folds=None, db_path=None, out_dir=None, verbose=True, tag=None):
    # name the run after its models so a partial rerun can't overwrite the full one
    tag = tag or ("all" if models is None else "-".join(models))
    models = models or MODELS
    horizons = horizons or HORIZONS
    folds = folds or FOLDS
    out_dir = out_dir or config.ARTIFACTS
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    panel = load_panel(db_path)
    for ym in folds:
        test_start = pd.Timestamp(f"{ym}-01")
        test_end = test_start + pd.offsets.MonthBegin(1)
        valid_start = test_start - pd.Timedelta(weeks=VALID_WEEKS)
        preds, fits, importance, embeddings = [], [], {}, None

        for h in horizons:
            frames = {}
            for name in models:
                model = make_model(name, verbose)
                if model.weather not in frames:
                    f = build_features(panel, h, weather=model.weather)
                    frames[model.weather] = f[~f.is_dst_shift]
                frame = frames[model.weather]
                train = frame[frame.hour_at < valid_start]
                valid = frame[(frame.hour_at >= valid_start) & (frame.hour_at < test_start)]
                test = frame[(frame.hour_at >= test_start) & (frame.hour_at < test_end)]

                if name == "gru":
                    model.attach_panel(panel)
                start = time.time()
                model.fit(train, valid)
                seconds = time.time() - start
                yhat = model.predict(test)
                print(f"{ym} h={h} {name:<24} {seconds:6.0f}s  wape {wape(test.y, yhat):.4f}", flush=True)

                fits.append({"fold": ym, "horizon": h, "model": name, "weather": model.weather, "run_at": pd.Timestamp.now(),
                             "fit_seconds": round(seconds, 1),
                             "epochs_or_rounds": getattr(model, "best_epoch", getattr(model, "rounds", None))})
                p = test[["zone_id", "hour_at", "borough", "is_holiday", "y"]].copy()
                p["yhat"] = yhat.astype(np.float32)
                p["model"], p["horizon"], p["fold"] = name, h, ym
                p["weather"], p["run_at"] = model.weather, pd.Timestamp.now()
                preds.append(p)
                if name == "lightgbm":
                    importance[f"h{h}"] = model.importance()
                if name == "mlp" and h == 1:
                    embeddings = model.zone_embeddings()

        pd.concat(preds).to_parquet(runs / f"{ym}_{tag}.parquet", index=False)
        pd.DataFrame(fits).to_csv(runs / f"{ym}_{tag}_fits.csv", index=False)
        if importance:
            pd.DataFrame(importance).rename_axis("feature").to_csv(runs / f"{ym}_importance.csv")
        if embeddings is not None:
            embeddings.to_csv(runs / f"{ym}_zone_embeddings.csv")


def block_bootstrap_wape(df, a, b, n_boot=1000, seed=0):
    """95% CI for WAPE(a) - WAPE(b), resampling whole days so hours in a day stay together."""
    wide = df[df.model.isin([a, b])].pivot_table(index=["zone_id", "hour_at"], columns="model",
                                                  values="yhat", observed=True)
    y = df[df.model == a].set_index(["zone_id", "hour_at"])["y"].reindex(wide.index)
    days = wide.index.get_level_values("hour_at").normalize()
    daily = pd.DataFrame({
        "err_a": np.abs(y - wide[a]).to_numpy(),
        "err_b": np.abs(y - wide[b]).to_numpy(),
        "y": y.to_numpy(),
        "day": days,
    })
    if daily[["err_a", "err_b"]].isna().any().any():
        raise ValueError(f"{a} and {b} don't cover the same hours")
    daily = daily.groupby("day").sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    ea, eb, yy = (daily[c].to_numpy()[idx].sum(axis=1) for c in ["err_a", "err_b", "y"])
    diff = (ea - eb) / yy
    point = (daily.err_a.sum() - daily.err_b.sum()) / daily.y.sum()
    return point, np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def _segments(preds, panel, first_fold):
    wx = panel[["zone_id", "hour_at", "precip_mm"]]
    preds = preds.merge(wx, on=["zone_id", "hour_at"], how="left")
    preds["rain"] = np.where(preds.precip_mm >= 1, "rain >= 1mm", np.where(preds.precip_mm > 0, "light rain", "dry"))
    preds["holiday"] = np.where(preds.is_holiday == 1, "holiday", "normal day")
    # volume tiers from history before the first test month
    hist = panel[panel.hour_at < pd.Timestamp(f"{first_fold}-01")].groupby("zone_id").pickups.mean()
    tiers = pd.qcut(hist, 3, labels=["low volume", "mid volume", "high volume"]).astype(str)
    preds["volume_tier"] = preds.zone_id.map(tiers)
    return preds


def report(db_path=None, out_dir=None):
    out_dir = out_dir or config.ARTIFACTS
    runs = out_dir / "runs"
    # later runs of the same model/fold/horizon replace earlier ones
    files = sorted(runs.glob("20??-??_*.parquet"))
    if not files:
        raise SystemExit("no backtest runs found, run `python -m taxi backtest` first")
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    latest = preds.groupby(["fold", "horizon", "model"]).run_at.transform("max")
    preds = preds[preds.run_at == latest].drop(columns="run_at")
    if preds.duplicated(["fold", "horizon", "model", "zone_id", "hour_at"]).any():
        raise SystemExit("duplicate predictions after picking the latest runs")
    # catches a month still using an old run with different inputs
    mixed = preds.groupby("model").weather.nunique()
    if (mixed > 1).any():
        raise SystemExit(f"models mix weather settings across months: {list(mixed[mixed > 1].index)}")
    preds = preds.drop(columns="weather")
    folds = sorted(preds.fold.unique())
    preds = _segments(preds, load_panel(db_path), folds[0])
    (out_dir / "predictions").mkdir(exist_ok=True)
    preds.to_parquet(out_dir / "predictions" / "backtest.parquet", index=False)

    overall = summarize(preds, ["model", "horizon"])
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    summarize(preds, ["model", "horizon", "fold"]).to_csv(out_dir / "metrics_by_fold.csv", index=False)
    segs = []
    for col in ["borough", "volume_tier", "rain", "holiday"]:
        s = summarize(preds, ["model", "horizon", col]).rename(columns={col: "segment"})
        segs.append(s.assign(segment_type=col))
    pd.concat(segs).to_csv(out_dir / "metrics_by_segment.csv", index=False)
    fits = pd.concat([pd.read_csv(f) for f in sorted(runs.glob("*_fits.csv"))])
    fits = fits.sort_values("run_at").drop_duplicates(["fold", "horizon", "model"], keep="last")
    fits.to_csv(out_dir / "fit_times.csv", index=False)

    last = folds[-1]
    if (runs / f"{last}_importance.csv").exists():
        pd.read_csv(runs / f"{last}_importance.csv").to_csv(out_dir / "lightgbm_importance.csv", index=False)
    if (runs / f"{last}_zone_embeddings.csv").exists():
        pd.read_csv(runs / f"{last}_zone_embeddings.csv").to_csv(out_dir / "zone_embeddings.csv", index=False)

    # is the gap between models bigger than day-to-day noise?
    pairs = [("lightgbm", "seasonal_naive"), ("lightgbm", "how_mean"), ("lightgbm_rain", "lightgbm"),
             ("lightgbm_weather", "lightgbm"), ("lightgbm_oracle", "lightgbm"),
             ("mlp", "lightgbm"), ("gru", "lightgbm"), ("gru", "mlp")]
    rows = []
    models = set(preds.model)
    for h in sorted(preds.horizon.unique()):
        for a, b in pairs:
            if a in models and b in models:
                for segment, part in [("all hours", preds), ("rain >= 1mm", preds[preds.rain == "rain >= 1mm"])]:
                    point, lo, hi = block_bootstrap_wape(part[part.horizon == h], a, b)
                    rows.append({"horizon": h, "model_a": a, "model_b": b, "hours": segment,
                                 "wape_diff": point, "ci_low": lo, "ci_high": hi})
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "model_comparisons.csv", index=False)

    # do the Poisson means give usable intervals? 80% interval coverage by zone volume
    trained = preds[~preds.model.isin(["persistence", "seasonal_naive", "how_mean"])].copy()
    lam = trained.yhat.clip(lower=1e-6)
    trained["inside"] = (trained.y >= poisson.ppf(0.1, lam)) & (trained.y <= poisson.ppf(0.9, lam))
    cov = trained.groupby(["model", "horizon", "volume_tier"], as_index=False).inside.mean()
    cov.rename(columns={"inside": "coverage_80"}).to_csv(out_dir / "interval_coverage.csv", index=False)

    (out_dir / "backtest_config.json").write_text(json.dumps(
        {"folds": folds, "horizons": sorted(int(h) for h in preds.horizon.unique()),
         "valid_weeks": VALID_WEEKS, "models": sorted(models)}, indent=2))

    print(overall.pivot(index="model", columns="horizon", values="wape").round(4).to_string())
    print()
    print(comp.round(4).to_string(index=False))
    return overall
