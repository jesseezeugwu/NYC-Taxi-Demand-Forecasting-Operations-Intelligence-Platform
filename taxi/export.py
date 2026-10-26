"""Write the small tables the Evidence dashboard reads (reports/sources/taxi/marts.duckdb)."""
import duckdb
import numpy as np
import pandas as pd

from taxi import config
from taxi.features import load_panel

SHOWCASE_ZONES = [161, 138, 132, 79, 230, 181]
SHOWCASE_MODELS = ["seasonal_naive", "lightgbm", "mlp", "gru"]


def demand_pressure(preds, panel, fold):
    """Forecast pickups per cab that dropped off the hour before. No driver IDs, so it's rough."""
    # fares from before the test month only
    start = pd.Timestamp(f"{fold}-01")
    hist = panel[panel.hour_at < start]
    fares = hist.groupby(["zone_id", "day_of_week", "hour_of_day"], as_index=False)[["earnings", "pickups"]].sum()
    fares["earnings_per_trip"] = fares.earnings / fares.pickups.replace(0, np.nan)

    p = preds[(preds.model == "lightgbm") & (preds.horizon == 1) & (preds.fold == fold)]
    prev = panel[["zone_id", "hour_at", "dropoffs"]].copy()
    prev["hour_at"] += pd.Timedelta(hours=1)
    p = p.merge(prev, on=["zone_id", "hour_at"])
    p["day_of_week"] = (p.hour_at.dt.dayofweek + 1) % 7
    p["hour_of_day"] = p.hour_at.dt.hour
    slots = p.groupby(["zone_id", "day_of_week", "hour_of_day"], as_index=False).agg(
        forecast_pickups=("yhat", "mean"), dropoffs_prev_hour=("dropoffs", "mean"))
    slots = slots.merge(fares[["zone_id", "day_of_week", "hour_of_day", "earnings_per_trip"]],
                        on=["zone_id", "day_of_week", "hour_of_day"], how="left")
    slots["pickups_per_arriving_cab"] = slots.forecast_pickups / (slots.dropoffs_prev_hour + 1)
    slots["dollars_per_arriving_cab"] = slots.pickups_per_arriving_cab * slots.earnings_per_trip
    return slots


def readme_figure(preds, zone_name, zone_id, out=config.ROOT / "docs" / "forecast.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = sorted(preds.fold.unique())[-1]
    p = preds[(preds.fold == last) & (preds.horizon == 1) & (preds.zone_id == zone_id)]
    start = p.hour_at.min() + pd.Timedelta(days=7 - p.hour_at.min().dayofweek)  # first Monday
    p = p[(p.hour_at >= start) & (p.hour_at < start + pd.Timedelta(days=7))]
    wide = p.pivot_table(index=["hour_at", "y"], columns="model", values="yhat").reset_index()

    folds = pd.read_csv(config.ARTIFACTS / "metrics_by_fold.csv")
    folds = folds[folds.horizon == 1].pivot(index="fold", columns="model", values="wape")

    ink, muted, grid = "#0b0b0b", "#898781", "#e6e5e1"
    colors = {"lightgbm": "#2a78d6", "seasonal_naive": "#eb6834"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)

    a.plot(wide.hour_at, wide.y, color=ink, linewidth=2, label="actual")
    a.plot(wide.hour_at, wide.lightgbm, color=colors["lightgbm"], linewidth=2, label="LightGBM")
    a.plot(wide.hour_at, wide.seasonal_naive, color=colors["seasonal_naive"], linewidth=2,
           linestyle=(0, (4, 2)), label="same hour last week")
    a.set_title(f"{zone_name}, 1 hour ahead, {start:%b %d}-{start + pd.Timedelta(days=6):%d %Y}",
                color=ink, fontsize=11, loc="left")
    a.set_ylabel("pickups per hour", color=muted, fontsize=9)
    a.set_ylim(0, wide[["y", "lightgbm", "seasonal_naive"]].max().max() * 1.2)  # room for the legend
    a.legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper left", ncol=3)
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a"))

    x = range(len(folds))
    for m, label in [("lightgbm", "LightGBM"), ("seasonal_naive", "same hour last week")]:
        b.plot(x, folds[m] * 100, color=colors[m], linewidth=2, marker="o", markersize=5, label=label)
        b.annotate(f"{folds[m].iloc[-1] * 100:.1f}%", (len(folds) - 1, folds[m].iloc[-1] * 100),
                   xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=ink)
    b.set_xticks(list(x))
    b.set_xticklabels([pd.Timestamp(f + "-01").strftime("%b") for f in folds.index])
    b.set_ylim(0, None)
    b.set_title("WAPE by test month, 1 hour ahead (%)", color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def run(db_path=None):
    art = config.ARTIFACTS
    preds = pd.read_parquet(art / "predictions" / "backtest.parquet")
    last_fold = sorted(preds.fold.unique())[-1]

    con = duckdb.connect(str(db_path or config.DB_PATH), read_only=True)
    zones = con.sql("select zone_id, zone_name, borough, service_zone from dim_zones").df()
    con.close()
    panel = load_panel(db_path)

    tables = {
        "model_scores": pd.read_csv(art / "metrics_overall.csv"),
        "model_scores_by_fold": pd.read_csv(art / "metrics_by_fold.csv"),
        "model_scores_by_segment": pd.read_csv(art / "metrics_by_segment.csv"),
        "model_comparisons": pd.read_csv(art / "model_comparisons.csv"),
        "lightgbm_importance": pd.read_csv(art / "lightgbm_importance.csv"),
        "weather_effects": pd.read_csv(art / "weather_effects.csv"),
        "rain_hour_profile": pd.read_csv(art / "rain_hour_profile.csv"),
    }

    show = preds[(preds.fold == last_fold) & (preds.horizon == 1) & preds.model.isin(SHOWCASE_MODELS)
                 & preds.zone_id.isin(SHOWCASE_ZONES)]
    cutoff = show.hour_at.min() + pd.Timedelta(days=14)
    show = show[show.hour_at < cutoff]
    wide = show.pivot_table(index=["zone_id", "hour_at", "y"], columns="model", values="yhat").reset_index()
    tables["forecast_showcase"] = wide.merge(zones[["zone_id", "zone_name"]], on="zone_id")

    emb = pd.read_csv(art / "zone_embeddings.csv", index_col="zone_id")
    x = emb.to_numpy() - emb.to_numpy().mean(axis=0)
    xy = x @ np.linalg.svd(x, full_matrices=False)[2][:2].T
    tables["zone_embeddings_2d"] = pd.DataFrame({"zone_id": emb.index, "pc1": xy[:, 0], "pc2": xy[:, 1]}).merge(zones)

    slots = demand_pressure(preds, panel, last_fold)
    tables["demand_pressure"] = slots.merge(zones[["zone_id", "zone_name", "borough"]], on="zone_id")

    config.REPORT_DB.unlink(missing_ok=True)
    out = duckdb.connect(str(config.REPORT_DB))
    for name, df in tables.items():
        out.register("df", df)
        out.execute(f"create table {name} as select * from df")
        out.unregister("df")
        print(f"{name}: {len(df):,} rows")
    out.close()
    print("wrote", readme_figure(preds, "Midtown Center", 161))
