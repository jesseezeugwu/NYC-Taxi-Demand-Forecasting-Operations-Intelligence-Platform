"""Small committed sample for CI: 3% of trips touching 12 busy zones, Sept-Nov 2025."""
import shutil

import duckdb
import pandas as pd

from taxi import config

ZONES = [161, 237, 236, 162, 230, 186, 142, 170, 132, 138, 79, 48]
START, END = "2025-09-01", "2025-12-01"
# random thinning keeps counts Poisson, just smaller
FRACTION = 0.03
DBT_VARS = {
    "start_date": START,
    "end_date": END,
    "zone_selection_end": "2025-10-01",
    "min_pickups_per_hour": 1.5,
}
# CI fails if lightgbm doesn't beat these
MUST_BEAT = ["persistence", "seasonal_naive", "how_mean"]


def make():
    config.SAMPLE_DIR.mkdir(exist_ok=True)
    zones = ",".join(map(str, ZONES))
    dest = (config.SAMPLE_DIR / "yellow_tripdata_sample.parquet").as_posix()
    con = duckdb.connect(str(config.DB_PATH), read_only=True)
    con.execute("select setseed(0.42)")
    con.execute(f"""
        copy (
            select * from raw.yellow_tripdata
            where tpep_pickup_datetime >= '{START}' and tpep_pickup_datetime < '{END}'
              and (PULocationID in ({zones}) or DOLocationID in ({zones}))
              and random() < {FRACTION}
        ) to '{dest}' (format parquet, compression zstd)
    """)
    n = con.execute(f"select count(*) from '{dest}'").fetchone()[0]
    con.close()

    w = pd.read_csv(config.RAW / "weather_hourly.csv", parse_dates=["hour_at"])
    w = w[(w.hour_at >= START) & (w.hour_at < END)]
    w.to_csv(config.SAMPLE_DIR / "weather_hourly.csv", index=False)
    pd.read_csv(config.RAW / "taxi_zone_lookup.csv").to_csv(config.SAMPLE_DIR / "taxi_zone_lookup.csv", index=False)
    print(f"sample: {n:,} trips, {len(w):,} weather hours")


def run():
    from taxi import backtest, ingest
    from taxi.cli import run_dbt

    ingest.load_sample()
    run_dbt(["deps"], target="sample")
    run_dbt(["build"], target="sample", extra_vars=DBT_VARS)

    out = config.ARTIFACTS / "sample"
    shutil.rmtree(out / "runs", ignore_errors=True)
    backtest.run(models=MUST_BEAT + ["lightgbm"], horizons=[1], folds=["2025-11"],
                 db_path=config.SAMPLE_DB_PATH, out_dir=out, verbose=False)
    overall = backtest.report(db_path=config.SAMPLE_DB_PATH, out_dir=out)
    wape = overall[overall.horizon == 1].set_index("model")["wape"]
    worse = [m for m in MUST_BEAT if wape["lightgbm"] >= wape[m]]
    if worse:
        raise SystemExit(f"lightgbm ({wape['lightgbm']:.4f}) didn't beat {worse}")
    print("lightgbm beat the baselines")
