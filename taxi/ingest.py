"""Download raw files and load them into DuckDB."""
import duckdb
import pandas as pd
import requests

from taxi import config


def _fetch(url, dest, timeout=300):
    if dest.exists():
        print(f"skip {dest.name}")
        return
    print(f"downloading {dest.name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def download_trips(months=None):
    config.RAW.mkdir(parents=True, exist_ok=True)
    for ym in months or config.months():
        _fetch(config.TRIP_URL.format(ym=ym), config.RAW / f"yellow_tripdata_{ym}.parquet")


def download_zones():
    config.RAW.mkdir(parents=True, exist_ok=True)
    _fetch(config.ZONES_URL, config.RAW / "taxi_zone_lookup.csv")


def download_weather():
    config.RAW.mkdir(parents=True, exist_ok=True)
    first_year = int(config.FIRST_MONTH[:4])
    last_year = int(config.LAST_MONTH[:4])
    for year in range(first_year, last_year + 1):
        _fetch(config.LCD_URL.format(year=year), config.RAW / f"lcd_central_park_{year}.csv")


def _num(series, trace=0.05):
    # LCD values look like "1.3", "T" (trace), "0.5s" (suspect flag), or blank
    s = series.astype(str).str.strip().replace({"T": str(trace), "Ts": str(trace)})
    s = s.str.replace(r"[a-zA-Z*]+$", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def parse_lcd(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw LCD v2 rows into one row per local clock hour.

    v2 files are metric and stamped in wall-clock time, same as TLC.
    Uses routine FM-15 reports only. Specials (FM-16) repeat the running
    precip total, so summing them would double count.
    """
    df = raw[raw["REPORT_TYPE"].astype(str).str.strip() == "FM-15"].copy()
    df["hour_at"] = pd.to_datetime(df["DATE"]).dt.floor("h")

    wx = df["HourlyPresentWeatherType"].fillna("").astype(str)
    out = pd.DataFrame({
        "hour_at": df["hour_at"],
        "temp_c": _num(df["HourlyDryBulbTemperature"]),
        # zero-precip hours are blank from late 2025 on
        "precip_mm": _num(df["HourlyPrecipitation"]).fillna(0.0),
        "wind_ms": _num(df["HourlyWindSpeed"]),
        "is_snow": wx.str.contains("SN", regex=False).astype(int),
    })

    # the fall-back hour shows up twice
    hourly = out.groupby("hour_at", as_index=False).agg(
        temp_c=("temp_c", "mean"),
        precip_mm=("precip_mm", "sum"),
        wind_ms=("wind_ms", "mean"),
        is_snow=("is_snow", "max"),
    )

    # fill the odd missing report so every hour has a row
    full = pd.DataFrame({"hour_at": pd.date_range(hourly.hour_at.min(), hourly.hour_at.max(), freq="h")})
    hourly = full.merge(hourly, on="hour_at", how="left")
    for col in ["temp_c", "wind_ms"]:
        hourly[col] = hourly[col].interpolate(limit=6, limit_direction="both")
        # longer gaps (3 days from 2025-08-30) get that hour's average over the surrounding 2 weeks
        by_hour = hourly.groupby(hourly.hour_at.dt.hour)[col]
        typical = by_hour.transform(lambda s: s.rolling(15, center=True, min_periods=1).mean())
        hourly[col] = hourly[col].fillna(typical)
    hourly["precip_mm"] = hourly["precip_mm"].fillna(0.0)
    hourly["is_snow"] = hourly["is_snow"].fillna(0).astype(int)
    return hourly


def build_weather_csv():
    files = sorted(config.RAW.glob("lcd_central_park_*.csv"))
    if not files:
        raise FileNotFoundError("no LCD files, run `python -m taxi download` first")
    raw = pd.concat([pd.read_csv(f, low_memory=False, dtype=str) for f in files], ignore_index=True)
    hourly = parse_lcd(raw)
    dest = config.RAW / "weather_hourly.csv"
    hourly.to_csv(dest, index=False)
    print(f"wrote {dest.name}: {len(hourly):,} hours")
    return dest


def load(db_path=None, trips_glob=None, weather_csv=None, zones_csv=None):
    db_path = db_path or config.DB_PATH
    trips_glob = trips_glob or str(config.RAW / "yellow_tripdata_*.parquet")
    weather_csv = weather_csv or config.RAW / "weather_hourly.csv"
    zones_csv = zones_csv or config.RAW / "taxi_zone_lookup.csv"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("create schema if not exists raw")
    # 2025 files added cbd_congestion_fee, so union by name
    con.execute(f"""
        create or replace table raw.yellow_tripdata as
        select * from read_parquet('{trips_glob}', union_by_name = true)
    """)
    con.execute(f"""
        create or replace table raw.weather_hourly as
        select * from read_csv('{weather_csv}', header = true)
    """)
    con.execute(f"""
        create or replace table raw.taxi_zones as
        select * from read_csv('{zones_csv}', header = true)
    """)
    for t in ["yellow_tripdata", "weather_hourly", "taxi_zones"]:
        n = con.execute(f"select count(*) from raw.{t}").fetchone()[0]
        print(f"raw.{t}: {n:,} rows")
    con.close()


def load_sample():
    s = config.SAMPLE_DIR
    load(
        db_path=config.SAMPLE_DB_PATH,
        trips_glob=str(s / "yellow_tripdata_sample.parquet").replace("\\", "/"),
        weather_csv=s / "weather_hourly.csv",
        zones_csv=s / "taxi_zone_lookup.csv",
    )
