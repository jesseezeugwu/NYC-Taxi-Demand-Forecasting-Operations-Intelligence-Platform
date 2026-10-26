from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DB_PATH = DATA / "taxi.duckdb"
SAMPLE_DIR = ROOT / "sample_data"
SAMPLE_DB_PATH = DATA / "sample.duckdb"
DBT_DIR = ROOT / "dbt"
ARTIFACTS = ROOT / "artifacts"
REPORT_DB = ROOT / "reports" / "sources" / "taxi" / "marts.duckdb"

# first and last month of trip data (inclusive)
FIRST_MONTH = "2024-01"
LAST_MONTH = "2026-05"

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
# NOAA Local Climatological Data v2, Central Park. The old v1 API stops in Aug 2025.
LCD_URL = "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/access/{year}/LCD_USW00094728_{year}.csv"


def months(first=FIRST_MONTH, last=LAST_MONTH):
    y, m = map(int, first.split("-"))
    ly, lm = map(int, last.split("-"))
    out = []
    while (y, m) <= (ly, lm):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out
