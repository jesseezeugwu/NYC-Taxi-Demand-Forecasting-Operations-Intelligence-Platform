import pandas as pd
import pytest

from taxi.ingest import parse_lcd


def lcd(rows):
    cols = ["DATE", "REPORT_TYPE", "HourlyDryBulbTemperature", "HourlyPrecipitation",
            "HourlyWindSpeed", "HourlyPresentWeatherType"]
    return pd.DataFrame(rows, columns=cols)


def test_specials_are_not_double_counted():
    out = parse_lcd(lcd([
        ["2025-02-08T20:20:00", "FM-16", "0", "1.0", "2", "-SN"],
        ["2025-02-08T20:51:00", "FM-15", "0", "3.6", "2", "-SN:03 |SN |SN"],
    ]))
    assert out.precip_mm.tolist() == [3.6]
    assert out.is_snow.tolist() == [1]


def test_trace_flags_and_blanks():
    out = parse_lcd(lcd([
        ["2025-06-01T00:51:00", "FM-15", "20.0", "T", "1", None],
        ["2025-06-01T01:51:00", "FM-15", "21.0s", "0.5s", "1", "RA"],
        ["2025-06-01T02:51:00", "FM-15", "22.0", None, "1", None],
    ]))
    assert out.precip_mm.tolist() == pytest.approx([0.05, 0.5, 0.0])
    assert out.temp_c.tolist() == [20.0, 21.0, 22.0]
    assert out.is_snow.sum() == 0


def test_fall_back_hour_collapses_and_gaps_fill():
    out = parse_lcd(lcd([
        ["2024-11-03T00:51:00", "FM-15", "10", "0", "1", None],
        ["2024-11-03T01:51:00", "FM-15", "10", "0.2", "1", None],
        ["2024-11-03T01:51:00", "FM-15", "12", "0.3", "1", None],   # repeated wall-clock hour
        ["2024-11-03T03:51:00", "FM-15", "14", "0", "1", None],     # 02:00 report missing
    ]))
    assert out.hour_at.is_unique
    assert len(out) == 4
    one = out.set_index("hour_at").loc[pd.Timestamp("2024-11-03 01:00")]
    assert one.precip_mm == pytest.approx(0.5)
    assert one.temp_c == 11
    two = out.set_index("hour_at").loc[pd.Timestamp("2024-11-03 02:00")]
    assert two.temp_c == pytest.approx(12.5)
    assert two.precip_mm == 0
