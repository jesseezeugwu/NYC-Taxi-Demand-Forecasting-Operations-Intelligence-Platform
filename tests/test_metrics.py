import numpy as np
import pandas as pd
import pytest

from taxi.metrics import bias, mae, summarize, wape
from taxi.models.baselines import HourOfWeekMean, Persistence, SeasonalNaive


def test_wape_handles_zero_hours():
    y = np.array([0, 0, 10])
    yhat = np.array([1, 0, 9])
    assert wape(y, yhat) == pytest.approx(2 / 10)
    assert mae(y, yhat) == pytest.approx(2 / 3)


def test_bias_sign():
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)
    assert bias([10, 10], [8, 8]) == pytest.approx(-0.2)


def test_summarize_groups():
    df = pd.DataFrame({"g": ["a", "a", "b"], "y": [10, 10, 5], "yhat": [9, 11, 5]})
    s = summarize(df, ["g"]).set_index("g")
    assert s.loc["a", "wape"] == pytest.approx(0.1)
    assert s.loc["b", "wape"] == 0


def test_baselines_read_the_right_columns():
    f = pd.DataFrame({"lag_0": [1.0], "lag_week": [2.0], "how_mean_4w": [3.0]})
    assert Persistence().predict(f)[0] == 1
    assert SeasonalNaive().predict(f)[0] == 2
    assert HourOfWeekMean().predict(f)[0] == 3
