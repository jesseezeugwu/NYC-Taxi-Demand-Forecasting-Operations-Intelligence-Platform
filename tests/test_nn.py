import numpy as np
import torch

from taxi.features import build_features
from taxi.models.nn import COUNT_COLS, SEQ_LEN, NeuralForecaster
from test_features import make_panel


def test_gru_window_ends_at_forecast_origin():
    panel = make_panel()
    frame = build_features(panel, 6, weather="observed")
    m = NeuralForecaster(use_gru=True)
    m.attach_panel(panel)
    m.zones = np.sort(frame.zone_id.unique())
    m.num_cols = COUNT_COLS + ["is_holiday"]
    m.mu, m.sd = 0.0, 1.0
    m.seq_mu, m.seq_sd = torch.zeros(3), torch.ones(3)

    t = m._tensors(frame)
    row = 500
    steps = t["origin"][row] + torch.arange(-SEQ_LEN + 1, 1)
    window = m.panel[t["panel_zone"][row], steps, 0]

    r = frame.iloc[row]
    z = panel[panel.zone_id == r.zone_id].set_index("hour_at")["pickups"]
    hours = [r.hour_at - np.timedelta64(6 + k, "h") for k in range(SEQ_LEN - 1, -1, -1)]
    expected = np.log1p(z.loc[hours].to_numpy(np.float32))
    assert np.allclose(window.numpy(), expected)


def test_fit_and_predict_run():
    panel = make_panel()
    frame = build_features(panel, 1, weather="observed")
    cut = frame.hour_at.quantile(0.8)
    m = NeuralForecaster(use_gru=True, max_epochs=2, verbose=False)
    m.attach_panel(panel)
    m.fit(frame[frame.hour_at < cut], frame[frame.hour_at >= cut])
    pred = m.predict(frame.tail(50))
    assert pred.shape == (50,) and (pred > 0).all()
