"""MLP and GRU forecasters. Both output a Poisson log-rate with log(4-week hour-of-week mean) as an offset."""
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

COUNT_COLS = ["lag_0", "lag_1", "lag_2", "lag_day", "lag_week", "lag_2week",
              "mean_3h", "mean_24h", "mean_week", "drop_0", "drop_mean_3h", "how_mean_4w"]
WEATHER_COLS = ["temp_c", "precip_mm", "wind_ms", "is_snow", "precip_3h"]
SEQ_LEN = 24


class Net(nn.Module):
    def __init__(self, n_zones, n_num, use_gru=False, hidden=256, gru_hidden=64):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, 16)
        self.how_emb = nn.Embedding(168, 8)
        self.month_emb = nn.Embedding(12, 3)
        self.gru = nn.GRU(3, gru_hidden, batch_first=True) if use_gru else None
        in_dim = 16 + 8 + 3 + n_num + (gru_hidden if use_gru else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, zone, how, month, num, offset, seq=None):
        parts = [self.zone_emb(zone), self.how_emb(how), self.month_emb(month), num]
        if self.gru is not None:
            _, h = self.gru(seq)
            parts.append(h[-1])
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1) + offset


class NeuralForecaster:
    def __init__(self, use_gru=False, weather="observed", max_epochs=None, patience=3, seed=7, verbose=True):
        self.use_gru = use_gru
        self.weather = weather
        self.max_epochs = max_epochs or (20 if use_gru else 30)
        self.batch_size = 4096 if use_gru else 8192
        self.lr = 3e-3
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.panel = None

    def attach_panel(self, panel):
        # the GRU reads raw hourly history straight from the panel
        n_zones = panel["zone_id"].nunique()
        n_hours = panel["hour_at"].nunique()
        cube = np.stack([
            np.log1p(panel["pickups"].to_numpy(np.float32)),
            np.log1p(panel["dropoffs"].to_numpy(np.float32)),
            np.log1p(panel["precip_mm"].to_numpy(np.float32)),
        ], axis=1).reshape(n_zones, n_hours, 3)
        self.panel = torch.as_tensor(cube)
        self.panel_zones = np.sort(panel["zone_id"].unique())

    def _tensors(self, frame):
        num = frame[self.num_cols].to_numpy(dtype=np.float32).copy()
        k = len(COUNT_COLS)
        num[:, :k] = np.log1p(num[:, :k])
        num = (num - self.mu) / self.sd
        t = {
            "zone": torch.as_tensor(np.searchsorted(self.zones, frame["zone_id"].to_numpy())),
            "how": torch.as_tensor(frame["hour_of_week"].to_numpy().astype(np.int64)),
            "month": torch.as_tensor(frame["month"].to_numpy().astype(np.int64) - 1),
            "num": torch.as_tensor(num),
            "offset": torch.as_tensor(np.log(frame["how_mean_4w"].to_numpy(np.float32) + 0.5)),
        }
        if self.use_gru:
            t["panel_zone"] = torch.as_tensor(np.searchsorted(self.panel_zones, frame["zone_id"].to_numpy()))
            # window ends at the forecast origin T - h
            t["origin"] = torch.as_tensor(frame["hour_idx"].to_numpy() - frame["horizon"].to_numpy())
        return t

    def _forward(self, t, idx):
        seq = None
        if self.use_gru:
            steps = t["origin"][idx].unsqueeze(1) + torch.arange(-SEQ_LEN + 1, 1).unsqueeze(0)
            seq = (self.panel[t["panel_zone"][idx].unsqueeze(1), steps] - self.seq_mu) / self.seq_sd
        return self.net(t["zone"][idx], t["how"][idx], t["month"][idx], t["num"][idx], t["offset"][idx], seq)

    def _train_epochs(self, t, y, epochs, valid=None):
        torch.manual_seed(self.seed)
        self.net = Net(len(self.zones), len(self.num_cols), use_gru=self.use_gru)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.PoissonNLLLoss(log_input=True)
        best, best_state, best_epoch, bad = np.inf, None, 0, 0

        for epoch in range(1, epochs + 1):
            start = time.time()
            self.net.train()
            perm = torch.randperm(len(y))
            for i in range(0, len(y), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._forward(t, idx), y[idx])
                loss.backward()
                opt.step()
            if valid is None:
                continue

            val = self._eval_loss(*valid, loss_fn)
            if self.verbose:
                print(f"    epoch {epoch}: valid {val:.4f} ({time.time() - start:.0f}s)")
            if val < best - 1e-4:
                best, best_state, best_epoch, bad = val, copy.deepcopy(self.net.state_dict()), epoch, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if valid is not None:
            if best_state is None:
                raise RuntimeError("validation loss never improved (nan?)")
            self.net.load_state_dict(best_state)
        return best_epoch

    def fit(self, train, valid):
        np.random.seed(self.seed)
        self.zones = np.sort(train["zone_id"].unique())
        self.num_cols = COUNT_COLS + ["is_holiday"] + [c for c in WEATHER_COLS if c in train.columns]

        both = pd.concat([train, valid])
        raw = both[self.num_cols].to_numpy(dtype=np.float32).copy()
        raw[:, :len(COUNT_COLS)] = np.log1p(raw[:, :len(COUNT_COLS)])
        self.mu, self.sd = raw.mean(axis=0), raw.std(axis=0) + 1e-6
        if self.use_gru:
            # scale the history with hours before the test month only
            seen = self.panel[:, : int(both["hour_idx"].max()) + 1]
            self.seq_mu, self.seq_sd = seen.mean(dim=(0, 1)), seen.std(dim=(0, 1)) + 1e-6

        # pick the epoch count on the validation weeks, then retrain on everything
        ytr = torch.as_tensor(train["y"].to_numpy(np.float32))
        yva = torch.as_tensor(valid["y"].to_numpy(np.float32))
        self.best_epoch = self._train_epochs(self._tensors(train), ytr, self.max_epochs,
                                             valid=(self._tensors(valid), yva))
        if self.verbose:
            print(f"    best epoch {self.best_epoch}, refitting on train + valid")
        yall = torch.as_tensor(both["y"].to_numpy(np.float32))
        self._train_epochs(self._tensors(both), yall, self.best_epoch)
        return self

    @torch.no_grad()
    def _eval_loss(self, t, y, loss_fn):
        self.net.eval()
        total = 0.0
        for i in range(0, len(y), 32768):
            idx = torch.arange(i, min(i + 32768, len(y)))
            total += loss_fn(self._forward(t, idx), y[idx]).item() * len(idx)
        return total / len(y)

    @torch.no_grad()
    def predict(self, frame):
        self.net.eval()
        t = self._tensors(frame)
        out = [torch.exp(self._forward(t, torch.arange(i, min(i + 32768, len(frame)))))
               for i in range(0, len(frame), 32768)]
        return torch.cat(out).numpy()

    def zone_embeddings(self):
        w = self.net.zone_emb.weight.detach().numpy()
        return pd.DataFrame(w, index=pd.Index(self.zones, name="zone_id"))
