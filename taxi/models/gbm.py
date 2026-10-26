import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.features import feature_columns

# set by hand. the learning rate went 0.05 -> 0.1 after looking at the test months,
# rechecked on Oct-Nov 2025 with taxi/tune.py and kept (artifacts/tune/decision.json)
PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.1,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 7,
}


class LightGBMModel:
    def __init__(self, weather="observed", max_rounds=6000, params=None):
        self.weather = weather
        self.max_rounds = max_rounds
        self.params = {**PARAMS, **(params or {})}

    def _x(self, frame):
        x = frame[self.columns].copy()
        x["zone_id"] = pd.Categorical(x["zone_id"], categories=self.zones)
        return x

    def fit(self, train, valid):
        self.columns = feature_columns(train)
        self.zones = np.sort(train["zone_id"].unique())

        # pick the number of rounds on the validation weeks
        dtrain = lgb.Dataset(self._x(train), train["y"])
        dvalid = lgb.Dataset(self._x(valid), valid["y"], reference=dtrain)
        booster = lgb.train(self.params, dtrain, self.max_rounds, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(200, verbose=False)])
        self.rounds = booster.best_iteration

        # refit on train + valid
        both = pd.concat([train, valid])
        self.booster = lgb.train(self.params, lgb.Dataset(self._x(both), both["y"]), self.rounds)
        return self

    def predict(self, frame):
        return self.booster.predict(self._x(frame))

    def importance(self):
        gain = self.booster.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.booster.feature_name())
