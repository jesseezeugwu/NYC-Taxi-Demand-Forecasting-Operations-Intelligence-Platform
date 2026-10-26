# baselines need no training, they just read a lag column


class Persistence:
    """Last known hour carries forward."""
    column = "lag_0"
    weather = "none"

    def fit(self, train, valid):
        return self

    def predict(self, frame):
        return frame[self.column].to_numpy(dtype=float)


class SeasonalNaive(Persistence):
    """Same hour last week."""
    column = "lag_week"


class HourOfWeekMean(Persistence):
    """Same hour of week averaged over the last 4 weeks."""
    column = "how_mean_4w"
