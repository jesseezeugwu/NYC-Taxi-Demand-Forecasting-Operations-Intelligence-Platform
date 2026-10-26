select model, cast(horizon as integer) as horizon, fold, round(wape, 4) as wape_pct
from main.model_scores_by_fold
