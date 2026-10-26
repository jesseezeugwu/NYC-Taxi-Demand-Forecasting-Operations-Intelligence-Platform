-- evidence formats *_pct columns as percents, so keep them as fractions
select
    model,
    cast(horizon as integer) as horizon,
    round(wape, 4) as wape_pct,
    round(mae, 2) as mae,
    round(rmse, 2) as rmse,
    round(bias, 4) as bias_pct
from main.model_scores
