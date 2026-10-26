select
    model,
    cast(horizon as integer) as horizon,
    segment_type,
    segment,
    cast(n as integer) as n,
    round(wape, 4) as wape_pct,
    round(bias, 4) as bias_pct
from main.model_scores_by_segment
