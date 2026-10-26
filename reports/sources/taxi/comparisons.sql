-- percentage points; not named *_pct so evidence doesn't round them to whole percents
select
    cast(horizon as integer) as horizon,
    model_a,
    model_b,
    hours,
    round(wape_diff * 100, 2) as diff_points,
    round(ci_low * 100, 2) as ci_low,
    round(ci_high * 100, 2) as ci_high
from main.model_comparisons
