-- in percent, same naming reason as comparisons.sql
select
    "group",
    granularity,
    term,
    level,
    round(pct_change, 1) as change,
    round(ci_low, 1) as ci_low,
    round(ci_high, 1) as ci_high,
    p_value
from main.weather_effects
