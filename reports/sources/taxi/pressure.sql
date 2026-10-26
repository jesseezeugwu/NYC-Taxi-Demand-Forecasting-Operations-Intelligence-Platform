select
    zone_name,
    borough,
    cast(day_of_week as integer) as day_of_week,
    cast(hour_of_day as integer) as hour_of_day,
    round(forecast_pickups, 1) as forecast_pickups,
    round(dropoffs_prev_hour, 1) as dropoffs_prev_hour,
    round(pickups_per_arriving_cab, 2) as pickups_per_arriving_cab,
    round(earnings_per_trip, 2) as earnings_per_trip,
    round(dollars_per_arriving_cab, 2) as dollars_per_arriving_cab
from main.demand_pressure
where forecast_pickups >= 20
