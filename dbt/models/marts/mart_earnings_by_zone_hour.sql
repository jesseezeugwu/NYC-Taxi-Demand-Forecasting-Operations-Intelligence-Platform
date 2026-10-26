-- typical fare and trip length by zone and hour of week
with trips as (
    select * from {{ ref('int_trips_valid') }}
),

zones as (
    select * from {{ ref('dim_zones') }}
),

aggregated as (
    select
        pickup_location_id as zone_id,
        extract(dow from pickup_at) as day_of_week,
        extract(hour from pickup_at) as hour_of_day,
        count(*) as trip_count,
        sum(driver_earnings) as total_earnings,
        sum(trip_duration_min) / 60.0 as total_hours,
        avg(driver_earnings) as earnings_per_trip,
        avg(trip_duration_min) as avg_trip_min,
        sum(driver_earnings) / nullif(sum(trip_duration_min) / 60.0, 0) as earnings_per_trip_hour
    from trips
    group by 1, 2, 3
)

select a.*, z.zone_name, z.borough
from aggregated a
left join zones z on a.zone_id = z.zone_id
