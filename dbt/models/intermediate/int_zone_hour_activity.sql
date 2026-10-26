-- pickups and dropoffs per zone-hour (dropoffs ~ empty cabs showing up)
with trips as (
    select * from {{ ref('int_trips_valid') }}
),

pickups as (
    select
        pickup_location_id as zone_id,
        date_trunc('hour', pickup_at) as hour_at,
        count(*) as pickups,
        sum(driver_earnings) as earnings,
        sum(trip_duration_min) as trip_minutes
    from trips
    group by 1, 2
),

dropoffs as (
    select
        dropoff_location_id as zone_id,
        date_trunc('hour', dropoff_at) as hour_at,
        count(*) as dropoffs
    from trips
    where dropoff_location_id between 1 and 263
    group by 1, 2
)

select
    coalesce(p.zone_id, d.zone_id) as zone_id,
    coalesce(p.hour_at, d.hour_at) as hour_at,
    coalesce(p.pickups, 0) as pickups,
    coalesce(p.earnings, 0) as earnings,
    coalesce(p.trip_minutes, 0) as trip_minutes,
    coalesce(d.dropoffs, 0) as dropoffs
from pickups p
full outer join dropoffs d
    on p.zone_id = d.zone_id
    and p.hour_at = d.hour_at
