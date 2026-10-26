with trips as (
    select * from {{ ref('stg_yellow_trips') }}
),

valid as (
    select
        trip_id,
        pickup_at,
        dropoff_at,
        pickup_location_id,
        dropoff_location_id,
        -- date_diff('minute') counts minute boundaries, so use seconds
        date_diff('second', pickup_at, dropoff_at) / 60.0 as trip_duration_min,
        fare_amount,
        tip_amount,
        extra,
        fare_amount + tip_amount + extra as driver_earnings
    from trips
)

select *
from valid
where trip_duration_min >= 1
  and trip_duration_min <= 300
  and driver_earnings > 0
  -- 264 and 265 are unknown zones
  and pickup_location_id between 1 and 263
