-- every modeled zone x every hour, zero-filled, with weather and calendar
with zones as (
    select * from {{ ref('dim_zones') }}
    where is_modeled
),

hours as (
    select unnest(range(
        cast('{{ var("start_date") }}' as timestamp),
        cast('{{ var("end_date") }}' as timestamp),
        interval 1 hour
    )) as hour_at
),

-- TLC timestamps are local wall-clock time. the spring-forward 2am never happens
-- and the fall-back 1am holds two hours of trips. keep the rows so lags stay
-- aligned, but flag them so they aren't trained on or scored.
hours_flagged as (
    select
        hour_at,
        ((hour_at at time zone 'America/New_York') at time zone 'America/New_York') != hour_at
        or date_diff('hour', (hour_at - interval 1 hour) at time zone 'America/New_York',
                     hour_at at time zone 'America/New_York') = 2 as is_dst_shift
    from hours
),

activity as (
    select * from {{ ref('int_zone_hour_activity') }}
),

weather as (
    select * from {{ ref('stg_weather_hourly') }}
),

holidays as (
    select holiday_date from {{ ref('nyc_holidays') }}
)

select
    z.zone_id,
    h.hour_at,
    z.borough,
    z.service_zone,
    coalesce(a.pickups, 0) as pickups,
    coalesce(a.dropoffs, 0) as dropoffs,
    coalesce(a.earnings, 0) as earnings,
    coalesce(a.trip_minutes, 0) as trip_minutes,
    w.temp_c,
    w.precip_mm,
    w.wind_ms,
    w.is_snow,
    extract(hour from h.hour_at) as hour_of_day,
    extract(dow from h.hour_at) as day_of_week,  -- 0 = Sunday
    extract(month from h.hour_at) as month,
    hol.holiday_date is not null as is_holiday,
    h.is_dst_shift
from zones z
cross join hours_flagged h
left join activity a
    on z.zone_id = a.zone_id
    and h.hour_at = a.hour_at
left join weather w
    on h.hour_at = w.hour_at
left join holidays hol
    on cast(h.hour_at as date) = hol.holiday_date
