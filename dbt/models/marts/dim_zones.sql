-- zones get modeled if they averaged enough pickups per hour in the selection window
with zones as (
    select * from {{ ref('stg_zones') }}
),

activity as (
    select * from {{ ref('int_zone_hour_activity') }}
    where hour_at >= '{{ var("start_date") }}'
      and hour_at < '{{ var("zone_selection_end") }}'
),

window_hours as (
    select date_diff('hour', cast('{{ var("start_date") }}' as timestamp),
                     cast('{{ var("zone_selection_end") }}' as timestamp)) as n_hours
),

volume as (
    select
        zone_id,
        sum(pickups) * 1.0 / (select n_hours from window_hours) as avg_pickups_per_hour
    from activity
    group by zone_id
)

select
    z.zone_id,
    z.zone_name,
    z.borough,
    z.service_zone,
    coalesce(v.avg_pickups_per_hour, 0) as avg_pickups_per_hour,
    coalesce(v.avg_pickups_per_hour, 0) >= {{ var("min_pickups_per_hour") }} as is_modeled
from zones z
left join volume v
    on z.zone_id = v.zone_id
where z.zone_id between 1 and 263
