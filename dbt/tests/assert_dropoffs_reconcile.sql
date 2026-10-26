-- dropoffs landing inside the window in a modeled zone should all make it into the panel
with trips as (
    select count(*) as n
    from {{ ref('int_trips_valid') }}
    where dropoff_location_id in (select zone_id from {{ ref('dim_zones') }} where is_modeled)
      and dropoff_at < '{{ var("end_date") }}'
),

panel as (
    select sum(dropoffs) as n from {{ ref('mart_zone_hour_demand') }}
)

select t.n as trips, p.n as panel_dropoffs
from trips t, panel p
where t.n != p.n
