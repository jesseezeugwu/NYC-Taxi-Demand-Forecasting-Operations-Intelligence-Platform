-- panel pickups should match a count straight off the raw table with the same rules
-- written out again, so a bad join or filter in between shows up here
with raw_count as (
    select count(distinct {{ dbt_utils.generate_surrogate_key([
            'VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'PULocationID',
            'DOLocationID', 'fare_amount', 'tip_amount', 'trip_distance', 'total_amount'
        ]) }}) as n
    from {{ source('raw', 'yellow_tripdata') }}
    where tpep_pickup_datetime >= '{{ var("start_date") }}'
      and tpep_pickup_datetime < '{{ var("end_date") }}'
      and epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime) between 60 and 18000
      and fare_amount + tip_amount + extra > 0
      and PULocationID in (select zone_id from {{ ref('dim_zones') }} where is_modeled)
),

panel as (
    select sum(pickups) as n from {{ ref('mart_zone_hour_demand') }}
)

select r.n as raw_trips, p.n as panel_pickups
from raw_count r, panel p
where r.n != p.n
