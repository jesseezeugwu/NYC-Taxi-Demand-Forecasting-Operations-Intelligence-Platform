-- used to be incremental, but TLC re-publishes whole months and a corrected trip
-- hashes to a new trip_id, so old rows stuck around. a full rebuild takes ~30s anyway.
with source as (
    select * from {{ source('raw', 'yellow_tripdata') }}
    -- the monthly files have a few stray timestamps from other years
    where tpep_pickup_datetime >= '{{ var("start_date") }}'
      and tpep_pickup_datetime < '{{ var("end_date") }}'
),

renamed as (
    select
        {{ dbt_utils.generate_surrogate_key([
            'VendorID',
            'tpep_pickup_datetime',
            'tpep_dropoff_datetime',
            'PULocationID',
            'DOLocationID',
            'fare_amount',
            'tip_amount',
            'trip_distance',
            'total_amount'
        ]) }} as trip_id,
        VendorID              as vendor_id,
        tpep_pickup_datetime  as pickup_at,
        tpep_dropoff_datetime as dropoff_at,
        PULocationID          as pickup_location_id,
        DOLocationID          as dropoff_location_id,
        passenger_count,
        trip_distance,
        payment_type,
        fare_amount,
        tip_amount,
        extra,
        total_amount
    from source
),

deduped as (
    select *, row_number() over (partition by trip_id order by pickup_at) as rn
    from renamed
)

select * exclude (rn)
from deduped
where rn = 1
