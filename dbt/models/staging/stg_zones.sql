with source as (
    select * from {{ source('raw', 'taxi_zones') }}
),

renamed as (
    select
        LocationID    as zone_id,
        Zone          as zone_name,
        Borough       as borough,
        service_zone
    from source
)

select * from renamed