select
    cast(hour_at as timestamp) as hour_at,
    cast(temp_c as double) as temp_c,
    cast(precip_mm as double) as precip_mm,
    cast(wind_ms as double) as wind_ms,
    cast(is_snow as integer) as is_snow
from {{ source('raw', 'weather_hourly') }}
