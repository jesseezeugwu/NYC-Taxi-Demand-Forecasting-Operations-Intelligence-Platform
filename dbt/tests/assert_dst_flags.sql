-- the only flagged hours should be 2am on a March Sunday or 1am on a November Sunday
select distinct hour_at
from {{ ref('mart_zone_hour_demand') }}
where is_dst_shift
  and not (
      dayofweek(hour_at) = 0
      and ((month(hour_at) = 3 and hour(hour_at) = 2)
        or (month(hour_at) = 11 and hour(hour_at) = 1))
  )
