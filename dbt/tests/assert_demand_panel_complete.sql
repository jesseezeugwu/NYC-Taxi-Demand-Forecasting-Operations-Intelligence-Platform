-- panel should be exactly modeled zones x hours in the window, no gaps or join fan-out
with expected as (
    select
        (select count(*) from {{ ref('dim_zones') }} where is_modeled)
        * date_diff('hour', cast('{{ var("start_date") }}' as timestamp),
                            cast('{{ var("end_date") }}' as timestamp)) as n
),

actual as (
    select count(*) as n from {{ ref('mart_zone_hour_demand') }}
)

select e.n as expected_rows, a.n as actual_rows
from expected e, actual a
where e.n != a.n
