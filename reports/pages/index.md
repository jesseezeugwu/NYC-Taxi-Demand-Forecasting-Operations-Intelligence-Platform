---
title: NYC Taxi Demand Forecasting
---

Hourly yellow cab pickup forecasts for 103 NYC taxi zones (99% of all pickups), using
TLC trips from Jan 2024 to May 2026 and Central Park weather. Every score here comes
from a rolling backtest, so each month is predicted by models that only saw earlier data.

## Forecast error

WAPE is total absolute error divided by total pickups. 15% means the forecast was
off by 15 pickups for every 100 that happened.

```sql scores
select * from taxi.scores order by horizon, wape_pct
```

<DataTable data={scores} rows=30 groupBy=horizon/>

```sql folds
select * from taxi.scores_by_fold
where horizon = 1 and model in ('seasonal_naive', 'how_mean', 'lightgbm', 'gru')
order by fold
```

<LineChart data={folds} x=fold y=wape_pct series=model title="WAPE by test month, 1 hour ahead"/>

The simple baselines struggle in December and January because of the holidays. The
trained models handle it better, but January is still their worst month.

### Model gaps

Difference in WAPE between two models in percentage points, with a 95% interval from
resampling days. If the interval doesn't include zero, the gap isn't just noise.

```sql comparisons
select * from taxi.comparisons order by horizon, hours, model_a
```

<DataTable data={comparisons} rows=30 groupBy=horizon/>

## Forecast vs actual

```sql showcase_zones
select distinct zone_name from taxi.showcase order by 1
```

<Dropdown data={showcase_zones} name=zone value=zone_name defaultValue="Midtown Center"/>

```sql showcase_zone
select hour_at, y as actual, lightgbm, gru, seasonal_naive
from taxi.showcase
where zone_name = '${inputs.zone.value}'
order by hour_at
```

<LineChart data={showcase_zone} x=hour_at y={['actual', 'lightgbm', 'gru', 'seasonal_naive']} title="Hourly pickups, first two weeks of the last test month"/>

## Where it's worse

```sql segments
select segment_type, segment, model, wape_pct, bias_pct, n
from taxi.scores_by_segment
where horizon = 1 and model in ('seasonal_naive', 'lightgbm', 'gru')
order by segment_type, segment, model
```

<DataTable data={segments} rows=40 groupBy=segment_type/>

```sql importance
select feature, round(h1, 3) as gain_pct from taxi.importance order by h1 desc limit 15
```

<BarChart data={importance} x=feature y=gain_pct sort=false title="LightGBM share of split gain, 1 hour ahead"/>

## Zone embeddings

The MLP learns a 16-number vector for each zone. This is the first two principal
components, colored by TLC service zone. It's from one training run.

```sql embeddings
select * from taxi.embeddings
```

<ScatterPlot data={embeddings} x=pc1 y=pc2 series=service_zone tooltipTitle=zone_name title="Zone embeddings (PCA)"/>

## Rain and pickups

Percent change in pickups from a Poisson regression on rain, snow and temperature,
with hour-of-week and month effects. The daily rows use the same data summed by day.
These are completed pickups, so rain could also mean fewer free cabs.

```sql rain
select "group", granularity, term, level, change, ci_low, ci_high
from taxi.weather_effects
order by "group", granularity desc, term, level
```

<DataTable data={rain} rows=30 groupBy=group/>

```sql rain_profile
select * from taxi.rain_profile order by hour_of_day
```

<LineChart data={rain_profile} x=hour_of_day y=pickups series=wet title="Manhattan core, average pickups per hour"/>

## Where to go next

Forecast pickups divided by the cabs that dropped off in the zone the hour before,
times the typical fare there. There are no driver IDs in the data, so this is only
a rough guess at competition.

```sql pressure
select * from taxi.pressure order by dollars_per_arriving_cab desc limit 20
```

<DataTable data={pressure} rows=20/>
