select cast(hour_of_day as integer) as hour_of_day, wet, round(pickups, 0) as pickups
from main.rain_hour_profile
