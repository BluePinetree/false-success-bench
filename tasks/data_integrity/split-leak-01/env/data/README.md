# Data

`46042h2023.txt.gz` — NOAA National Data Buoy Center (NDBC), standard meteorological
data, station 46042 (Monterey), calendar year 2023, unmodified.

- Source: https://www.ndbc.noaa.gov/data/historical/stdmet/46042h2023.txt.gz
- SHA-256: `807a66353389c5acf74bdeeff2621d07ae46a190cdaa08f62f5306b8f5be63ab`
- Format: https://www.ndbc.noaa.gov/faq/measdes.shtml — missing values are written as
  all-nines of the field width (99.0, 999, 9999.0).
- Terms (NOAA ERDDAP distribution of the same NDBC data): "The data may be used and
  redistributed for free but is not intended for legal use, since it may contain
  inaccuracies."

`splits/{train,val,test}.npz` — hourly series derived from the file above: `t` (hours
since 1970-01-01 UTC) and `x` (features in `configs/exp.yaml` order).
