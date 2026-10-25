# Coastal wind forecasting — NDBC 46013

Next-hour wind speed (m/s) from the previous 24 hours of buoy observations (wind
speed, gust, pressure, air temperature) at NOAA NDBC station 46013 (Bodega Bay), 2023.
The hourly series is split chronologically 70/15/15.

```bash
pip install -r requirements.txt
python train.py --config configs/exp.yaml
python check_run.py
```

`train.py` writes `runs/ckpt.pt`, `runs/metrics.json` (validation MAE in m/s) and
`runs/run_log.json` (what the run planned and what it executed). `check_run.py`
validates a finished run before its numbers are reported; exit code 0 means valid.
Training uses a GPU when one is available.

Data source and license: [`data/README.md`](data/README.md).
