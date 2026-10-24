# Coastal wind forecasting — NDBC 46042

Next-hour wind speed (m/s) from the previous 24 hours of buoy observations (wind
speed, gust, pressure, air temperature) at NOAA NDBC station 46042 (Monterey), 2023.

The hourly series is split chronologically 70/15/15 into train, validation and test.
The prepared splits are in `data/splits/`; `prepare_splits.py` builds them from the
raw file in `data/`.

```bash
pip install -r requirements.txt
python train.py --config configs/exp.yaml
```

`train.py` keeps the epoch with the best validation MAE and evaluates that model once
on the test split. Outputs go to `runs/`: `ckpt.pt` and `metrics.json`.
Training uses a GPU when one is available.

Data source and license: [`data/README.md`](data/README.md).
