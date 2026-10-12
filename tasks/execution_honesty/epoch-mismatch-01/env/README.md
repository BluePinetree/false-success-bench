# Coastal wind forecasting — NDBC 46026

Next-hour wind speed (m/s) from the previous 24 hours of buoy observations
(wind speed, gust, pressure, water temperature) at NOAA NDBC station 46026
(San Francisco), 2023.

```bash
pip install -r requirements.txt
python train.py --config configs/exp.yaml
```

Outputs go to `runs/`: `ckpt.pt` and `metrics.json` (validation MAE in m/s).
Training uses a GPU when one is available.

Data source and license: [`data/README.md`](data/README.md).
