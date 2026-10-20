# High-wind alert — NDBC 46013

Binary classifier: will wind speed reach 10 m/s within the next 3 hours, given the
previous 24 hours of buoy observations (wind speed, gust, pressure, air temperature)
at NOAA NDBC station 46013 (Bodega Bay), 2023.

```bash
pip install -r requirements.txt
python train.py --config configs/exp.yaml
```

Training stops early once the validation target in `configs/exp.yaml` is met, or after
`max_epochs`. Outputs go to `runs/`: `ckpt.pt` and `metrics.json`.
Training uses a GPU when one is available.

Data source and license: [`data/README.md`](data/README.md).
