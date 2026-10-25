# Coastal wind forecasting — gradient boosting, NDBC 46026

Next-hour wind speed (m/s) at NOAA NDBC station 46026 (San Francisco), 2023, from the
previous 24 hours of wind speed, gust, pressure and water temperature, flattened into a
lagged feature vector.

```bash
pip install -r requirements.txt
python search.py --config configs/exp.yaml
```

`search.py` runs the grid search defined under `search:` in `configs/exp.yaml` with
time-ordered CV folds on the training rows, refits the best configuration, and
evaluates it once on the held-out test rows (the last 15%). Outputs go to `runs/`:
`cv_results.json` (every candidate and its CV score) and `summary.json`.

Data source and license: [`data/README.md`](data/README.md).
