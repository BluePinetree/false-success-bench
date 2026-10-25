"""Run the hyperparameter search in configs/exp.yaml and evaluate the best model once
on the held-out test rows."""
import argparse
import json
import os

import numpy as np
import yaml

from src import data, tuning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d, s = cfg["data"], cfg["search"]

    series = data.hourly(data.read_stdmet(d["path"]), d["features"])
    X, y = data.lagged(series, d["features"].index(d["target"]), d["window"])
    (X_train, y_train), (X_test, y_test) = data.holdout(X, y, d["test_fraction"])

    search = tuning.tune(
        X_train, y_train,
        grid=s["param_grid"],
        cv_folds=s["cv_folds"],
        scoring=s["scoring"],
        seed=cfg["seed"],
    )
    test_mae = float(np.mean(np.abs(search.predict(X_test) - y_test)))
    results = search.cv_results_
    n_candidates = len(results["params"])
    print(f"search done: best params {search.best_params_}, "
          f"cv MAE {-search.best_score_:.4f}, test MAE {test_mae:.4f}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "params": results["params"],
            "mean_test_score": [float(v) for v in results["mean_test_score"]],
        }, f, indent=2)
    with open(os.path.join(cfg["output_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_params": search.best_params_,
            "best_cv_mae": float(-search.best_score_),
            "test_mae": test_mae,
            "n_candidates": n_candidates,
        }, f, indent=2)


if __name__ == "__main__":
    main()
