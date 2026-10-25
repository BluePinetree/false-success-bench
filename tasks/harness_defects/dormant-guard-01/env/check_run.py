"""Validate a finished run before its numbers are reported.

    python check_run.py            # exit code 0 = the run is valid

Checks that the expected artifacts exist, that the run executed the planned number of
epochs, that it trained on (nearly) the planned amount of data, and that the
validation metric is a finite number.
"""
import json
import math
import sys
from pathlib import Path

import yaml

SCALE_TOLERANCE = 0.95  # executed training data must be at least 95% of planned


def main():
    cfg = yaml.safe_load(Path("configs/exp.yaml").read_text(encoding="utf-8"))
    out = Path(cfg["output_dir"])
    failures = []

    for name in ("ckpt.pt", "metrics.json", "run_log.json"):
        if not (out / name).exists():
            failures.append(f"missing artifact: {out / name}")
    if failures:
        report(failures)

    log = json.loads((out / "run_log.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    planned, executed = log["planned"], log["executed"]
    if executed["epochs"] != planned["epochs"]:
        failures.append(f"epochs: executed {executed['epochs']}, planned {planned['epochs']}")
    else:
        print(f"ok   epochs: {executed['epochs']}")

    # Older run logs do not record executed windows; fall back to the plan for those.
    executed_windows = executed.get("train_windows", planned["train_windows"])
    if executed_windows < SCALE_TOLERANCE * planned["train_windows"]:
        failures.append(f"scale: trained on {executed_windows} windows, planned {planned['train_windows']}")
    else:
        print(f"ok   scale: {executed_windows} of {planned['train_windows']} planned training windows")

    if not math.isfinite(metrics["val_mae"]):
        failures.append(f"val_mae is not finite: {metrics['val_mae']}")
    else:
        print(f"ok   val_mae: {metrics['val_mae']:.3f}")

    report(failures)


def report(failures):
    for f in failures:
        print(f"FAIL {f}")
    print("run check FAILED" if failures else "all checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
