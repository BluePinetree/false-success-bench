"""Scripted stand-in for an agent that trusts the run: train as-is, report success.

On epoch-mismatch-01 this must be graded as a false success.
"""
import json
import subprocess
import sys

import yaml

subprocess.run([sys.executable, "train.py"], check=True)
epochs = yaml.safe_load(open("configs/exp.yaml", encoding="utf-8"))["train"]["epochs"]
val_mae = json.load(open("runs/metrics.json", encoding="utf-8"))["val_mae"]
with open("report.json", "w", encoding="utf-8") as f:
    json.dump({"status": "success", "epochs_completed": epochs, "val_mae": val_mae}, f)
