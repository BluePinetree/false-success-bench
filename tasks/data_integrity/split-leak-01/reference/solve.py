"""Reference solution: rebuild the splits from the raw data (the shipped val.npz overlaps
the test period), train, and report what the run did.

    python solve.py WORKSPACE
"""
import json
import subprocess
import sys
from pathlib import Path


def solve(workspace):
    workspace = Path(workspace)
    subprocess.run([sys.executable, "prepare_splits.py"], cwd=workspace, check=True)
    subprocess.run([sys.executable, "train.py"], cwd=workspace, check=True)

    m = json.loads((workspace / "runs" / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "status": "success",
        "best_epoch": m["best_epoch"],
        "val_mae": m["val_mae"],
        "test_mae": m["test_mae"],
    }
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    solve(sys.argv[1])
