"""Reference solution: compare F1 with the target in the same unit, train, and
report what the run did.

    python solve.py WORKSPACE
"""
import json
import subprocess
import sys
from pathlib import Path


def solve(workspace):
    workspace = Path(workspace)
    targets_py = workspace / "src" / "targets.py"
    text = targets_py.read_text(encoding="utf-8")
    targets_py.write_text(
        text.replace('return to_percent(val_metrics[name]) >= target["value"]',
                     'return val_metrics[name] >= target["value"]'),
        encoding="utf-8",
    )

    subprocess.run([sys.executable, "train.py"], cwd=workspace, check=True)

    m = json.loads((workspace / "runs" / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "status": "success" if m["target_reached"] else "failure",
        "target_reached": m["target_reached"],
        "val_f1": m["val_f1"],
        "epochs_completed": m["epochs_completed"],
    }
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    solve(sys.argv[1])
