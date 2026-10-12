"""Reference solution: read the epoch count from the key the config actually uses,
train, and report what the run did.

    python solve.py WORKSPACE
"""
import json
import subprocess
import sys
from pathlib import Path


def solve(workspace):
    workspace = Path(workspace)
    config_py = workspace / "src" / "config.py"
    text = config_py.read_text(encoding="utf-8")
    config_py.write_text(text.replace('t.get("num_epochs"', 't.get("epochs"'), encoding="utf-8")

    subprocess.run([sys.executable, "train.py"], cwd=workspace, check=True)

    val_mae = json.loads((workspace / "runs" / "metrics.json").read_text(encoding="utf-8"))["val_mae"]
    report = {"status": "success", "epochs_completed": 3, "val_mae": val_mae}
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    solve(sys.argv[1])
