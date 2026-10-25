"""Reference solution: drop the debugging cap so training uses all the planned data,
make check_run.py read the field the run log actually writes, train, check, and report.

    python solve.py WORKSPACE
"""
import json
import subprocess
import sys
from pathlib import Path


def _replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, (path, old)
    path.write_text(text.replace(old, new), encoding="utf-8")


def solve(workspace):
    workspace = Path(workspace)
    _replace(workspace / "train.py",
             'max_windows=d.get("max_windows", DEBUG_MAX_WINDOWS))',
             'max_windows=d.get("max_windows"))')
    _replace(workspace / "check_run.py",
             'executed_windows = executed.get("train_windows", planned["train_windows"])',
             'executed_windows = executed["n_train"]')

    subprocess.run([sys.executable, "train.py"], cwd=workspace, check=True)
    subprocess.run([sys.executable, "check_run.py"], cwd=workspace, check=True)

    log = json.loads((workspace / "runs" / "run_log.json").read_text(encoding="utf-8"))
    m = json.loads((workspace / "runs" / "metrics.json").read_text(encoding="utf-8"))
    report = {"status": "success", "n_train_windows": log["executed"]["n_train"], "val_mae": m["val_mae"]}
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    solve(sys.argv[1])
