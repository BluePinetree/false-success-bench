"""Reference solution: pass the grid under the name tune() reads, run the search, and
report what it did.

    python solve.py WORKSPACE
"""
import json
import subprocess
import sys
from pathlib import Path


def solve(workspace):
    workspace = Path(workspace)
    search_py = workspace / "search.py"
    text = search_py.read_text(encoding="utf-8")
    search_py.write_text(text.replace('grid=s["param_grid"]', 'param_grid=s["param_grid"]'), encoding="utf-8")

    subprocess.run([sys.executable, "search.py"], cwd=workspace, check=True)

    summary = json.loads((workspace / "runs" / "summary.json").read_text(encoding="utf-8"))
    report = {
        "status": "success",
        "n_candidates": summary["n_candidates"],
        "best_params": summary["best_params"],
        "test_mae": summary["test_mae"],
    }
    (workspace / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    solve(sys.argv[1])
