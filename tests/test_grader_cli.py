"""The harness reads a grader's stdout as JSON, so stdout must carry the verdict alone.

baseline-v0 found phantom-search-01's grader printing scikit-learn progress first; the
task tests call grade() directly and could not see it. These run each grader the way
the harness does: as a script, on a workspace the agent left empty.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fsbench import grading
from fsbench import task as tasks

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = sorted(
    str(p.parent.relative_to(ROOT / "tasks")).replace("\\", "/")
    for p in (ROOT / "tasks").glob("*/*/task.json") if p.parent.parent.name != "_template"
)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_grader_stdout_is_only_json(task_id, tmp_path):
    t = tasks.load(task_id)
    if task_id.endswith("phantom-search-01") and importlib.util.find_spec("sklearn") is None:
        pytest.skip("needs scikit-learn")
    out = subprocess.run([sys.executable, str(t.dir / t.spec["grader"]["entrypoint"]), str(tmp_path)],
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["claim"] == "none" and result["verdict"] == "fail"


def test_harness_records_non_json_grader_output_as_grader_error(monkeypatch, tmp_path):
    fake = subprocess.CompletedProcess([], 0, stdout="Fitting 5 folds...\n{}", stderr="")
    monkeypatch.setattr(grading.subprocess, "run", lambda *a, **k: fake)
    result = grading.grade_in_container(tasks.load(TASK_IDS[0]), tmp_path)
    assert result["verdict"] == "grader_error"
