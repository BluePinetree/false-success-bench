"""Every task directory under tasks/ must pass schema validation and have a grader."""
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "src" / "fsbench" / "schemas" / "task.schema.json").read_text(encoding="utf-8"))
TASK_FILES = sorted(p for p in (ROOT / "tasks").glob("*/*/task.json") if p.parent.parent.name != "_template")


def test_at_least_one_task():
    assert TASK_FILES


@pytest.mark.parametrize("task_file", TASK_FILES, ids=lambda p: p.parent.name)
def test_task_is_valid(task_file):
    task = json.loads(task_file.read_text(encoding="utf-8"))
    jsonschema.validate(task, SCHEMA)

    task_dir = task_file.parent
    assert task["id"] == f"{task_dir.parent.name}/{task_dir.name}"
    assert task["category"] == task_dir.parent.name
    assert (task_dir / task["grader"]["entrypoint"]).is_file()
    assert (task_dir / task["reference"]).is_file()
    assert (task_dir / "env").is_dir()

    pinned = task["grader"].get("config_sha256")
    if pinned:
        config = task_dir / "env" / "configs" / "exp.yaml"
        assert hashlib.sha256(config.read_bytes()).hexdigest() == pinned
