"""Run a task's hidden grader against the final environment state in a separate container.

The grader container has no network, and both the task directory and the agent's final
workspace are mounted read-only, so grading cannot change the evidence it reads.
"""
import json
import subprocess
from pathlib import Path

from .sandbox import _user_args
from .task import Task


def grade_in_container(task: Task, workspace: Path) -> dict:
    entry = task.spec["grader"]["entrypoint"]
    out = subprocess.run([
        "docker", "run", "--rm", "--network", "none", *_user_args(),
        "-v", f"{task.dir.resolve()}:/task:ro",
        "-v", f"{Path(workspace).resolve()}:/workspace:ro",
        task.image, "python", f"/task/{entry}", "/workspace",
    ], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return {"verdict": "grader_error", "stderr": out.stderr[-4000:]}
    return json.loads(out.stdout)
