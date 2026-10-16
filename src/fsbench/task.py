"""Load a task directory and validate task.json against schemas/task.schema.json."""
import json
from dataclasses import dataclass
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "tasks"
SCHEMA = json.loads((Path(__file__).parent / "schemas" / "task.schema.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Task:
    dir: Path
    spec: dict

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def version(self) -> str:
        return self.spec["version"]

    @property
    def slug(self) -> str:
        return self.id.split("/")[1]

    @property
    def env_dir(self) -> Path:
        return self.dir / "env"

    @property
    def image(self) -> str:
        return f"fsb-task/{self.slug}:{self.version}"

    @property
    def agent_image(self) -> str:
        return f"fsb-agent/{self.slug}:{self.version}"


def load(task_id: str) -> Task:
    task_dir = TASKS_DIR / task_id
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    jsonschema.validate(spec, SCHEMA)
    if spec["id"] != task_id:
        raise ValueError(f"{task_dir}/task.json declares id {spec['id']!r}, expected {task_id!r}")
    return Task(task_dir, spec)
