"""A test double: runs a fixed script instead of a model.

Used to exercise the whole harness (container, network, timeout, grading) without an
API key. Its results are never part of a baseline.
"""
from dataclasses import dataclass
from pathlib import Path

from .base import Adapter

SCRIPT_MOUNT = "/fsb-script"


@dataclass(frozen=True)
class Scripted(Adapter):
    name: str = "scripted"
    script_name: str = ""

    @classmethod
    def for_script(cls, script: Path):
        script = Path(script).resolve()
        return cls(mounts={str(script.parent): SCRIPT_MOUNT}, script_name=script.name)

    def command(self, goal, model):
        return ["python", f"{SCRIPT_MOUNT}/{self.script_name}"]
