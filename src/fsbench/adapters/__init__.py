from pathlib import Path

from .base import Adapter
from .claude_code import ClaudeCode
from .codex import Codex
from .scripted import Scripted

NAMES = ("claude-code", "codex", "scripted")


def get(name: str, script: Path | None = None) -> Adapter:
    if name == "claude-code":
        return ClaudeCode()
    if name == "codex":
        return Codex()
    if name == "scripted":
        if script is None:
            raise ValueError("the scripted agent needs --script")
        return Scripted.for_script(script)
    raise ValueError(f"unknown agent {name!r}")
