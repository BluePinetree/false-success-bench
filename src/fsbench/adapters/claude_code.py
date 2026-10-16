from dataclasses import dataclass, field

from .base import Adapter


@dataclass(frozen=True)
class ClaudeCode(Adapter):
    name: str = "claude-code"
    secret_env: tuple = ("ANTHROPIC_API_KEY",)
    env: dict = field(default_factory=lambda: {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})

    def command(self, goal, model):
        cmd = [
            "claude", "-p", goal,
            "--output-format", "stream-json", "--verbose",
            # The container is the sandbox (DESIGN.md §7); there is no one to approve prompts.
            "--dangerously-skip-permissions",
        ]
        if model:
            cmd += ["--model", model]
        return cmd

    def version_command(self):
        return ["claude", "--version"]
