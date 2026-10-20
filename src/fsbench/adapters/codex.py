from dataclasses import dataclass, field

from .base import Adapter


@dataclass(frozen=True)
class Codex(Adapter):
    name: str = "codex"
    secret_env: tuple = ("OPENAI_API_KEY", "CODEX_API_KEY")
    # `codex exec` authenticates from CODEX_API_KEY, not OPENAI_API_KEY.
    env_aliases: dict = field(default_factory=lambda: {"CODEX_API_KEY": "OPENAI_API_KEY"})

    def command(self, goal, model):
        cmd = [
            "codex", "exec", "--json",
            # The container is the sandbox (DESIGN.md §7); task workspaces are not git repos.
            "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
        ]
        if model:
            cmd += ["--model", model]
        return cmd + [goal]

    def version_command(self):
        return ["codex", "--version"]
