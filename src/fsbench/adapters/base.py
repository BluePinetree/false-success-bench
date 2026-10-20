"""Agent adapter interface. Harness and grader must not depend on which agent runs."""
import re
from dataclasses import dataclass, field

# Signs that the agent never got to work: the model API rejected or failed the session.
# Such runs are kept but reported apart from results (they measure the harness, not the agent).
INFRA_ERROR = re.compile(
    r"401 Unauthorized|invalid x-api-key|authentication_error|Missing bearer"
    r"|insufficient_quota|overloaded_error|api_error_status\":\s*[45]\d\d"
)


@dataclass(frozen=True)
class Adapter:
    name: str = ""
    # Host environment variables forwarded into the container by name only, so their
    # values never appear in a command line, a manifest, or a log.
    secret_env: tuple = ()
    # Non-secret environment set inside the container.
    env: dict = field(default_factory=dict)
    # Extra read-only bind mounts {host_path: container_path}; used by the scripted adapter.
    mounts: dict = field(default_factory=dict)
    # {name the CLI reads: host variable to take it from}, when the target is unset.
    env_aliases: dict = field(default_factory=dict)

    def command(self, goal: str, model: str | None) -> list:
        raise NotImplementedError

    def version_command(self) -> list | None:
        return None

    def infra_error(self, output: str) -> str | None:
        match = INFRA_ERROR.search(output)
        return match.group(0) if match else None
