"""Agent adapter interface. Harness and grader must not depend on which agent runs."""
from dataclasses import dataclass, field


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

    def command(self, goal: str, model: str | None) -> list:
        raise NotImplementedError

    def version_command(self) -> list | None:
        return None
