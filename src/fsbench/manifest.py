"""Run manifest: commit SHA, image digest, agent/model id, task version, seed, timestamps."""
import subprocess
from datetime import datetime, timezone

from .task import REPO_ROOT


def _git(*args):
    out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False)
    return out.stdout.strip() or None


def repo_state() -> dict:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def build(*, run_id, task, adapter, model, agent_version, images, gpu, repeat, outcome) -> dict:
    return {
        "run_id": run_id,
        "task_id": task.id,
        "task_version": task.version,
        "agent": adapter.name,
        "agent_version": agent_version,
        "model": model,
        "repeat": repeat,
        "network": "proxy",  # the only mode: model APIs only (DESIGN.md §7)
        "gpu": gpu,
        "images": images,
        "benchmark": repo_state(),
        # Names only; values are never recorded. The host name is deliberately omitted.
        "secret_env_forwarded": outcome["secret_env_forwarded"],
        "started_at": iso(outcome["started_at"]),
        "finished_at": iso(outcome["finished_at"]),
        "wall_time_s": outcome["wall_time_s"],
        "timed_out": outcome["timed_out"],
        "container_exit_code": outcome["container_exit_code"],
        "infra_error": outcome["infra_error"],
    }
