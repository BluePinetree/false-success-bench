"""Run the task environment in Docker: timeout, resource limits, stdout/stderr capture,
process-tree cleanup. The grader directory is never mounted into the agent container.

Process-tree cleanup is done at the container level: on timeout the whole container is
killed, so no child process (DataLoader workers, a stray training run) can outlive it
or keep holding a GPU.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from .task import REPO_ROOT, Task

DOCKER_DIR = REPO_ROOT / "docker"
PROXY_IMAGE = "fsb-proxy:0.1.0"
PROXY_PORT = 8888
CONTAINER_WORKSPACE = "/workspace"


def docker(*args, timeout=None, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=check)


def image_id(tag: str) -> str:
    return docker("image", "inspect", "--format", "{{.Id}}", tag).stdout.strip()


def dns_args():
    """`--dns` flags from FSB_DNS (comma-separated), for hosts whose network blocks the
    public resolvers Docker falls back to. Applied only to containers that need the
    outside world: the proxy, and the agent in `open` mode."""
    servers = [s.strip() for s in os.environ.get("FSB_DNS", "").split(",") if s.strip()]
    return [arg for s in servers for arg in ("--dns", s)]


def build_images(task: Task, claude_code_version: str, codex_version: str, build_network=None) -> dict:
    """Build the task image, the task+agents image on top of it, and the proxy image.

    `build_network="host"` lets build steps use the host's resolver; only package
    downloads during the build are affected, never a run.
    """
    net = ["--network", build_network] if build_network else []
    subprocess.run(["docker", "build", *net, "-t", task.image, str(task.env_dir)], check=True)
    subprocess.run([
        "docker", "build", *net, "-t", task.agent_image,
        "-f", str(DOCKER_DIR / "agent.Dockerfile"),
        "--build-arg", f"TASK_IMAGE={task.image}",
        "--build-arg", f"CLAUDE_CODE_VERSION={claude_code_version}",
        "--build-arg", f"CODEX_VERSION={codex_version}",
        str(DOCKER_DIR),
    ], check=True)
    subprocess.run(["docker", "build", *net, "-t", PROXY_IMAGE, str(DOCKER_DIR / "proxy")], check=True)
    return {tag: image_id(tag) for tag in (task.image, task.agent_image, PROXY_IMAGE)}


def start_proxy(proxy: str, net: str) -> str:
    """Start an egress proxy on the default bridge and attach it to the internal network."""
    docker("run", "-d", "--name", proxy, *dns_args(), PROXY_IMAGE)
    docker("network", "connect", net, proxy)
    return f"http://{proxy}:{PROXY_PORT}"


def _user_args():
    # Run as the invoking user so the workspace stays owned by them, and because
    # Claude Code refuses its no-approval mode as root.
    if hasattr(os, "getuid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp/agent-home"]
    return []


def _redact(path: Path, secrets: list):
    if not secrets or not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for value in secrets:
        text = text.replace(value, "[REDACTED]")
    path.write_text(text, encoding="utf-8")


def run_agent(task: Task, adapter, model, run_dir: Path, gpu: str | None, network: str) -> dict:
    """Run one agent attempt in a fresh container. Returns what happened, not a verdict."""
    run_dir = Path(run_dir)
    workspace = run_dir / "workspace"
    shutil.copytree(task.env_dir, workspace)

    name = f"fsb-{run_dir.name}"
    res = task.spec["resources"]
    secrets = [os.environ[k] for k in adapter.secret_env if os.environ.get(k)]

    created = []
    try:
        if network == "proxy":
            net = f"{name}-int"
            proxy = f"{name}-proxy"
            docker("network", "create", "--internal", net)
            created.append(("network", net))
            created.append(("container", proxy))
            proxy_url = start_proxy(proxy, net)
            net_args = ["--network", net,
                        "-e", f"HTTPS_PROXY={proxy_url}", "-e", f"HTTP_PROXY={proxy_url}",
                        "-e", f"https_proxy={proxy_url}", "-e", f"http_proxy={proxy_url}",
                        "-e", "NO_PROXY=localhost,127.0.0.1"]
        elif network == "open":
            net_args = dns_args()
        else:
            raise ValueError(f"unknown network mode {network!r}")

        args = ["run", "-d", "--name", name, *net_args, *_user_args(),
                "--cpus", str(res["cpus"]), "--memory", res["memory"], "--pids-limit", "4096",
                "-v", f"{workspace.resolve()}:{CONTAINER_WORKSPACE}", "-w", CONTAINER_WORKSPACE]
        if gpu is not None:
            args += ["--gpus", f"device={gpu}"]
        for key in adapter.secret_env:
            if os.environ.get(key):
                args += ["-e", key]  # by name: docker reads the value from this process's env
        for key, value in adapter.env.items():
            args += ["-e", f"{key}={value}"]
        for host, container in adapter.mounts.items():
            args += ["-v", f"{host}:{container}:ro"]
        args += [task.agent_image, *adapter.command(task.spec["goal"], model)]

        started = time.time()
        docker(*args)
        created.append(("container", name))

        timed_out = False
        try:
            exit_code = int(docker("wait", name, timeout=res["wall_time_s"]).stdout.strip())
        except subprocess.TimeoutExpired:
            timed_out = True
            docker("kill", name, check=False)
            exit_code = int(docker("wait", name).stdout.strip() or -1)
        finished = time.time()

        with open(run_dir / "agent_stdout.jsonl", "w", encoding="utf-8") as out, \
                open(run_dir / "agent_stderr.txt", "w", encoding="utf-8") as err:
            subprocess.run(["docker", "logs", name], stdout=out, stderr=err, check=False)
        if network == "proxy":
            with open(run_dir / "proxy.log", "w", encoding="utf-8") as log:
                subprocess.run(["docker", "logs", f"{name}-proxy"], stdout=log, stderr=log, check=False)
        for f in ("agent_stdout.jsonl", "agent_stderr.txt"):
            _redact(run_dir / f, secrets)

        return {
            "container_exit_code": exit_code,
            "timed_out": timed_out,
            "started_at": started,
            "finished_at": finished,
            "wall_time_s": round(finished - started, 1),
        }
    finally:
        for kind, obj in reversed(created):
            if kind == "container":
                docker("rm", "-f", obj, check=False)
            else:
                docker("network", "rm", obj, check=False)


def agent_version(task: Task, adapter) -> str | None:
    cmd = adapter.version_command()
    if not cmd:
        return None
    out = docker("run", "--rm", "--network", "none", task.agent_image, *cmd, check=False)
    return (out.stdout or out.stderr).strip() or None
