"""Harness logic that can be checked without Docker: adapters, run arguments, report."""
import os
import subprocess
from pathlib import Path

import pytest

from fsbench import adapters, report, sandbox
from fsbench import task as tasks

TASK_ID = "execution_honesty/epoch-mismatch-01"
FIXTURES = Path(__file__).parent / "fixtures"


def test_task_loads_with_image_tags():
    t = tasks.load(TASK_ID)
    assert t.image == "fsb-task/epoch-mismatch-01:0.1.0"
    assert t.agent_image == "fsb-agent/epoch-mismatch-01:0.1.0"


def test_unknown_task_raises():
    with pytest.raises(FileNotFoundError):
        tasks.load("execution_honesty/does-not-exist-01")


@pytest.mark.parametrize("name", ["claude-code", "codex"])
def test_cli_adapters_pass_goal_and_model(name):
    a = adapters.get(name)
    cmd = a.command("do the task", "some-model")
    assert "do the task" in cmd
    assert cmd[cmd.index("--model") + 1] == "some-model"
    assert a.secret_env


def test_scripted_adapter_mounts_its_script_read_only():
    a = adapters.get("scripted", FIXTURES / "naive_epoch_agent.py")
    assert list(a.mounts.values()) == ["/fsb-script"]
    assert a.command("ignored", None) == ["python", "/fsb-script/naive_epoch_agent.py"]


class FakeDocker:
    """Records docker calls instead of running them."""

    def __init__(self):
        self.calls = []
        self.envs = []

    def __call__(self, *args, timeout=None, check=True, env=None):
        self.envs.append(env)
        self.calls.append(list(args))
        if args[0] == "wait":
            stdout = "0\n"
        elif args[:2] == ("network", "inspect"):
            stdout = "172.30.0.1 172.30.0.0/16\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(sandbox, "docker", fake)
    monkeypatch.setattr(sandbox, "wait_listening", lambda host, port: None)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    return fake


def _run_args(fake):
    return next(c for c in fake.calls if c[0] == "run" and "-w" in c)


def test_run_forwards_secrets_by_name_only(fake_docker, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("claude-code"), "m", tmp_path / "run", None)
    flat = " ".join(" ".join(c) for c in fake_docker.calls)
    assert "sk-ant-secret-value" not in flat
    args = _run_args(fake_docker)
    assert args[args.index("ANTHROPIC_API_KEY") - 1] == "-e"


def test_run_never_mounts_grader_or_reference(fake_docker, tmp_path):
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", "0")
    args = _run_args(fake_docker)
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
    assert mounts == [f"{(tmp_path / 'run' / 'workspace').resolve()}:/workspace"]
    hidden = [(t.dir / d).resolve() for d in ("grader", "reference")]
    assert not any(str(h) in m for h in hidden for m in mounts)
    assert args[args.index("--gpus") + 1] == "device=0"


def test_run_uses_internal_network_and_cleans_up(fake_docker, tmp_path):
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None)
    create = next(c for c in fake_docker.calls if c[:2] == ["network", "create"])
    assert "--internal" in create
    removed = [c for c in fake_docker.calls if c[:2] in (["rm", "-f"], ["network", "rm"])]
    assert len(removed) == 3  # agent container, proxy container, network


def test_timeout_kills_the_container(fake_docker, tmp_path, monkeypatch):
    def wait_times_out(*args, timeout=None, check=True, env=None):
        if args[0] == "wait" and timeout is not None:
            raise subprocess.TimeoutExpired(args, timeout)
        return FakeDocker.__call__(fake_docker, *args, timeout=timeout, check=check, env=env)

    monkeypatch.setattr(sandbox, "docker", wait_times_out)
    t = tasks.load(TASK_ID)
    outcome = sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None)
    assert outcome["timed_out"] is True
    assert any(c[0] == "kill" for c in fake_docker.calls)


def test_codex_key_alias_is_forwarded_by_name(fake_docker, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    t = tasks.load(TASK_ID)
    outcome = sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None)
    args = _run_args(fake_docker)
    assert "CODEX_API_KEY" in args and "OPENAI_API_KEY" in args
    assert not any("sk-openai-secret" in a for c in fake_docker.calls for a in c)
    run_env = fake_docker.envs[fake_docker.calls.index(args)]
    assert run_env["CODEX_API_KEY"] == "sk-openai-secret"
    assert "CODEX_API_KEY" not in os.environ  # the harness's own environment is untouched
    assert outcome["secret_env_forwarded"] == ["OPENAI_API_KEY", "CODEX_API_KEY"]


def test_auth_failure_without_report_is_an_infra_error(fake_docker, tmp_path, monkeypatch):
    def fake_logs(cmd, stdout=None, stderr=None, check=False):
        if stdout is not None and cmd[:2] == ["docker", "logs"] and not cmd[2].endswith("-proxy"):
            stdout.write('{"type":"turn.failed","error":{"message":"401 Unauthorized: Missing bearer"}}\n')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_logs)
    t = tasks.load(TASK_ID)
    outcome = sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None)
    assert outcome["infra_error"] == "401 Unauthorized"


def test_report_excludes_infra_errors():
    runs = [_run("codex", "success", "pass", False),
            {**_run("codex", "none", "fail", False), "infra_error": "401 Unauthorized"}]
    (row,) = report.summarize(runs)
    assert row["runs"] == 1
    assert row["infra_errors_excluded"] == 1


def test_proxy_listens_only_on_the_runs_internal_gateway(fake_docker, tmp_path):
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None)
    proxy_run = next(c for c in fake_docker.calls if c[0] == "run" and sandbox.PROXY_IMAGE in c)
    assert proxy_run[proxy_run.index("--network") + 1] == "host"
    assert "FSB_PROXY_LISTEN=172.30.0.1" in proxy_run
    assert "FSB_PROXY_ALLOW=172.30.0.0/16" in proxy_run
    agent = _run_args(fake_docker)
    assert agent[agent.index("--network") + 1].endswith("-int")
    assert "HTTPS_PROXY=http://172.30.0.1:8888" in agent


def _run(agent, claim, verdict, false_success, network="proxy"):
    return {"agent": agent, "model": "m", "network": network,
            "result": {"claim": claim, "verdict": verdict, "false_success": false_success}}


def test_report_fsr_uses_success_claims_as_denominator():
    runs = [
        _run("codex", "success", "fail", True),
        _run("codex", "success", "pass", False),
        _run("codex", "partial", "fail", False),
        _run("codex", "none", "fail", False),
        _run("scripted", "success", "fail", True),
    ]
    (row,) = report.summarize(runs)
    assert row["runs"] == 4
    assert row["success_claims"] == 2
    assert row["fsr"] == 0.5
    assert row["unconditional_false_success"] == 0.25
    assert row["no_claim"] == 1
