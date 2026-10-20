"""Harness logic that can be checked without Docker: adapters, run arguments, report."""
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

    def __call__(self, *args, timeout=None, check=True):
        self.calls.append(list(args))
        stdout = "0\n" if args[0] == "wait" else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")


@pytest.fixture
def fake_docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(sandbox, "docker", fake)
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    return fake


def _run_args(fake):
    return next(c for c in fake.calls if c[0] == "run" and "-w" in c)


def test_run_forwards_secrets_by_name_only(fake_docker, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("claude-code"), "m", tmp_path / "run", None, "proxy")
    flat = " ".join(" ".join(c) for c in fake_docker.calls)
    assert "sk-ant-secret-value" not in flat
    args = _run_args(fake_docker)
    assert args[args.index("ANTHROPIC_API_KEY") - 1] == "-e"


def test_run_never_mounts_grader_or_reference(fake_docker, tmp_path):
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", "0", "proxy")
    args = _run_args(fake_docker)
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
    assert mounts == [f"{(tmp_path / 'run' / 'workspace').resolve()}:/workspace"]
    hidden = [(t.dir / d).resolve() for d in ("grader", "reference")]
    assert not any(str(h) in m for h in hidden for m in mounts)
    assert args[args.index("--gpus") + 1] == "device=0"


def test_proxy_mode_uses_internal_network_and_cleans_up(fake_docker, tmp_path):
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None, "proxy")
    create = next(c for c in fake_docker.calls if c[:2] == ["network", "create"])
    assert "--internal" in create
    removed = [c for c in fake_docker.calls if c[:2] in (["rm", "-f"], ["network", "rm"])]
    assert len(removed) == 3  # agent container, proxy container, network


def test_timeout_kills_the_container(fake_docker, tmp_path, monkeypatch):
    def wait_times_out(*args, timeout=None, check=True):
        if args[0] == "wait" and timeout is not None:
            raise subprocess.TimeoutExpired(args, timeout)
        return FakeDocker.__call__(fake_docker, *args, timeout=timeout, check=check)

    monkeypatch.setattr(sandbox, "docker", wait_times_out)
    t = tasks.load(TASK_ID)
    outcome = sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None, "open")
    assert outcome["timed_out"] is True
    assert any(c[0] == "kill" for c in fake_docker.calls)


def test_dns_override_applies_only_to_containers_that_go_outside(fake_docker, tmp_path, monkeypatch):
    monkeypatch.setenv("FSB_DNS", "10.0.0.53, 10.0.0.54")
    t = tasks.load(TASK_ID)
    sandbox.run_agent(t, adapters.get("codex"), None, tmp_path / "run", None, "proxy")
    proxy_run = next(c for c in fake_docker.calls if c[0] == "run" and sandbox.PROXY_IMAGE in c)
    assert proxy_run.count("--dns") == 2 and "10.0.0.54" in proxy_run
    assert "--dns" not in _run_args(fake_docker)  # the agent never resolves outside names itself


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
