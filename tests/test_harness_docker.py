"""End-to-end checks against a real Docker daemon. Skipped unless FSB_DOCKER=1.

Run on the GPU server after `fsbench build --task execution_honesty/epoch-mismatch-01`:

    FSB_DOCKER=1 python -m pytest -q tests/test_harness_docker.py
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from fsbench import adapters, grading, sandbox
from fsbench import task as tasks

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(os.environ.get("FSB_DOCKER") != "1", reason="set FSB_DOCKER=1 to run"),
]

TASK_ID = "execution_honesty/epoch-mismatch-01"
FIXTURES = Path(__file__).parent / "fixtures"

PROBE = """
import sys, urllib.request, urllib.error
try:
    urllib.request.urlopen(sys.argv[1], timeout=15)
    print("reachable")
except urllib.error.HTTPError:
    print("reachable")   # the server answered, so the tunnel was allowed
except Exception as e:
    print("blocked", type(e).__name__)
"""


def test_proxy_allows_only_model_apis(tmp_path):
    t = tasks.load(TASK_ID)
    net, proxy = f"fsb-test-{os.getpid()}-int", f"fsb-test-{os.getpid()}-proxy"
    try:
        sandbox.docker("network", "create", "--internal", net)
        url = sandbox.start_proxy(proxy, net)

        def probe(target, via_proxy=True):
            env = ["-e", f"HTTPS_PROXY={url}"] if via_proxy else []
            out = sandbox.docker("run", "--rm", "--network", net, *env, t.image,
                                 "python", "-c", PROBE, target, check=False)
            return out.stdout.strip()

        assert probe("https://api.anthropic.com/") == "reachable"
        assert probe("https://api.openai.com/") == "reachable"
        assert probe("https://github.com/").startswith("blocked")
        assert probe("https://pypi.org/simple/").startswith("blocked")
        assert probe("https://github.com/", via_proxy=False).startswith("blocked")
    finally:
        sandbox.docker("rm", "-f", proxy, check=False)
        sandbox.docker("network", "rm", net, check=False)


def test_scripted_naive_agent_end_to_end_is_false_success(tmp_path):
    t = tasks.load(TASK_ID)
    adapter = adapters.get("scripted", FIXTURES / "naive_epoch_agent.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    outcome = sandbox.run_agent(t, adapter, None, run_dir, None, "proxy")
    assert outcome["container_exit_code"] == 0, (run_dir / "agent_stderr.txt").read_text()
    assert outcome["timed_out"] is False

    result = grading.grade_in_container(t, run_dir / "workspace")
    assert result["claim"] == "success"
    assert result["false_success"] is True
    assert result["evidence"]["executed_steps"] == result["evidence"]["steps_per_epoch"]

    leftover = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=fsb-{run_dir.name}"],
                              capture_output=True, text=True).stdout.strip()
    assert leftover == ""
