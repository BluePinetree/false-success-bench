"""Synthetic run accounting; these fixtures are not agent benchmark results."""
import json
from argparse import Namespace

import pytest

from fsbench import cli, report


def _saved_run(root, run_id, result=None, **fields):
    run_dir = root / run_id
    run_dir.mkdir()
    manifest = dict(run_id=run_id, task_id="execution_honesty/epoch-mismatch-01",
                    agent="codex", model="m", network="proxy")
    manifest.update(fields)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if result is not None:
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return run_dir


def _result(claim, verdict):
    return dict(claim=claim, verdict=verdict,
                false_success=claim == "success" and verdict == "fail")


def test_missing_result_is_visible_without_writing_a_result(tmp_path):
    run_dir = _saved_run(tmp_path, "missing")
    runs = report.load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["result"] is None
    (row,) = report.summarize(runs)
    assert row["total_runs"] == row["missing_results"] == 1
    assert row["graded_runs"] == 0
    assert row["fsr"] is row["unconditional_false_success"] is None
    assert "missing_result" in report.per_run_markdown(runs)
    assert not (run_dir / "result.json").exists()


def test_mixed_run_accounting_and_denominators(tmp_path):
    for run_id, result in [
        ("false", _result("success", "fail")),
        ("pass", _result("success", "pass")),
        ("partial", _result("partial", "fail")),
        ("no-claim", _result("none", "fail")),
        ("grader-error", {"verdict": "grader_error"}),
        ("error-with-claim", _result("success", "grader_error")),
        ("missing", None),
    ]:
        _saved_run(tmp_path, run_id, result)
    _saved_run(tmp_path, "infra", _result("success", "fail"), infra_error="401 Unauthorized")
    # Infrastructure status takes precedence even when result.json is missing.
    _saved_run(tmp_path, "infra-missing", infra_error="401 Unauthorized")
    runs = report.load_runs(tmp_path)
    (row,) = report.summarize(runs)
    assert row["total_runs"] == 9
    assert row["runs"] == 7  # legacy: all non-infrastructure runs
    assert row["graded_runs"] == 4
    assert row["grader_errors"] == 2
    assert row["infra_errors_excluded"] == 2
    assert row["missing_results"] == 1
    assert row["total_runs"] == sum(row[k] for k in (
        "graded_runs", "grader_errors", "infra_errors_excluded", "missing_results"))
    assert row["grader_pass"] == 1 and row["grader_fail"] == 3
    assert row["success_claims"] == 2 and row["no_claim"] == 1
    assert row["false_success"] == 1
    assert row["fsr"] == 0.5
    assert row["unconditional_false_success"] == 0.25
    assert row["task_success_rate"] == 0.25
    assert row["self_reported_success_rate"] == 0.5
    md = report.to_markdown([row])
    assert "grader errors" in md and "missing results" in md
    assert "graded success claims" in md and "graded runs" in md
    detail = report.per_run_markdown(runs)
    assert len(detail.splitlines()) == 11  # header + separator + every manifest
    assert "grader_error" in detail and "infra_error" in detail


@pytest.mark.parametrize("result", [None, {"verdict": "grader_error"},
                                  _result("success", "grader_error")])
def test_no_graded_runs_has_no_performance_rates(tmp_path, result):
    _saved_run(tmp_path, "ungraded", result)
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["graded_runs"] == row["success_claims"] == row["false_success"] == 0
    for key in ("fsr", "unconditional_false_success", "task_success_rate",
                "self_reported_success_rate"):
        assert row[key] is None


def test_graded_failure_without_agent_report_is_not_missing_result(tmp_path):
    _saved_run(tmp_path, "no-agent-report", _result("none", "fail"))
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["graded_runs"] == row["grader_fail"] == row["no_claim"] == 1
    assert row["missing_results"] == 0
    assert row["fsr"] is None
    assert row["unconditional_false_success"] == 0


def test_infra_only_group_is_preserved(tmp_path):
    _saved_run(tmp_path, "infra", infra_error="401 Unauthorized")
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["total_runs"] == row["infra_errors_excluded"] == 1
    assert row["runs"] == row["graded_runs"] == row["missing_results"] == 0


def test_normal_graded_results_preserve_legacy_fields(tmp_path):
    for i, (claim, verdict) in enumerate([
        ("success", "fail"), ("success", "pass"), ("partial", "fail"), ("none", "fail")
    ]):
        _saved_run(tmp_path, str(i), _result(claim, verdict))
    (row,) = report.summarize(report.load_runs(tmp_path))
    expected = dict(agent="codex", model="m", network="proxy", runs=4, grader_pass=1,
                    success_claims=2, no_claim=1, false_success=1, fsr=0.5,
                    unconditional_false_success=0.25, grader_errors=0, infra_errors_excluded=0)
    assert {k: row[k] for k in expected} == expected


def test_regrade_recovers_missing_and_error_without_rerunning_agent(tmp_path, monkeypatch):
    missing = _saved_run(tmp_path, "missing")
    error = _saved_run(tmp_path, "error", {"verdict": "grader_error"})
    normal = _saved_run(tmp_path, "normal", _result("none", "fail"))
    original = (normal / "result.json").read_bytes()
    before = report.load_runs(tmp_path)
    monkeypatch.setattr(cli.tasks, "load", lambda task_id: task_id)
    calls = []

    def grade(task, workspace):
        calls.append(workspace)
        return _result("success", "fail")

    monkeypatch.setattr(cli.grading, "grade_in_container", grade)
    monkeypatch.setattr(cli.manifest, "repo_state", lambda: {"commit": "test"})
    cli.cmd_regrade(Namespace(runs_dir=tmp_path, all=False))
    assert set(calls) == {missing / "workspace", error / "workspace"}
    assert (normal / "result.json").read_bytes() == original
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["total_runs"] == row["graded_runs"] == 3
    assert row["missing_results"] == row["grader_errors"] == 0
    assert row["fsr"] == 1.0
    assert row["unconditional_false_success"] == pytest.approx(2 / 3)
    for run_dir in (missing, error):
        assert json.loads((run_dir / "result.json").read_text())["graded_by"]["regraded"]
    # Loading a missing result never persisted a placeholder; this snapshot stays missing.
    assert next(r for r in before if r["run_id"] == "missing")["result"] is None


def test_empty_and_scripted_only_reports(tmp_path):
    assert report.summarize(report.load_runs(tmp_path)) == []
    _saved_run(tmp_path, "scripted", agent="scripted")
    assert report.summarize(report.load_runs(tmp_path)) == []
    (row,) = report.summarize(report.load_runs(tmp_path), include_scripted=True)
    assert row["missing_results"] == row["total_runs"] == 1


@pytest.mark.parametrize("result", [{}, {"verdict": "unknown", "claim": "success", "false_success": True}])
def test_invalid_verdict_does_not_enter_performance_denominators(tmp_path, result):
    _saved_run(tmp_path, "invalid-verdict", result)
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["total_runs"] == row["grader_errors"] == 1
    assert row["graded_runs"] == row["false_success"] == row["success_claims"] == 0
    assert row["fsr"] is row["unconditional_false_success"] is None


def test_error_only_and_graded_groups_remain_separate(tmp_path):
    _saved_run(tmp_path, "graded", _result("success", "fail"))
    _saved_run(tmp_path, "missing", model="other")
    _saved_run(tmp_path, "error", {"verdict": "grader_error"}, network="offline")
    rows = report.summarize(report.load_runs(tmp_path))
    assert len(rows) == 3
    assert sum(r["total_runs"] for r in rows) == 3
    graded = next(r for r in rows if r["graded_runs"])
    assert graded["fsr"] == graded["unconditional_false_success"] == 1
    assert all(r["fsr"] is None for r in rows if r is not graded)


@pytest.mark.parametrize("per_run", [False, True])
def test_report_cli_exposes_missing_and_grader_errors(tmp_path, capsys, per_run):
    _saved_run(tmp_path, "missing")
    _saved_run(tmp_path, "error", {"verdict": "grader_error"})
    argv = ["report", "--runs-dir", str(tmp_path)]
    if per_run:
        argv.append("--per-run")
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert ("missing_result" if per_run else "missing results") in text
    assert ("grader_error" if per_run else "grader errors") in text


def test_false_success_is_derived_from_graded_claim_and_verdict(tmp_path):
    _saved_run(tmp_path, "false", {**_result("success", "fail"), "false_success": False})
    _saved_run(tmp_path, "partial", {**_result("partial", "fail"), "false_success": True})
    (row,) = report.summarize(report.load_runs(tmp_path))
    assert row["false_success"] == row["success_claims"] == 1
    assert row["fsr"] == 1
    detail = report.per_run_markdown(report.load_runs(tmp_path))
    assert "| success | fail | True |" in detail
    assert "| partial | fail | False |" in detail
