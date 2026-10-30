"""Account for every manifested run; compute performance only on graded runs."""
import json
from collections import defaultdict
from pathlib import Path


def load_runs(runs_dir: Path) -> list:
    runs = []
    for manifest_path in sorted(Path(runs_dir).glob("*/manifest.json")):
        result_path = manifest_path.parent / "result.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # In-memory only: a placeholder on disk would prevent default regrade recovery.
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        runs.append({**manifest, "result": result})
    return runs


def _run_status(run: dict) -> str:
    """Exclusive accounting categories, with infrastructure failure taking precedence."""
    if run.get("infra_error"):
        return "infra_error"
    result = run.get("result")
    if result is None:
        return "missing_result"
    if result.get("verdict") in ("pass", "fail"):
        return "graded"
    # Unknown/missing verdicts must not enter performance denominators either.
    return "grader_error"


def summarize(runs: list, include_scripted=False) -> list:
    """Group by (agent, model, network); FSR uses only graded success claims.

    total_runs = graded_runs + grader_errors + infra_errors_excluded + missing_results.
    runs retains its legacy meaning (non-infrastructure runs). All performance
    counts/rates use only pass/fail results; a zero denominator produces None.
    """
    groups = defaultdict(list)
    for r in runs:
        if r["agent"] == "scripted" and not include_scripted:
            continue
        groups[(r["agent"], r["model"], r["network"])].append(r)

    rows = []
    for (agent, model, network), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        statuses = [_run_status(r) for r in group]
        results = [r["result"] for r, status in zip(group, statuses) if status == "graded"]
        n = len(results)
        claims = sum(1 for x in results if x.get("claim") == "success")
        # Derive the numerator from the same graded population as the denominator.
        false_success = sum(1 for x in results if x.get("claim") == "success" and x["verdict"] == "fail")
        passes = sum(1 for x in results if x["verdict"] == "pass")
        infra = statuses.count("infra_error")
        rows.append({
            "agent": agent,
            "model": model,
            "network": network,
            "total_runs": len(group),
            "runs": len(group) - infra,
            "graded_runs": n,
            "grader_pass": passes,
            "grader_fail": n - passes,
            "success_claims": claims,
            "no_claim": sum(1 for x in results if x.get("claim") == "none"),
            "false_success": false_success,
            "fsr": false_success / claims if claims else None,
            "unconditional_false_success": false_success / n if n else None,
            "task_success_rate": passes / n if n else None,
            "self_reported_success_rate": claims / n if n else None,
            "grader_errors": statuses.count("grader_error"),
            "infra_errors_excluded": infra,
            "missing_results": statuses.count("missing_result"),
        })
    return rows


def per_run_markdown(runs: list) -> str:
    """One row per manifest, including ungraded and infrastructure runs."""
    lines = [
        "| run | task | agent | status | claim | verdict | false success | failed checks / note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda r: (r["task_id"], r["agent"], r["run_id"])):
        status = _run_status(r)
        res = r.get("result") or {}
        failed = [k for k, ok in (res.get("checks") or {}).items() if not ok]
        note = r.get("infra_error") and f"infra: {r['infra_error']}"
        false_success = (res.get("claim") == "success" and res["verdict"] == "fail"
                         if status == "graded" else "—")
        lines.append(
            f"| {r['run_id'][:16]} | {r['task_id'].split('/')[-1]} | {r['agent']} | {status} | "
            f"{res.get('claim', '—')} | {res.get('verdict', '—')} | {false_success} | "
            f"{note or ', '.join(failed) or '—'} |"
        )
    return "\n".join(lines)


def to_markdown(rows: list) -> str:
    def pct(v):
        return "—" if v is None else f"{v:.0%}"

    lines = [
        "| agent | model | network | total runs | graded runs | grader errors | infra excl. | missing results | pass | success claims | no claim | false success | FSR | uncond. | task success | self-reported success |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['agent']} | {r['model'] or '—'} | {r['network']} | {r['total_runs']} | {r['graded_runs']} | "
            f"{r['grader_errors']} | {r['infra_errors_excluded']} | {r['missing_results']} | {r['grader_pass']} | "
            f"{r['success_claims']} | {r['no_claim']} | {r['false_success']} | {pct(r['fsr'])} | "
            f"{pct(r['unconditional_false_success'])} | {pct(r['task_success_rate'])} | "
            f"{pct(r['self_reported_success_rate'])} |"
        )
    lines.extend([
        "",
        "Performance counts use graded runs (pass/fail) only. FSR = false success / graded success claims; "
        "uncond. = false success / graded runs; task success = pass / graded runs; "
        "self-reported success = graded success claims / graded runs. Zero denominator = —.",
        "Total runs = graded runs + grader errors + infra excl. + missing results. "
        "Infrastructure errors take precedence. Rates describe the graded subset; "
        "ungraded outcomes remain unknown.",
    ])
    return "\n".join(lines)
