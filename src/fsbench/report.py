"""Aggregate results: success, self-reported success, false-success, variance and CIs."""
import json
from collections import defaultdict
from pathlib import Path


def load_runs(runs_dir: Path) -> list:
    runs = []
    for manifest_path in sorted(Path(runs_dir).glob("*/manifest.json")):
        result_path = manifest_path.parent / "result.json"
        if not result_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        runs.append({**manifest, "result": result})
    return runs


def summarize(runs: list, include_scripted=False) -> list:
    """One row per (agent, model, network). FSR = false successes / success claims (DESIGN.md §6)."""
    groups = defaultdict(list)
    for r in runs:
        if r["agent"] == "scripted" and not include_scripted:
            continue
        groups[(r["agent"], r["model"], r["network"])].append(r["result"])

    rows = []
    for (agent, model, network), results in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(results)
        claims = sum(1 for x in results if x.get("claim") == "success")
        false_success = sum(1 for x in results if x.get("false_success") is True)
        rows.append({
            "agent": agent,
            "model": model,
            "network": network,
            "runs": n,
            "grader_pass": sum(1 for x in results if x.get("verdict") == "pass"),
            "success_claims": claims,
            "no_claim": sum(1 for x in results if x.get("claim") == "none"),
            "false_success": false_success,
            "fsr": false_success / claims if claims else None,
            "unconditional_false_success": false_success / n if n else None,
            "grader_errors": sum(1 for x in results if x.get("verdict") == "grader_error"),
        })
    return rows


def to_markdown(rows: list) -> str:
    def pct(v):
        return "—" if v is None else f"{v:.0%}"

    lines = [
        "| agent | model | network | runs | pass | success claims | no claim | false success | FSR | uncond. |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['agent']} | {r['model'] or '—'} | {r['network']} | {r['runs']} | {r['grader_pass']} | "
            f"{r['success_claims']} | {r['no_claim']} | {r['false_success']} | {pct(r['fsr'])} | "
            f"{pct(r['unconditional_false_success'])} |"
        )
    return "\n".join(lines)
