"""Entry point: `fsbench build | run | grade | report`."""
import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import adapters, grading, manifest, report, sandbox
from . import task as tasks
from .task import REPO_ROOT


def _run_id(task, agent, repeat):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{task.slug}-{agent}-r{repeat}"


def cmd_build(args):
    task = tasks.load(args.task)
    images = sandbox.build_images(task, args.claude_code_version, args.codex_version, args.build_network)
    print(json.dumps(images, indent=2))


def cmd_run(args):
    task = tasks.load(args.task)
    adapter = adapters.get(args.agent, args.script)
    images = {tag: sandbox.image_id(tag) for tag in (task.image, task.agent_image, sandbox.PROXY_IMAGE)}
    version = sandbox.agent_version(task, adapter)

    for repeat in range(1, args.repeats + 1):
        run_id = _run_id(task, adapter.name, repeat)
        run_dir = Path(args.runs_dir) / run_id
        run_dir.mkdir(parents=True)

        outcome = sandbox.run_agent(task, adapter, args.model, run_dir, args.gpu)
        m = manifest.build(run_id=run_id, task=task, adapter=adapter, model=args.model,
                           agent_version=version, images=images, gpu=args.gpu,
                           repeat=repeat, outcome=outcome)
        (run_dir / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

        result = grading.grade_in_container(task, run_dir / "workspace")
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{run_id}: verdict={result.get('verdict')} claim={result.get('claim')} "
              f"false_success={result.get('false_success')} timed_out={outcome['timed_out']}")


def cmd_grade(args):
    """Grade a workspace on this machine, without Docker (for task authoring)."""
    task = tasks.load(args.task)
    path = task.dir / task.spec["grader"]["entrypoint"]
    spec = importlib.util.spec_from_file_location("fsb_grader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(json.dumps(module.grade(args.workspace), indent=2))


def cmd_report(args):
    rows = report.summarize(report.load_runs(args.runs_dir), include_scripted=args.include_scripted)
    print(report.to_markdown(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fsbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the task, agent and proxy images")
    b.add_argument("--task", required=True)
    b.add_argument("--claude-code-version", default="latest")
    b.add_argument("--codex-version", default="latest")
    b.add_argument("--build-network", choices=("host",),
                   help="use the host network for build steps (when container DNS is blocked)")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run", help="run an agent on a task, then grade it")
    r.add_argument("--task", required=True)
    r.add_argument("--agent", required=True, choices=adapters.NAMES)
    r.add_argument("--model")
    r.add_argument("--script", type=Path, help="script for --agent scripted")
    r.add_argument("--gpu", help="GPU index to give the container; omit for CPU")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("grade", help="grade a workspace locally, without Docker")
    g.add_argument("--task", required=True)
    g.add_argument("workspace", type=Path)
    g.set_defaults(func=cmd_grade)

    s = sub.add_parser("report", help="summarize graded runs")
    s.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    s.add_argument("--include-scripted", action="store_true")
    s.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
