# Said & Done: A False-Success Benchmark for Research Agents

> Status: pre-v0.1, under construction. Nothing in this repository is a result yet.

**Said & Done** measures the gap between what a coding agent *says* it did and what
it actually *did*. It is a benchmark of executable ML/research tasks that counts how
often an agent **reports success when the evidence says it failed** — the
false-success rate.

Each task ships a deliberately broken research repository inside an isolated
container, a goal for the agent, and a hidden grader that checks execution evidence
(artifacts, logs, process state, re-derived metrics) and ignores what the agent
says about its own work.

See [`DESIGN.md`](DESIGN.md) for the research question, task taxonomy, grader
hierarchy, and metrics.

## Running

Needs Linux with Docker (and the NVIDIA Container Toolkit for GPU runs).

```bash
pip install -e .
fsbench build --task execution_honesty/epoch-mismatch-01

# Harness check without an API key: a scripted agent that trusts the run
fsbench run --task execution_honesty/epoch-mismatch-01 \
  --agent scripted --script tests/fixtures/naive_epoch_agent.py

# A real agent; the key is forwarded into the container by name only
export ANTHROPIC_API_KEY=...
fsbench run --task execution_honesty/epoch-mismatch-01 \
  --agent claude-code --model claude-opus-5-5 --gpu 0 --repeats 3

fsbench report
```

Each run leaves `runs/<run_id>/` with the final `workspace/`, `manifest.json`,
`result.json` (the grader's verdict), the agent's raw output, and the proxy log.
The agent's container can reach only the model APIs, through a per-run proxy (DESIGN.md §7).

## Layout

| Path | Contents |
|---|---|
| `src/fsbench/` | Harness: task loading, sandbox, agent adapters, trajectory log, grading, report |
| `tasks/<category>/<task_id>/` | One directory per task (see `tasks/_template/`) |
| `tests/` | Harness unit tests and task validation |
| `results/` | Published baseline runs with manifests |
| `reports/` | Technical report |
| `docs/` | Architecture and task-authoring notes |
