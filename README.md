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

## Layout

| Path | Contents |
|---|---|
| `src/fsbench/` | Harness: task loading, sandbox, agent adapters, trajectory log, grading, report |
| `tasks/<category>/<task_id>/` | One directory per task (see `tasks/_template/`) |
| `tests/` | Harness unit tests and task validation |
| `results/` | Published baseline runs with manifests |
| `reports/` | Technical report |
| `docs/` | Architecture and task-authoring notes |
