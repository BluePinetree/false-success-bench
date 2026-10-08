# <task_id>

- **Category:**
- **Provenance:** which real failure this task comes from
- **Why realistic:**
- **Why gradable:**
- **Known ambiguity:**
- **Expected failure mode:**

| Directory | Contents | Visible to agent |
|---|---|---|
| `env/` | Dockerfile + broken repository | yes |
| `grader/` | Hidden grader entrypoint | **no** |
| `reference/` | Human solution, used to check the grader accepts a valid fix | **no** |
