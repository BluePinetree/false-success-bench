# Report accounting validation

These are **synthetic regression checks**, not measurements of Codex or Claude
Code. The comparison base is commit `06bc685b1b3078ac27e82ef9d0dd0ce56c8d6741`.
No agents, model APIs, GPUs, or baseline workspaces were used.

## Mixed synthetic fixture: before and after

The same nine manifests were loaded and summarized by the original module from
`git show 06bc685b:src/fsbench/report.py` and the
modified module. `test_mixed_run_accounting_and_denominators` reproduces the
fixture: four graded runs (one false success, one pass, one partial failure, one
no-claim failure), two grader errors (one with a saved success claim), one missing
result, and two infrastructure errors (one also missing its result).

| Quantity | Before | After |
|---|---:|---:|
| Manifests loaded | 7 | 9 |
| Total run count field | absent | 9 |
| `runs` (non-infrastructure) | 6 | 7 |
| Graded run count field | absent | 4 |
| Grader errors | 2 | 2 |
| Infrastructure errors excluded | 1 | 2 |
| Missing results, excluding infrastructure | silently omitted | 1 |
| Success claims used for FSR | 3 | 2 |
| False successes | 1 | 1 |
| FSR | 1/3 = 33.33% | 1/2 = 50% |
| Unconditional false-success | 1/6 = 16.67% | 1/4 = 25% |
| Task success rate | not emitted | 1/4 = 25% |
| Self-reported success rate | not emitted | 2/4 = 50% |

The error fixture with a saved claim deliberately exercises denominator safety;
the current grader's normal error path emits only a verdict and diagnostics.
Even an error without a claim diluted the old unconditional denominator.
The new rates describe the graded subset, not the unknown outcomes.

## Compatibility and recovery

- The original four-result, normally graded fixture preserves every existing
  summary field exactly: runs=4, pass=1, claims=2, no-claim=1, false-success=1,
  FSR=0.5, unconditional=0.25, grader-errors=0, infra-excluded=0.
- Missing agent reports with a completed failing grade remain graded no-claim
  runs. Missing harness results remain ungraded.
- Missing-only, error-only, infrastructure-only, empty, grouped, scripted, and
  invalid-verdict cases have explicit regression checks. Undefined rates are
  `None`, not zero. Both report CLI modes expose incomplete outcomes.
- A mocked grader integration test invokes the existing default `cmd_regrade`:
  one missing and one error result become graded, a normal result is unchanged
  byte for byte, and regrade provenance is retained. No placeholder is written
  by report loading. This is not a live Docker or agent execution test.

## Test results

Before changing the report module, the initial regression tests together with
`tests/test_harness_units.py` produced **9 failed, 15 passed**. The original normal
aggregation compatibility assertion passed. After the fix, including six extra
edge/CLI consistency cases, these files produced **30 passed**.

The full repository suite completed with **76 passed, 2 skipped** in 436.14 s
using Python 3.12, CPU PyTorch 2.15.0, and scikit-learn 1.8.0. The two
Docker integration checks were skipped by their existing opt-in rule. The final
cached-flag consistency regression was added after full-suite collection and is
included in the final 30-test targeted run above. No implementation changed
after the full-suite run began. Task validation/training tests are not agent
benchmark measurements.

Run the relevant checks with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_report_accounting.py tests/test_harness_units.py
PYTHONPATH=src python -m pytest -q
```

## Actual benchmark measurements

There are no published run manifests/results in this checkout's `results/`, and
no private baseline-v0 workspaces were supplied. No actual benchmark performance
was recomputed. The historical regrading mentioned in DESIGN.md is not an
independently verified measurement in this validation.
