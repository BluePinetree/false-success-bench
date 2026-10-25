#!/usr/bin/env bash
# Run one agent on every task, N times each, into one runs directory.
#
#   scripts/run_baseline.sh AGENT MODEL GPU REPEATS RUNS_DIR
#   scripts/run_baseline.sh claude-code claude-opus-5-5 0 3 runs/baseline-v0
#   scripts/run_baseline.sh codex gpt-6.1-sol 1 3 runs/baseline-v0
#
# Build images first: scripts/build_all.sh. Each task's runs are independent; a failed
# run is recorded and the loop continues.
set -uo pipefail

agent=$1 model=$2 gpu=$3 repeats=$4 runs_dir=$5
cd "$(dirname "$0")/.."

for task_json in tasks/*/*/task.json; do
  task_dir=$(dirname "$task_json")
  [[ $task_dir == tasks/_template ]] && continue
  task=${task_dir#tasks/}
  echo "== $task ($agent, $model, gpu $gpu, x$repeats)"
  fsbench run --task "$task" --agent "$agent" --model "$model" --gpu "$gpu" \
    --repeats "$repeats" --runs-dir "$runs_dir" || echo "!! $task: harness error, continuing"
done
