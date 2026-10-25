#!/usr/bin/env bash
# Build the task, agent and proxy images for every task.
#
#   scripts/build_all.sh [--build-network host]
set -euo pipefail
cd "$(dirname "$0")/.."

for task_json in tasks/*/*/task.json; do
  task_dir=$(dirname "$task_json")
  [[ $task_dir == tasks/_template ]] && continue
  echo "== ${task_dir#tasks/}"
  fsbench build --task "${task_dir#tasks/}" "$@"
done
