#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"
if [[ "${1:-}" == "--dry-run" ]]; then
  exec python "$ROOT/scripts/matrix_runner.py" --profile paper_faithful --checkpoint-every "$CHECKPOINT_EVERY" --dry-run "${@:2}"
fi
if [[ "${1:-}" == "--execute" ]]; then
  exec python "$ROOT/scripts/matrix_runner.py" --profile paper_faithful --checkpoint-every "$CHECKPOINT_EVERY" --execute "${@:2}"
fi
echo "Usage: CHECKPOINT_EVERY=5 $0 --dry-run | --execute [--device auto|cpu|cuda:N] [--recover-empty-run]" >&2
exit 2
