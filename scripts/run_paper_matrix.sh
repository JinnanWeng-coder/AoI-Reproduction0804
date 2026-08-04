#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  exec python "$ROOT/scripts/matrix_runner.py" --profile paper_faithful --dry-run "${@:2}"
fi
if [[ "${1:-}" == "--execute" ]]; then
  exec python "$ROOT/scripts/matrix_runner.py" --profile paper_faithful --execute "${@:2}"
fi
echo "Usage: $0 --dry-run | --execute [--device auto|cpu|cuda:N]" >&2
exit 2
