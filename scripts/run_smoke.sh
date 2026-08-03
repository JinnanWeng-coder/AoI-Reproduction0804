#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT/Main.py" --profile paper_faithful --scenario p05_n04_g25 --seed 2 --device "${1:-cpu}" --smoke --run-name "${2:-smoke_paper_faithful_p05_n04_g25_seed02}"

