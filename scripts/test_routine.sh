#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${1:-}" == "--live-smoke" ]]; then
  export FACTICLI_RUN_LIVE_SMOKE=1
fi

python3 -m compileall src
python3 -m unittest discover -s tests -p "test_*.py" -v
