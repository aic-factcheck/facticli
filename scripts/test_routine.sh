#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python3"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ "${1:-}" == "--live-smoke" ]]; then
  export FACTICLI_RUN_LIVE_SMOKE=1
fi

"$PYTHON_BIN" -m compileall src
"$PYTHON_BIN" -m unittest discover -s tests -p "test_*.py" -v
