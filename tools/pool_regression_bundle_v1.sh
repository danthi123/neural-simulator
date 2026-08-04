#!/usr/bin/env bash
# Read-only CPU regression bundle for rsynced pool40/41/42 source copies.
set -euo pipefail

if (( $# != 0 )); then
  echo "usage: bash tools/pool_regression_bundle_v1.sh" >&2
  echo "This bundle accepts no arguments or seeds." >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "pool regression bundle: missing deployed interpreter: $PYTHON" >&2
  exit 1
fi

# Keep the source copy read-only: pytest state and bytecode do not belong in it.
export LC_ALL=C
export TZ=UTC
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export SIM_BACKEND=numpy

cd "$ROOT"

echo "[pool-regression-v1] document structure"
"$PYTHON" tools/check_docs.py

echo "[pool-regression-v1] lifecycle, documentation, and Stage-0 controller"
"$PYTHON" -m pytest -q -x -p no:cacheprovider \
  tests/test_pool_regression_bundle_v1.py \
  tests/test_experiment_automation_lifecycle.py \
  tests/test_doc_rules.py \
  tests/test_v13_stage0_controller.py \
  tests/test_v13_stage0_manifest.py

echo "[pool-regression-v1] PASS"
