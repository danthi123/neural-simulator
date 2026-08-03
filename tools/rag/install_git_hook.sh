#!/usr/bin/env bash
# Arm the repository-controlled hooks and verify the complete RAG workflow.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
PY=${PYTHON:-python3}
exec "$PY" tools/rag/check_workflow.py --install
