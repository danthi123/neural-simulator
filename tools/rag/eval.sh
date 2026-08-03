#!/usr/bin/env bash
# Worktree-safe launcher for the accumulating RAG quality evaluation.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
COMMON=$(git rev-parse --path-format=absolute --git-common-dir)
CANONICAL=$(dirname "$COMMON")
PY=${SIM_RAG_PYTHON:-$CANONICAL/.venv-rag/bin/python}
if [ ! -x "$PY" ]; then
  echo "RAG interpreter missing: $PY" >&2
  exit 1
fi
exec "$PY" "$REPO/tools/rag/rag_eval.py" "$@"
