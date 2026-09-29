#!/usr/bin/env bash
# gpu_add.sh — the ONLY way the local model may queue a GPU job (tools/gpu_queue.sh add itself stays denied to it,
# see tools/local_llm/claude_local_settings.json). Two checks on top of gpu_queue.sh's own enqueue-time shape check
# (tools/queue_job_shape_check.sh, which only asks "would this even start"):
#
#   1. the job must `cd` into a git checkout that is at a COMMITTED, CLEAN revision -- so whatever actually runs on
#      the 3090 is exactly what is on disk in the repo's history: reproducible, and reviewable by Claude from the
#      commit alone. A job with no `cd` at all, or one whose checkout has uncommitted changes, is refused outright.
#   2. if the job passes a GATE seed (42/43/44/100/101/102, CLAUDE.md's six) as --seed/--seeds, it must also carry
#      --prereg <path> (or a `PREREG=<path>` assignment in the job string itself, the same style as the existing
#      `SIM_BACKEND=cupy ...` prefix convention) naming a file that is ITSELF committed in that same checkout at
#      that same revision -- CLAUDE.md's own guardrail table (docs/plans/2026-09-29-local-model-transition-PLAN.md
#      section 4): "ran GATE seeds without a committed prereg" (2026-09-28).
#
# Then it forwards the exact command, unchanged, to `tools/gpu_queue.sh add`.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPUQ="${GPU_ADD_QUEUE_SH:-$ROOT/tools/gpu_queue.sh}"   # override is TEST-ONLY; NEVER set in production
CMD="${1:-}"

usage() { echo "usage: gpu_add.sh '<full gpu command incl. --json out>'" >&2; exit 2; }
[ -n "$CMD" ] || usage

# 1. The job must start with `cd <checkout> && ...` (or `; ...` / `| ...`) -- the first statement, and nothing
#    fancier: this is a SHAPE check like queue_job_shape_check.sh's own, not a shell parser.
first_stmt=$(printf '%s' "$CMD" | sed -E 's/(&&|;|\|).*//')
cd_path=""
case "$first_stmt" in
  cd\ *) cd_path=$(printf '%s' "$first_stmt" | awk '{print $2}') ;;
esac
if [ -z "$cd_path" ]; then
  echo "⛔ REFUSED: the job must start with 'cd <checkout> && ...' -- gpu_add.sh only runs jobs against a specific, checked git checkout (never the dispatcher's own ambient cwd)." >&2
  exit 1
fi
# EXPAND ~ / $HOME / $ROOT-STYLE REFERENCES WITHOUT A SHELL (2026-09-29 security review, H2: `eval echo
# $cd_path` ran the CALLER-SUPPLIED path through a SECOND round of full shell parsing -- a cd_path like
# `$(id>/tmp/pwned)` (one whitespace-free token, so it survives the awk field-split above intact) made `eval`
# execute `id` and redirect its output, not merely "expand a variable". python's os.path.expandvars only
# textually substitutes `$NAME`/`${NAME}` references from the environment (regex requires a bare word or a
# `{...}` immediately after `$` -- `$(...)` matches neither and is left completely untouched, byte for byte,
# never interpreted as shell syntax); os.path.expanduser handles a leading `~`. No shell is ever invoked.
cd_path_expanded=$(python3 -c '
import os, sys
print(os.path.expanduser(os.path.expandvars(sys.argv[1])))
' "$cd_path" 2>/dev/null || true)
if [ -z "$cd_path_expanded" ] || [ ! -d "$cd_path_expanded" ] || ! git -C "$cd_path_expanded" rev-parse HEAD >/dev/null 2>&1; then
  echo "⛔ REFUSED: '$cd_path' is not a git checkout." >&2
  exit 1
fi
if [ -n "$(git -C "$cd_path_expanded" status --porcelain 2>/dev/null)" ]; then
  echo "⛔ REFUSED: '$cd_path' has uncommitted changes -- the job must run against a COMMITTED, clean revision." >&2
  exit 1
fi
rev=$(git -C "$cd_path_expanded" rev-parse HEAD)

# 2. GATE seeds require a committed prereg. Parsing is deliberately narrow (python, not a bash regex maze): find
#    --seed/--seeds' value(s) and --prereg's value / a PREREG= assignment; a value this can't confidently read is
#    never silently accepted as "no gate seed" (see GATE_HIT default below).
# SHARED WITH tools/agent_gateway_submit.py (2026-09-29 security review, H2): this used to be an embedded
# heredoc here; it now lives once in tools/gate_seed_guard.py so the structured gpu-submit/pool-submit path
# enforces the identical rule instead of a second, potentially-diverging copy of it.
PY_OUT=$(python3 "$ROOT/tools/gate_seed_guard.py" "$CMD")
read -r GATE_HIT PREREG_PATH <<< "$PY_OUT"

if [ "$GATE_HIT" = "1" ]; then
  if [ "$PREREG_PATH" = "-" ] || [ -z "$PREREG_PATH" ]; then
    echo "⛔ REFUSED: this job runs a GATE seed (42/43/44/100/101/102) but carries no --prereg <path> (or PREREG=<path>) -- CLAUDE.md non-negotiable, guardrail table row 'ran GATE seeds without a committed prereg' (2026-09-28)." >&2
    exit 1
  fi
  prereg_rel="$PREREG_PATH"
  case "$prereg_rel" in
    "$cd_path"/*) prereg_rel="${prereg_rel#"$cd_path"/}" ;;
    "$cd_path_expanded"/*) prereg_rel="${prereg_rel#"$cd_path_expanded"/}" ;;
  esac
  prereg_rel="${prereg_rel#./}"
  if ! git -C "$cd_path_expanded" cat-file -e "$rev:$prereg_rel" 2>/dev/null; then
    echo "⛔ REFUSED: --prereg '$PREREG_PATH' is not a file committed at $cd_path's HEAD ($rev) -- it must exist in the repo history the job actually runs against, not just on disk." >&2
    exit 1
  fi
fi

exec bash "$GPUQ" add "$CMD"
