#!/usr/bin/env bash
# lb_shard_guarded_jobs.sh -- print GUARDED lb_shard.py job lines and refuse to emit any line carrying an
# UNDECLARED BRAIN_* override (research/lbf-row-registry-hook, 2026-09-24, plan step S08 item 5 / lane AG-REG).
#
# WHY: a production-default battery (e.g. B2a, `--no-fixes`) is only honest if it measures the checked-out
# revision's DEFAULT -- tools/assert_flipped_defaults.py already enforces that AT RUN TIME (it fails a job whose
# live environment carries any BRAIN_* flag outside its own FLIPPED dict). This script adds the STATIC counterpart
# the plan step asks for: a grep over the JOBS.TXT TEXT ITSELF, before a single job is ever dispatched, so a bad
# `--extra-env BRAIN_X=1` on the `jobs` invocation is caught here rather than surfacing job-by-job across the pool.
#
#   tools/lb_shard_guarded_jobs.sh --tag b2a0924 --no-fixes --probe-set adequate \
#       --seeds 42 43 44 100 101 102 --root ~/derisk-pool/revisions/<M1-SHA>
#
# Every non-flag argument is passed straight through to `tools/lb_shard.py jobs`. Output: one guarded job line per
# faculty x seed, each prefixed with `.venv/bin/python tools/assert_flipped_defaults.py &&` (the existing runtime
# guard), written to stdout AND to JOBS.txt in the current directory (matching the flipdefaults-* precedent,
# 97d900c30). Exits nonzero, printing NOTHING to JOBS.txt, if any line carries a BRAIN_*= token that is neither one
# of assert_flipped_defaults.py's own FLIPPED keys nor explicitly allow-listed via --allow-brain-flag KEY (repeatable).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

ALLOW=()
PASSTHRU=()
NO_FIXES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-brain-flag) ALLOW+=("$2"); shift 2 ;;
    --no-fixes) NO_FIXES=1; PASSTHRU+=("$1"); shift ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done
# The undeclared-BRAIN_* check only makes sense for a --no-fixes (production-default) job: assert_flipped_defaults.py
# ITSELF only guards a job that passes no fix flag (its own docstring: "Those batteries pass NO fix flag"). A plain
# `jobs` call without --no-fixes legitimately carries FIX_ENV's BRAIN_* flags (including BRAIN_PMEM_OP_STABILIZER,
# which is NOT in the guard's FLIPPED set -- research/findings/2026-09-23-operating-point-stabilizer-go-6seed.md's
# 2026-09-24 caveat: it is a per-seed table, never a flip candidate) -- prefixing/checking those would be a false
# positive, not a safety net.
if [ "$NO_FIXES" -eq 0 ]; then
  echo "[lb_shard_guarded_jobs] --no-fixes not given: this is not a production-default job -- emitting UNGUARDED, unchecked lines (pass --no-fixes for the B2a-style guarded/checked path)" >&2
  exec "$ROOT/.venv/bin/python" "$ROOT/tools/lb_shard.py" jobs "${PASSTHRU[@]}"
fi

# The guard's own declared set (research.runners not required -- read straight from source, same discipline
# assert_flipped_defaults.py itself uses to avoid importing cupy-dependent modules on a CPU-only node).
export ROOT
GUARD_KEYS=$("$ROOT/.venv/bin/python" - <<'PY'
import ast, os
path = os.path.join(os.environ["ROOT"], "tools", "assert_flipped_defaults.py")
tree = ast.parse(open(path).read(), filename=path)
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FLIPPED" for t in node.targets):
        d = ast.literal_eval(node.value)
        print("\n".join(sorted(d)))
        break
PY
) || GUARD_KEYS=""

RAW=$("$ROOT/.venv/bin/python" "$ROOT/tools/lb_shard.py" jobs "${PASSTHRU[@]}")

# Every BRAIN_<NAME>= token across all lines, deduplicated.
FOUND=$(printf '%s\n' "$RAW" | grep -oE 'BRAIN_[A-Z0-9_]+=' | sed 's/=$//' | sort -u || true)
UNDECLARED=""
for k in $FOUND; do
  declared=0
  for g in $GUARD_KEYS; do [ "$k" = "$g" ] && declared=1 && break; done
  for a in "${ALLOW[@]:-}"; do [ "$k" = "$a" ] && declared=1 && break; done
  [ "$declared" -eq 0 ] && UNDECLARED="$UNDECLARED $k"
done
if [ -n "$UNDECLARED" ]; then
  echo "[lb_shard_guarded_jobs] REFUSING: undeclared BRAIN_* flag(s) in the job text:$UNDECLARED" >&2
  echo "[lb_shard_guarded_jobs]   guard-declared: $(echo "$GUARD_KEYS" | tr '\n' ' ')" >&2
  echo "[lb_shard_guarded_jobs]   pass --allow-brain-flag <NAME> if this is a genuine prereg-declared addition" >&2
  exit 1
fi

printf '%s\n' "$RAW" | sed 's#^#.venv/bin/python tools/assert_flipped_defaults.py \&\& #' | tee JOBS.txt
