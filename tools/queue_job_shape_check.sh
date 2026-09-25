#!/usr/bin/env bash
# queue_job_shape_check.sh — ENQUEUE-TIME "would this even start?" check, shared by every queue producer
# (tools/pool_queue.sh add, tools/gpu_queue.sh add, tools/queue_add.sh). Sourced, like
# tools/pool_revision_marker.sh, so there is exactly one place this predicate is spelled out.
#
# WHY (2026-09-25). Six SETTLE A2 pool lines were queued as
#   'A2 wiring seed 42: mem_gb=8 && cd ~/derisk-pool/revisions/... && env ... .venv/bin/python -u -m
#    research.runners._affect_marker_settle_congruence --run-wiring --seeds 42 ...  #checked:...'
# -- a prose label ("A2 wiring seed 42:") glued onto the front of an otherwise-correct job -- so on the node
# bash ran `A2` as a command, got "command not found" (rc=127), and nothing ran, while the board and the
# queue's own claim record said the jobs were dispatched (research/queue/pool.queue.claims:1923-1925,1970-1972
# -- six lines, seeds 42/43/44/100/101/102; research/FAILURE_LOG.md's 2026-09-25 SETTLE A2 row).
#
# This is not hypothetical or a one-off: research/queue/gpu_queue.log already carries a REAL, historical
# instance of the identical failure class (2026-08-31/09-01, three separate cycles, lines ~672190/711341/
# 760341) -- a queued job that was the single bare word `status` (almost certainly `gpu_queue.sh add status`,
# meant to run `gpu_queue.sh status`), which ran as `bash: line 1: status: command not found`, rc=127, in
# under a second each time, and was never flagged as anything other than an ordinary DONE(rc=127) line.
#
# NEITHER of pool_queue.sh's existing checks (COMMAND VALIDITY GATE / REMOTE VALIDITY) catches this class: both
# key off finding `-m research.runners.X` ANYWHERE in the line and validate THAT module's --help -- a line
# whose first word is prose/garbage but that happens to mention a real module further down the very same line
# (exactly the A2 shape) sails through every one of them, because none of them ever asks "would bash even
# attempt to run this line, or does it die on word 1?".
#
# WHAT THIS CHECKS (and nothing more -- it is a SHAPE check, not a correctness check):
#   1. Does the line PARSE as shell (`bash -n`)? Catches a syntax-broken line (unbalanced quotes, a dangling
#      `&&`, ...).
#   2. Does the line's FIRST simple command's first word -- after skipping any leading `VAR=val` assignments
#      -- name something bash could actually invoke: a shell keyword/builtin (cd, :, env(*), until, for, if,
#      [, true, ...), a function, or a PATH-shaped token (anything containing a "/", e.g. `.venv/bin/python`
#      or an absolute interpreter path)? If the word resolves to none of those, the line would die on argv[0]
#      with "command not found" (rc=127) the instant a shell tried to run it -- REFUSE it before it is queued.
#      (* `env` itself resolves via PATH like any external command; it is listed only as the canonical
#      example of "a path/command that resolves".)
#
#   A PATH-shaped first word (contains "/") is accepted WITHOUT checking it actually exists on disk: whether
#   `.venv/bin/python` exists depends entirely on WHERE the job eventually runs (the local checkout for a GPU
#   job, `~/derisk-pool/sim` or an isolated `~/derisk-pool/revisions/<sha>` for a pool job, none of which is
#   necessarily this process's own cwd -- a worktree, in particular, has no `.venv` of its own at all) --
#   that is exactly what pool_queue.sh's own REMOTE VALIDITY check (and gpu_queue.sh's local argparse check)
#   already verify, each in the one environment where the answer is actually meaningful. This check only
#   rules out the "not even a real command shape" case those checks never look for.
#
# Uses bash's OWN parser to find the first simple command, via a DEBUG trap that fires with the exact source
# text of the command about to run and exits IMMEDIATELY -- before that command, or anything after it, is
# ever executed (verified: a job of `touch canary && ...` never creates the canary file). This is deliberate:
# the check must never actually run any part of an untrusted-looking job, even a destructive one.
#
# Deliberately does NOT `set -e`/`-u`/`-o pipefail` at this top level: this file is SOURCED by callers with
# different shell-option regimes (pool_queue.sh already runs under `-uo pipefail`; gpu_queue.sh runs under
# plain `-e`), and a source-time `set` change here would alter the CALLER's shell options for the rest of its
# own run, not just this file. The function below is written defensively (every expansion has a `${VAR:-}` or
# is a `local`) so it behaves the same regardless of the caller's options; the direct-invocation CLI path at
# the bottom sets its own options explicitly.

# queue_job_runnable_check <job-line>
#   OK:      prints nothing, returns 0
#   REFUSE:  prints one "⛔ REFUSED: ..." message to stdout, returns 1
queue_job_runnable_check() {
  local job="${1:-}"
  if [ -z "$(printf '%s' "$job" | tr -d '[:space:]')" ]; then
    printf '⛔ REFUSED: empty job line.\n'
    return 1
  fi

  # 1. SYNTAX.
  local syn
  if ! syn=$(bash -n -c "$job" 2>&1); then
    printf '⛔ REFUSED: not valid shell syntax (bash -n): %s\n' "${syn:-parse error}"
    return 1
  fi

  # 2. FIRST SIMPLE COMMAND, via bash's own parser. The job text is appended VERBATIM to a tiny script (never
  #    spliced into a quoted string) so no quoting in the job itself -- single quotes, double quotes, embedded
  #    newlines, anything -- can break out of or corrupt this check.
  local tmp first_cmd trap_rc
  tmp=$(mktemp "${TMPDIR:-/tmp}/queue_shape_check.XXXXXX") || return 0   # cannot even get a tmp file -> do not block queueing on an infra failure
  {
    printf '%s\n' 'trap '"'"'printf "%s" "$BASH_COMMAND"; exit 71'"'"' DEBUG'
    printf '%s\n' "$job"
  } > "$tmp"
  first_cmd=$(timeout 5 bash "$tmp" </dev/null 2>/dev/null)
  trap_rc=$?
  rm -f "$tmp"
  if [ "$trap_rc" -ne 71 ] || [ -z "$first_cmd" ]; then
    # The job ran to completion (or failed) without the trap ever seeing a traceable simple command -- e.g. a
    # bare comment/blank body, or a shape this technique cannot see into. Never refuse on a shape we cannot
    # positively identify as broken; the syntax check above already covers the case that matters here.
    return 0
  fi

  # 3. Strip leading `VAR=val` assignments (any number: `A=1 B=2 cmd ...` is valid shell).
  local -a words
  read -r -a words <<<"$first_cmd"
  local i=0 n=${#words[@]}
  while [ "$i" -lt "$n" ] && [[ "${words[$i]}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; do
    i=$((i + 1))
  done
  if [ "$i" -ge "$n" ]; then
    return 0   # pure assignment(s), nothing else to invoke -- valid shell
  fi
  local word="${words[$i]}"

  # 4. Would bash know how to run this word?
  case "$word" in
    */*) return 0 ;;   # path-shaped -- existence is environment-specific; see the header note above
  esac
  if type -t "$word" >/dev/null 2>&1; then
    return 0
  fi
  printf '⛔ REFUSED: first word %s of the job'"'"'s first command is not a shell keyword/builtin/function and does not resolve on PATH -- it would die instantly with "command not found" (rc=127), exactly like the 2026-09-25 SETTLE A2 lines and the historical `status` job in gpu_queue.log. Full first command: %s\n' \
    "'$word'" "$first_cmd"
  return 1
}

if [ "${BASH_SOURCE[0]:-}" = "${0}" ]; then
  set -uo pipefail
  [ "$#" -eq 1 ] || { echo "usage: $0 '<job line>'" >&2; exit 2; }
  if out=$(queue_job_runnable_check "$1"); then
    exit 0
  else
    printf '%s\n' "$out" >&2
    exit 1
  fi
fi
