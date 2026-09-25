#!/usr/bin/env bash
# corpus_check_lib.sh — ONE resolution of "where is THE corpus-check log", sourced by every producer/consumer
# of it: tools/before_you_build.sh (writer), tools/pool_queue.sh + tools/gpu_queue.sh `add` (readers, to carry
# the freshest check into a queued job's env). research/runners/__init__.py's _corpus_check_state() mirrors
# this same resolution in Python (no cross-language sourcing), by design kept in lockstep with this file.
#
# INCIDENT 2026-09-25 (gate corpus-check-required blocked research/score-gap4-c26-0925 @ ed6758f61 despite a
# real corpus check having been run at 01:38:31). ROOT CAUSE: before_you_build.sh wrote its log under
# $PWD/research/queue/.corpus_checks.jsonl, so EVERY git worktree kept its own copy (115 existed under
# .claude/worktrees/*/research/queue/) while the runner's stamp read a DIFFERENT one (whichever checkout's
# research/runners/__init__.py happened to be imported, in practice the main root when a pool/GPU dispatcher
# launched the job from there) — a worktree's own fresh check never reached the stamp its own run produced.
#
# FIX: resolve ONE shared log per repo — the parent of git's *common* dir, which every worktree of one repo
# shares (a worktree's `.git` is a file pointing at the main checkout's `.git/worktrees/<name>`; the common
# dir is the one thing they all agree on) — instead of each worktree's own $PWD. This is the exact same
# resolution tools/gpu_queue.sh (SHARED_ROOT) and tools/pool_queue.sh already use for their own singleton
# queue/daemon state, applied here to the corpus-check log for the same reason.
#
# LIMITATION, stated once here rather than re-derived at each call site: a pool node or an isolated
# `derisk-pool/revisions/<sha>` checkout is an rsync'd copy with NO `.git` at all, so `git rev-parse` fails
# there and this resolves to the FALLBACK root's own (nonexistent, until created) log — there is no shared log
# to find on a machine with no git checkout. That is precisely why (3) below carries the check via the job's
# OWN environment (CORPUS_CHECK_WHEN/QUERY) instead of relying solely on a log file being reachable at
# execution time.

# corpus_check_shared_log <fallback-root>
#   Echoes the ONE shared corpus-check log path for the repo containing <fallback-root> (or the caller's cwd
#   if git resolves from there instead — `git rev-parse` follows cwd, not the fallback argument). Outside a
#   git checkout (no .git anywhere, e.g. an rsync'd pool-node copy), falls back to
#   <fallback-root>/research/queue/.corpus_checks.jsonl. SIM_CORPUS_CHECK_LOG overrides both (tests only).
corpus_check_shared_log() {
  local fallback_root="${1:-$PWD}"
  if [ -n "${SIM_CORPUS_CHECK_LOG:-}" ]; then
    printf '%s' "$SIM_CORPUS_CHECK_LOG"
    return 0
  fi
  local common
  common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
  if [ -n "$common" ]; then
    printf '%s/research/queue/.corpus_checks.jsonl' "$(dirname "$common")"
  else
    printf '%s/research/queue/.corpus_checks.jsonl' "$fallback_root"
  fi
}

# corpus_check_latest <log-path>
#   Prints "<when>\t<query>" for the LAST well-formed JSON line in <log-path> (query has tabs/newlines
#   flattened to spaces so the tab-delimited output stays one line), or nothing at all when the log is
#   absent, empty, or has no parseable line. stdlib-only (python3, already a hard dependency of this repo's
#   tooling) so this never depends on a JSON-in-bash parser.
corpus_check_latest() {
  local log="$1"
  [ -f "$log" ] || return 0
  python3 - "$log" <<'PYEOF'
import json
import sys

path = sys.argv[1]
last = None
try:
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and obj.get("when") is not None:
                last = obj
except OSError:
    last = None

if last is not None:
    when = last.get("when", "")
    query = str(last.get("query", "")).replace("\t", " ").replace("\n", " ").replace("\r", " ")
    print("%s\t%s" % (when, query))
PYEOF
}
