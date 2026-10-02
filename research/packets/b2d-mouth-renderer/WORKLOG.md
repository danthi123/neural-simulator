# B2d stub arm WORKLOG

## 2026-10-01 17:15 (Hermes, local model): every-session check (T9)

- Worktree created from `origin/research/b2d-mouth-renderer` @ aa37cdb1f (no local branch existed).
- Section 4 dry-run (`POOL_ROOT=/home/dant123/Projects/neural-simulator`, S override per runbook — the script
  defaults to the workstation path which does not exist here): `43 fresh b2d0929 lines still queued (>= 10):
  nothing queued`, `queued ... well-formed: 43`. lb.json-landed = 2 of 258, dispatched-ever = 0. Wave not due
  -> nothing queued, no ledger change.
- Section 5 GPU check: `gpu_queue.sh status` -> state running, current idle, queued 0; gpu.queue mouth lines = 0;
  landed `b2d0929-mouth/` entries = 5 (of 18), no GPU job running. Per the table this is the "queue drained"
  row -> the mouth arm's (re-)queueing is Claude's (NEEDS-CLAUDE #3/#6); scoring gate (258 stub + mouth) not met.
- GUESS: entries in `b2d0929-mouth/` counted as dirs-or-files without inspection; scoring not attempted either way.
- Note: this branch's `tools/biology_check.py` is also pre-fix (no `_projects_dir`) — if a findings commit is
  ever needed here it may block like [iw2-gate]; this WORKLOG commit is packets-path only.
- 2026-10-01 23:37 EDT (every-session section-4 block, S paths + POOL_ROOT override): dry-run and
  real run both printed `43 fresh b2d0929 lines still queued (>= 10): nothing queued` (rc=0,
  fresh-queued=43, stale=0, dispatched-ever=0, lb.json 2 of 258) - the documented "previous wave is
  still waiting" case; ledger b2d_waves.tsv unchanged, no commit. Section-5 mouth check skipped per
  RULING b2d-mouth-gpu (T23 replaces it).
