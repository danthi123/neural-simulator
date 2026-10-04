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

## 2026-10-02 23:33 (local model) — every-session §4 duty

- Wave script refused at first: its `POOL_ROOT` default still pointed at the pre-migration workstation tree
  `/home/dant123/Projects/sim` (gone). Default migrated to `/home/dant123/Projects/neural-simulator` (RULING
  home-to-server); the driver also exports `POOL_ROOT` explicitly.
- Duty result: `fresh-queued=43 stale-queued=0 lb.json-landed=2 of 258` → **nothing queued** (43 ≥ 10 threshold,
  normal — wave 1 still waiting in the deep queue). Ledger unchanged.

## 2026-10-03 11:39 (local model) — every-session §4/§6 duty

- Wave script (S paths, POOL_ROOT override as of 10-02): `fresh-queued=43 stale-queued=0 dispatched-ever=0
  lb.json-landed=2 of 258` → **nothing queued** (43 ≥ 10 threshold — the documented "previous wave is still
  waiting" case; rc=0, ledger b2d_waves.tsv unchanged, no ledger commit).
- §6 counts: stub 2/258 landed; mouth lb.json **18 present but unvetted** (NEEDS-CLAUDE #4: ran at P=5818dc6
  from the workstation revisions export, uncommitted, no s7) → T23 preflight and the registered 18-line GPU
  queue stay blocked on that open line; §5's own check is superseded by RULING b2d-mouth-gpu (T23's HANDOFF
  governs). W gpu.queue holds 0 b2d0929-mouth lines and the card is idle — nothing to yield.
- T9 scoring stays gated: 258 stub cells not landed (2/258) and the mouth arm awaits Claude's #4 ruling.
- 2026-10-04 06:3x (local model) **T0 queue check (runbook 1.2)**: no pool-hold — every key with an unmet Start
  still has lines moving (`_stage1_c1_category/gate` 54q, `_stage2_exit/dev` 8q, `b2d0929/` 43q, `iw1-R-` 4q+8r,
  `iw2-flip-` 25q+8r); no `[pool-hold]` NEEDS.
- 2026-10-04 06:4x (local model) **T23 mouth arm queued (RULING b2d-mouth-gpu)**: all **18/18** registered lines
  from `jobs/mouth_5818dc6.txt`, unchanged, queued from S with
  `cd $WT/b2d-mouth-renderer && bash tools/gpu_remote.sh add '<line>'` (failures=0). Verified:
  `gpu_remote.sh status` → queued: **18**, dispatcher up (pid 2502), state PAUSED, current idle,
  VRAM free 1177MiB (min_free 3000); 0 pre-existing b2d0929-mouth lines before the adds.
  **yield attempted and refused**: `gpu_remote.sh yield --max-hours 12` printed
  `refusing to yield: GAME_MODE is set (owner gaming/away) -- see: bash tools/game.sh status` (not the
  `not loaded` case) → NEEDS "B2d mouth arm: yield refused (...)" per HANDOFF section 5; the lines wait for the
  owner's card, nothing forced. Note (guess marked): `draft-ca3-symmetric-ti` also holds 2 tracked changes
  (`tool_health.json` auto-stamp + `runs.jsonl` append) that section 5 says would also block yield, and
  `runs.jsonl` is never addable by rule — the GAME_MODE line carries the block regardless.
  Next (sections 6-9): per-session counts; steps 7-9 after 258+18 land; A1.4(c) re-runs only as reported.
  HANDOFF NEEDS #6's queueing clause superseded by the ruling; #3 and the HANDOFF untouched (Claude's).
