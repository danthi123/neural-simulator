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
- 2026-10-04 07:1x (local model) **T9 every-session duty + T0**: queue check clean (no [pool-hold]: c1 54q,
  stage2-dev 8q, b2d 43q, iw1-R 4q+8r, iw2-flip 25q+4r all with lines). `b2d_queue_next_wave.sh --status`:
  fresh-queued=43 stale-0 dispatched-ever=0, all 43 well-formed, threshold 10 -> **nothing queued** (previous wave
  still waiting; no unqueued line action; ledger unchanged so no wave-ledger commit). Counts: stub lb 2/258
  (HANDOFF glob), mouth lb **18/18 landed** (from the 10-01 export, per NEEDS 04:56). §5 check: 18 mouth lines
  queued from S (06:4x); `gpu_remote.sh yield` retried at 07:1x -> still
  `refusing to yield: GAME_MODE is set (owner gaming/away)` **on the workstation** (S's game.sh reads off; the two
  hosts differ, guess marked) -> the 06:45 NEEDS line stands unchanged, no duplicate written. Next sessions:
  retry yield; wave when fresh-queued drops below 10; steps 7-9 after 258+18.
- 2026-10-04 08:1x (local model) **T0 + T9 session duty**: T0 clean (no pool-hold: c1 54q, stage2 8q, b2d 43q,
  iw1-R 4q+8r, iw2-flip 18q+8r). Wave status: fresh-queued=43 stale-0 -> **nothing queued** (>=10 rule); stub
  lb 2/258, mouth 18/18 (10-01 export). **Mouth queue 18 -> 17: first dispatch b2d0929-mouth/s42/affect-coloring
  START 08:04:42 -> DONE(rc=143) at 08:04:51 (killed ~9 s; GAME_MODE pause-term suspected - GUESS).** Yield
  retried 08:1x: still `refusing to yield: GAME_MODE is set` workstation-side. Remote queue: PAUSED, current idle,
  queued 17, VRAM free 3056 (>= min_free 3000), dispatcher up. Per HANDOFF section 5 ("If a run fails, take
  NEEDS-CLAUDE 'B2d mouth arm failed (<its message>)'") wrote the NEEDS line; did NOT auto-requeue the killed
  line (17 remain; Claude's call on retry order). Note: status.sh reads `wspool: unreachable` (T0 r-counts may
  under-read that node; pool40/41/42 reachable) - guess marked.
- 2026-10-04 09:1x (local model) **T0 + T9 session duty**: T0 clean (no pool-hold; iw2-flip 16q+4r, c1 54q,
  b2d 43q, iw1-R 4q+8r, stage2 8q). Wave status fresh-queued=43 stale-0 -> nothing queued (>=10); stub lb 2/258,
  mouth 18/18 (10-01 export). Yield retried: still `refusing to yield: GAME_MODE is set` workstation-side (4th
  refusal). Remote queue unchanged since 08:04's rc=143 kill: PAUSED, idle, queued 17, VRAM free 3056; no new
  mouth dispatch. Open NEEDS unchanged (yield + arm-failed lines stand for Claude). Pool depth 125, pool40
  fail24h=3 (pre-existing).

## 2026-10-04 15:00 — RULING game-mode-pauses applied: re-queued the one rc=143 line (once); arm ALIVE

New rulings read (RUNBOOK 618-625): game-mode-pauses (rc=143 during owner game/test = expected pause, not a
failure; re-queue each such job ONCE when W accepts a yield again; "Act now: GAME_MODE is off") +
commit-attribution (never a Co-Authored-By Claude trailer; self-identify).
Evidence gathered before acting: game.sh status -> GAME_MODE off, GPU_PAUSE off; gpu_remote status ->
state running, queued 13, VRAM free 20347MiB; gpu_queue.log tail shows a b2d0929-mouth cell RUNNING NOW
(s43 affect-appraisal-interoceptive, revision 5818dc6) -> the 17 queued lines dispatch on their own again;
today's load_bearing result arrivals (8+24) are the 4 queue departures completing as lb shards, not more
kills. Only ONE rc=143 kill is documented (b2d0929-mouth/s42/affect-coloring, 08:04, DONE(rc=143)).
Action: re-queued THAT line exactly once via gpu_remote add -> ADD_RC=0, depth 13->14. Notes: (a) gpu_queue
add has NO dedupe (blind append), so a blanket re-add of all 18 would duplicate the 13 still queued — skipped
per the "each job that ended rc=143 ... once" wording; (b) gpu_remote add queues at the worktree HEAD
(c8b48244d, worklog-only commits since the pinned 5818dc6 -> identical code, assert_flipped_defaults.py in
the line still gates); (c) no yield attempted this round — the arm is healthy and dispatching, and yield
would pause it; the ruling only directs re-queue. gpu_queue.log holds no per-line dispatch/rc history in a
6000-line window (stdout dominates), so outcome accounting is via arrivals (results) + queued depth.

## 2026-10-05 23:30 session duty (section 4 every-later-session line + section 5)

- (d) wave 2 queued: `dry run: 43 line(s) would be queued as wave 2`, then `rc=0`
  and `queued b2d0929 lines well-formed: 41` — the count is 41 (not 43) only
  because s43/one-brain-substrate and s43/comprehension-monitor were dispatched
  to pool41/pool42 at 23:23 within the same minute; all 43 lines are accounted
  for (41 fresh in pool.queue + 2 dispatched). Wave state at start:
  fresh-queued=0, dispatched-ever=43, lb.json-landed=43 of 258.
  First run of `b2d_queue_next_wave.sh` hit a 180 s turn limit mid-wave (21
  lines in); the resume path completed the remaining 22. Ledger commit
  `1d3d25432` (no push, wake rule: sim-sync publishes main).
- (b) mouth check: 0 `b2d0929-mouth` lines in research/queue/gpu.queue and the
  local model is down (game.sh status) -> table row `arm waits for Claude`
  (NEEDS #6); nothing queued, no yield attempted.

- 2026-10-06 01:2x session duty: `b2d_queue_next_wave.sh` ran (dry + live) -> 29 fresh b2d0929 lines still
  queued (>= 10): nothing queued (wave 2 draining, 12 lines ran since 23:11); ledger unchanged (no new wave);
  stub-cell arrivals tracked by wakes (18 + 28 new load_bearing files 00:13/01:14). Mouth-arm section-5 table not
  evaluable from this host: the block's env.sh paths are S-side and `gpu_remote.sh log 400` hangs over ssh here
  (guarded 15 s call returned empty twice); local gpu.queue mouth lines = 0, local gpu_queue idle. Per the handled
  [T23] row the S-side paused arm resumes per RULING game-mode-pauses; not touched from here.

## 2026-10-06 09:5x EDT - T23 step 4 scored, step 6 landed (driver: Hermes)

- Step 0 preconditions: uncommitted code none, prereg-ok, job file 18/18 pin refs, corpus hash matches
  CORPUS_SHA. The "mouth lb.json > 0" STOP row fired (18 present) but its parenthetical ("nothing was ever
  queued") is stale: this driver queued the registered 18/18 on 2026-10-04 per RULING b2d-mouth-gpu (runbook
  06:45 line); the arm has since completed on W. JUDGMENT (marked): treated the row as consumed at that
  queueing, and took step 4's own first trigger ("18 lb.json are in $P") as the governing state.
- Step 4 score (`tools/b2d_mouth_score.py`, run per HANDOFF; `| head -60` caused a BrokenPipeError on the
  scorer's final print - harmless, the JSON was already written; the printed `rc=0` is head's):
  verdict **COMPLETE**, preconditions empty, rerun_cells empty -> step 6.
  18 cells: 17 MOUTH-SILENT, 1 DEFINED (s*/affect-drives-response); totals n_defined 1,
  n_reply_load_bearing 0, n_reply_exercised 1, n_mouth_silent 17, n_fail 0; qwen_calls total 4.
  per_row: affect-coloring MOUTH-SILENT, affect-appraisal-interoceptive MOUTH-SILENT,
  affect-drives-response MIXED. Matches A1.3's registered pre-run prediction (MOUTH-SILENT on most cells).
- Step 6 substitutions (rsync absent on S): `cp -a` byte copy; git add done per-existing-pathspec (the
  never-rerun `b2d_mouth_reruns.tsv` and `_b2d0929_attempt1/` do not exist, and a missing pathspec would
  abort the whole add). Data + b2d_mouth_score.json committed on the packet branch with the Agent trailer.

## 2026-10-07 07:41 EDT — wave 4 (every-session line)

- `b2d_queue_next_wave.sh --dry-run`: 43 line(s) would be queued as wave 4; queued for real: 43 attempted, 43 well-formed, rc=0 (lines 130-172 of 258; fresh-queued was 0 < threshold 10).
- lb.json-landed 129/258. Ledger b2d_waves.tsv updated; push skipped (RULING a2-no-push).

## 2026-10-08 07:26 EDT — wave 5 (every-session line)

- `b2d_queue_next_wave.sh --dry-run`: 43 line(s) would be queued as wave 5; queued for real: 43 attempted, 43 well-formed, rc=0 (lines 173-215 of 258; fresh-queued was 0 < threshold 10).
- lb.json-landed 172/258. Ledger b2d_waves.tsv updated; push skipped (RULING a2-no-push).
- Section-5 every-session check: gpu.queue holds 0 mouth lines (dispatcher down, card idle); the mouth arm already
  completed on W and its step-4 score is COMPLETE (17 MOUTH-SILENT, 1 DEFINED; step 6 committed 10-07), so no GPU
  action is owed — the arm waits only on NEEDS-CLAUDE #6 (registered arm schedule).

## 2026-10-08 12:44 EDT — every-session line

- Stub arm: lb.json-landed 184/258 (+12 since 07:26). fresh-queued=29 stale=0 >= threshold 10 -> wave script
  correctly queued nothing (wave 6 comes when < 10 fresh remain).
- Mouth arm: all 18/18 lb.json present on the shared tree; step-4 score unchanged (COMPLETE, 17 MOUTH-SILENT +
  1 DEFINED, committed 10-07); still awaiting NEEDS-CLAUDE #6. gpu.queue: 0 lines, card idle, dispatcher down
  (start is denied to the local model; nothing registered waits on it).
- No ledger change this session (no wave queued).

## 2026-10-09 13:50 — §6 while-running status (Hermes, driver session)

- stub arm: **231/258** lb.json landed (27 outstanding, all queued); mouth arm **18/18**.
- `b2d_queue_next_wave.sh --status` from $W: fresh-queued=26 stale-queued=0 — at/above the
  threshold 10 and waves 1-6 cover all 258 registered lines, so **no wave queued**.
- ledger note: waves 4-6 (commits 898a03fa1, bc8aa27db, b922d6142) live on this branch,
  committed from this worktree after the 2026-10-06 merge; main's b2d_waves.tsv still shows
  the 130-row pre-wave-4 snapshot and heals on the next Claude merge. No tooling runs from
  main's copy (step 6/7 read $W), so nothing is blocked.

## 2026-10-09 14:47 — §6 every-session status (Hermes, driver session)

- stub arm: **232/258** lb.json (26 outstanding, all queued, stale=0); mouth 18/18.
- fresh-queued=25 >= threshold 10 and waves 1-6 cover the full set -> **no wave queued,
  no ledger change**. Progress steady (~1 cell/10 min); no ended-badly wake since 11:35.
