- 2026-10-06 18:10 (Hermes): steps 0-2 in progress: C5-ON-MAIN, gate branch synced to main -> bc66ccaa3
  (no conflict, GATE_SHA untouched), BOTH-OK, tests 14 passed + 26 passed, provision via provision.req
  (RULING iw2-provision; pool_provision.sh not used) -> ok on pool41/42/pool40/wspool, TSVs emitted
  (8 organ + 34 dev). Canary (s7 inner intact_a flag 1) launched detached; scoring waits on it.
- 2026-10-06 18:21 (Hermes): canary PASS: stage2_exit_gate_bc66ccaa3_wspool_20261006T181338, rc=0,
  wall=430s, json=1, errors=0, tracebacks=0 (auto-recorded in pool_canaries.tsv).
- 2026-10-06 18:55 (Hermes): queued all lines per HANDOFF step 2: 8 organ + 34 dev + 1 regression
  battery (BRAIN_IDLE_REPLAY flag ON vs OFF) at bc66ccaa3, 43 adds 0 fail; dispatcher already
  consuming. Scoring (step 3) on next wake when results land in $P.
- 2026-10-06 19:28 (Hermes): step 3 partial: organ scored PASS 8/8 (n_seeds 8, failing {}) at op 0.
  Reply shards 10/34 landed (battery pending); --score-reply runs when all 34 land.
- 2026-10-06 20:32 (Hermes): step 3 partial: --score-reply verdict PASS n_pass 8; answer_rate all
  8 seeds status ok, flags []; s7_flag_on_vs_off differs ["p_private"], only_idle_turn_differs true;
  all routes idle_replay; 0 error fields across 42 shards. Battery line running on wspool since
  19:59; data commit + INFO wait on it.
- 2026-10-06 21:40 (Hermes): §5 partial. Regression battery KILLED on wspool at its own memcap 10
  after ~85 min, no result JSON. Delegated the verdict to one MiMo Pro delegate (RULING models),
  which returned STOP, no re-queue. Quoted reasoning: "The battery result is missing and the
  one-re-queue clause nominally covers it, but the record proves no re-queue at pool-fitting
  parameters can produce it: the same runner in the same numpy mode is a measured ~27-28 GB job
  (GAP_CLOSURE_MISSION 2026-09-09 RAM-WALL entry; tools/mem_ok.sh's 'the flip battery's real
  footprint' comment), which exceeds every pool node (mini-PCs 15 GB, wspool's 20 GB slice), so
  the proposed raised budget would only repeat the kill later and burn the single permitted
  re-queue, while mem_ok 28 / memcap 30 can never dispatch to any node... the routes that can
  (gpu_queue.sh add, which section 2 reserves, and the authorized r7i AWS lane, which c1-lane
  assigns to Claude) are not mine to take... the no-defer law is satisfied by escalating the
  capability to Claude's routing, never by abandoning it." Data commit (42 files, no battery
  JSON) done; §5 INFO deferred (its regression slot cannot be filled); NEEDS STOP filed.
- 2026-10-06 22:20 C5 battery verdict (delegate, MiMo Pro, quoted): "STOP. §5 fires only on battery
  all_pass true; repo precedent GAP_CLOSURE RAM-WALL 'the numpy flip battery is a measured ~27-28 GB
  job... above every pool node', and this kill at 10 GB is larger than my 42 reply shards' 5.68 GB
  ceiling - no lawful re-queue can land it (40 GB r7i / gpu_queue cupy-solo / bench flag + respec are
  §2-reserved). File [s2-c5-build] STOP; commit the 42 files (no battery JSON); do NOT send INFO; no
  flip." Acting on it: data landed d77dd5b77 (organ 8/8, reply 8/8, peak RSS 5.7 GB); NEEDS
  b7db48857 ([s2-c5-build] STOP + [driver-venv] INFO: dearpygui pruned from the driver venv, package
  installs blocked in single-query).
- 2026-10-06 22:20 gate note: .lane_waiver 6h/24h renewal budget was exhausted, so the
  lane-starvation gate was served the designed escape route: three fresh registrations in
  research/coordination/lane_builds.jsonl (C = s2-c5-build in flight, E = b2d-amend2 step-1 GPU jobs
  running on W, B = t1-ldc-step4a draft build, ordered by RULING T1-LDC, starts next wake). Idle
  lanes A/D = 2 < MAX_IDLE_LANES 3; gate passes without a waiver.
- 2026-10-08 04:50 (Hermes): c5-on-arm-diagnose queued (RULING c5-on-arm-diagnose, after c5-battery-void): the
  ON-arm worker per probe group at rev bc66ccaa37da346779f16b8eaf9b0f1585e78b88 -- `--worker --env
  '{"BRAIN_IDLE_REPLAY":"1"}' --turns <labels>`, mem_ok 9 1 + memcap 10, `/usr/bin/time -v -o <g>_on.time`
  as the ONLY RSS/wall instrumentation (the runner/memcap/dispatcher record neither; `-o` survives an
  OOM kill), numpy, no seed. Groups g1..g6 partition all 26 PROBE_TURNS labels, session chains kept
  (hold,held / dr_a,dr_b,dr_c / bc_a,bc_b in order) -- partition + exact cmds designed by one MiMo Pro
  delegate per RULING models; 6 adds 0 fail; per-group outputs under
  research/findings/raw/_regression_battery/c5_on_diag/. `--flag --subset` was unusable by design:
  run_regression_battery hardcodes arm filenames (collisions) and runs BOTH arms. Verified pool41/42/40
  hold the pinned revision and /usr/bin/time. One bogus line (bash's readonly GROUPS=gids array made the
  loop queue `--turns 100`) withdrawn immediately; withdrawn.log carries the reason. Reading per ruling:
  a group killed at the cap (peak ~= 10 GB or rc 137) is the finding; survivors' g*_on.json get
  committed with the OFF arm JSON + per-group table, then the RESULT naming the turns and the growth.
  Next wake: sync results (tools/pool_sync.sh) when the lines land.
- 2026-10-08 08:55 (Hermes): c5-on-arm-diagnose first attempt: 4 of 6 lines died rc=127 on
  wspool within minutes -- the pool sandbox executes the line's own `/usr/bin/time`, and that
  node does not have it (`/opt/host-status/sim-pool-jobs/exits.json`: g2 08:37, g3 08:38, g4
  08:43, g6 08:47, all rev bc66ccaa3; g2_on.log = "/opt/pool-sandbox/bin/systemd-run: line 26:
  /usr/bin/time: No such file or directory"). The 04:50 entry verified the pin on pool41/42/40
  only; W took 4 of the 6 lines. g1 (pool40, live 140 MB log) and g5 (pool41) continue unwrapped
  of this failure. Requeued g2/g3/g4/g6 -- identical worker at rev bc66ccaa37da346779f16b8eaf9b0f1585e78b88,
  same turns, mem_ok 9 1 + memcap 10, SIM_BACKEND=numpy -- with GNU time dropped so any node runs
  them; those four groups lose the .time peak-RSS/wall stat the 04:50 entry called the only such
  instrumentation (rc-137-at-cap stays detectable via exits.json + missing json; memcap and the
  runner json remain). W should `apt install time` before future W lines use it -- NEEDS
  [c5-on-arm-diagnose] INFO filed; FAILURE_LOG row added.
- 2026-10-08 09:12 (Hermes): c5-on-arm-diagnose attempt 2 failed differently from attempt 1:
  the 08:55 requeue was typed out anew instead of copied, which added a layer of backslashes to
  the `--env` JSON, so the worker parsed the env as a STRING and crashed
  `AttributeError: 'str' object has no attribute 'items'` at onebrain_regression_battery.py:986
  (rc=1 on wspool 08:58 for g2/g3/g4; g6's line consumed without an exit row yet). Pinned to
  the queue-time CMD, not the node: g1/g5 from the ORIGINAL queue still run clean (their [memcap]
  line shows `--env {"BRAIN_IDLE_REPLAY":"1"}` while attempt 2's shows the escaped quoted
  form, and both source scripts differ at that byte range). Attempt 3 queued 09:12, ok=4
  fail=0: the CMD rebuilt PROGRAMMATICALLY from c5_queue2.sh (make_c5_fix.py copies the CMD
  line byte-for-byte and removes only the /usr/bin/time segment), groups g2/g3/g4/g6, same rev
  bc66ccaa37da346779f16b8eaf9b0f1585e78b88/memcap 10/turns. Rule taken: any requeued CMD is
  copied by a transformer from the source script, never retyped (shell quoting does not survive
  transcription). g1 (pool40) and g5 (pool41) workers continue.
