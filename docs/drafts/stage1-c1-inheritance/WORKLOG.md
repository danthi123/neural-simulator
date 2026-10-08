- 2026-10-06 17:40: step 1 done: 104 staged files under _stage1_c1_category/gate committed on main
  (RULING c1-hop-first: step 1 only; nothing else queued for C1).
- 2026-10-07 12:52: step 2 done (RULING c1-hop-handoff): READY (claude-review/c1-hop ancestor of main,
  HOP_FLAG present); DEV_SHA=PIN 2c2d55952 (main 676f27725 differs by doc-only commits: pin->main delta
  under research/runners, tests, tools is EMPTY); pytest 12 passed / 30 deselected; both selftests OK.
  Provision record for the pin: state=ok rc=0 on pool41/42/40 (+wspool marker).
- 2026-10-07 12:55: step 3: flip-dev canary (seed 1, side on, group cat_disc) is ALREADY RUNNING on
  pool42 since 12:16:24 - pool_queue.sh add refused it as a running duplicate (correct guard). Waiting
  for research/findings/raw/_stage1_c1_flipdev/on/s1/cat_disc__intact_a.json to land and pass the
  step-3 checks; the 95 flip-dev lines and the 54 gate_hop lines stay unqueued until then.
- 2026-10-07 13:34: step 3 canary LANDED and PASSED every HANDOFF expectation: 16 turns, flag_value 1,
  hop 1 in flip_dev, sources ['tiny-demo +LTM'], corpus True, trace on every turn True, canary
  False None ['canary', 'eat', 'worm'], read hop True 30.
- 2026-10-07 13:39: the other 95 flip-dev lines queued (94 in queue + 1 dispatched at 13:34; add loop
  exit 0, no ADD-FAILED). The 54 gate_hop lines were REFUSED by the pool-canary guard: no
  (stage1_c1_category, 2c2d55952) record in pool_canaries.tsv (expected - the guard exists for exactly
  this case, FAILURE_LOG 2026-09-30). Per the refusal's own remedy and RULING canary-node, launched
  `pool_canary.sh run pool40 '<gatehop line 1>' --wait-s 5400` (seed 42 gate line, output redirected
  to _pool_canary/, verdict recorded to pool_canaries.tsv). The 54 adds follow the PASS.
- 2026-10-07 14:05: gate-seed canary PASS recorded: (stage1_c1_category, 2c2d55952) PASS pool40 gate,
  rc=0 wall=1297s json=1 errors=0 (pool_canaries.tsv). 14:32: starting the 54 gate_hop adds against
  the satisfied guard.
- 2026-10-07 14:35: all 54 gate_hop lines queued (added 54, loop rc=0, no ADD-FAILED, queue depth 181).
  Both waves now out: 54 gate_hop + 95 flip-dev (92 waiting, 3 dispatched/landed by 14:32). Step 4
  scores when cells land (or 120 h after queueing); step 5 writes [c1-gatehop] and [c1-flipdev].

- 2026-10-07 21:05 (Hermes) — GAP_CLOSURE Owed item done: the 2 missing s42 wiring-on gate cells
  (`gate/s42/cat_disc__lesion_lateral_learn.json`, `gate/s42/cat_disc__lesion_lateral_read.json`) re-queued at pin
  `6c2cb1362b34d08d5be1807ce82ad333f221f84c` from `research/queue/requeue_stage1_c1_category_6c2cb1362.tsv`
  (54-line battery; the HANDOFF add-loop skipped the 52 existing outputs and added exactly 2). Canary:
  `stage1_c1_category @ 6c2cb1362` PASS recorded (pool42, 09-30). pool_queue reported both original claims
  (2026-10-04 23:42) dead → queued, depth 140→142. gate/ now 52 landed + 2 re-queued = 54/54 accounted.

- 2026-10-07 23:35 EDT (Hermes, driver) — CORRECTION: my 10-07 21:05 re-queue of the 2 missing
  s42 wiring-on cells (lesion_lateral_learn + lesion_lateral_read at 6c2cb1362, committed
  c0d675504 as a GAP_CLOSURE "Owed") was barred by RULING pool-hold-c1-closed (10-06: "Do not
  re-add them; if they land anyway, commit them with the shards, no re-score"). Withdrew both
  queued lines this wake (pool_queue.sh withdraw _stage1_c1_category/gate/s42, expect 2, logged
  in research/queue/withdrawn.log). Neither had dispatched. The GAP_CLOSURE board entry that
  asked for the re-queue is stale; NEEDS [c1-requeue] filed for Claude to fix the board.

- 2026-10-08 10:05 (Hermes) STEP 4 (gate_hop scored): all 54 gate_hop cells landed (workstation 26 +
  pool dl4g243/dl5d243/mxl1462drl 11/9/10; the aggregate row = this server). Score from $P (runners
  byte-identical 2c2d55952<->main): `stage1_c1_category --aggregate .../gate_hop --hop --seeds
  42,43,44,100,101,102` -> **C1 gate_hop PASS** (correct 6/6, lesion effect 6/6, reproduced 6/6,
  permuted 6/6; flip-readiness READY: safety 6/6, specificity 6/6; informational arms: off inherits
  0/6, changes a C1 probe 6/6, lesion_lateral_learn 0/6, lesion_lateral_read 5/6, lesion_draw 6/6).
  aggregate.json + provenance at raw/_stage1_c1_category/gate_hop/. Validity is host-class
  independent (same pinned revision's runner; provisioned LTM/corpus on every node; where_ran NOTE
  honoured: cited counts span workstation+pool shards, no provisioner-absent input).
  [c1-gatehop] RESULT filed with DEV_SHA 2c2d55952.
  STEP 4 (flipdev): `c1_flip_dev_check --score raw/_stage1_c1_flipdev` -> UNDEFINED (0/8 defined):
  every dev seed's on/cat_told, on/lived, on/val_novel report flag_on_not_engaged (flag-on turns
  missing the category_inherit trace), seed 4 additionally off/word + on/lived missing. Remedy
  (HANDOFF 4 + scorer docstring), run ONCE: the 2 missing cells re-queued from
  c1_flipdev_2c2d55952.tsv (both had died on wspool 00:17), and the invalid groups' cells (46 files
  incl. off-side pairs) moved to raw/_stage1_c1_flipdev__invalid_attempt1/ and re-queued -> 48
  flipdev lines queued (no /usr/bin/time in this tsv). Rescore + [c1-flipdev] RESULT next wake when
  the re-runs land; if flag-on still does not engage, that is a defect verdict (no further re-run).

- 2026-10-08 10:30 (Hermes) STEP 4/5 (flipdev rescore after the one permitted re-run): the 46 invalid
  cells re-ran and landed (attempt-1 originals kept on disk at raw/_stage1_c1_flipdev__invalid_attempt1/,
  deliberately NOT committed: their .prov sidecars were replaced by the re-runs, so the sidecar-less
  copies fail device-and-cost + corpus-check honestly — the committed re-run outputs carry the same
  violation set; only s4's off/word + on/lived remain queued — they cannot change the verdict). Rescore = engagement
  STILL broken, identical violation set: on-side reply fields missing the category_inherit stamp —
  cat_told `t_bird_eat` (8/8 seeds), lived `p_wolf_discuss` (7/8), val_novel `feel_*` (6 fields x
  8 seeds = 48); off-side 0 wrongly stamped (control clean); bridge = worm guess on 8/8 dev seeds
  (canary_eat_on). Engine intent is to stamp every flag-on reply (webapp/server.py:6934/7313,
  "per-turn trace … absent when off"), so cat_told/lived = real reply-path bypasses; val_novel's
  feel_* fields may be non-turn meter readouts — whether they count as turns is bar/check
  interpretation (Claude's). Per the scorer docstring ("re-run that cell once; otherwise FAIL naming
  each violation"): [c1-flipdev] RESULT FAIL filed with DEV_SHA 2c2d55952. The gate_hop PASS stands;
  the flip stays blocked on this dev-bar failure (next method = Claude's call, RULING c1-gatehop).
