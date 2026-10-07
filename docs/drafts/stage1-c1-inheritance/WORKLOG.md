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
