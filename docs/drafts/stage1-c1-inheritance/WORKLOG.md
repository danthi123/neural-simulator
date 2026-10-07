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
