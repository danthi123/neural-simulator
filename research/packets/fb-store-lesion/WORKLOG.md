# fb-store-lesion packet WORKLOG

## 2026-10-06 — RULING `iw2-decision` task: prereg run, steps started

- Corpus step (prereg section 3, first): `bash tools/before_you_build.sh "belief store attractor lesion false
  belief persistence"` -> rc=0, logged to .git/corpus_checks_shared.jsonl (top hits: the prereg itself, the iw2
  false-belief review section, the 2026-07-24 W3 register finding; no prior finding cuts the belief attractor).
- Provisioning: first request at the prereg's literal pin 67c909507c22410df5ee09a839dca66c20e9d380 was REFUSED
  (state=refused, "stale: does not contain published main 290e3be3f08f ... main changed more than docs since").
  Re-pinned to the main tip 290e3be3f08f0a0ebe7a364eed3410093ae9f5a0 per the provisioner's hint: it is a
  descendant of 67c909507 (merge-base check) and carries prereg, jobs_template, fb_store_score and the row module
  (git cat-file all OK); the commits between are data/docs only. Provision request -> state=ok at 05:45:10Z on
  pool41, pool42, pool40, wspool (attempt 1/3, rc 0, "ALL PROVISION DONE").
- Line build: dev-seed-7 line (own tag fbstore-290e3be-dev7) and the six gate lines (sed of jobs_template.txt,
  <PIN>=290e3be3f..., <PIN7>=290e3be) both shape-check rc=0 (tools/queue_job_shape_check.sh).
- Dev line QUEUED (pool_queue add --checked 'corpus: ...', depth now 36) at ~01:47 EDT. Gate lines NOT queued yet:
  the prereg order is dev shard first, confirm checklist item 4 (row exercised, lesion arm records store_lesioned),
  then the six gate lines. Next: when _shards/fbstore-290e3be-dev7/s7/tom-false-belief-store/lb.json lands, check
  it, then add gate6.txt, then score once:
  `python tools/fb_store_score.py --tag fbstore-290e3be --pin 290e3be3f08f0a0ebe7a364eed3410093ae9f5a0 --out
  research/findings/raw/_fb_store/verdict.json` and NEEDS-CLAUDE "[fb-store] RESULT".
- One re-run per not-DEFINED cell, logged (prereg section 3); reading table fixed in prereg section 4; prediction
  STORE-NOT-LOAD-BEARING (section 5, an expectation not a bar).

## 2026-10-06 01:54-02:1x wake — dev shard CONFIRMED, gate canary launched

- Dev shard landed (dispatched 01:47:00 to wspool, ~3.5 min): verdict=pass, n_exercised=1, n_not_exercised=0,
  lesion_reproduced=true, flag_resolves=true, deterministic=true, null_control_clean=true, n_null_controls_checked=1,
  and the row note records "Engagement: false_belief_tom.store_lesioned is True on the lesion arm" -> prereg
  checklist items 1-5 all satisfied (item 4 explicitly confirmed).
- Gate lines: pool_queue.sh add needs, in order, (a) a committed prereg path cited as 'prereg <path>' in --checked
  (now: research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md, verified present at pin), and (b) a pool
  canary record for (load_bearing_fraction, 290e3be3f) - gate seeds require a pool-node canary first.
- Canary launched: `pool_canary.sh run auto '<seed-42 line>' --wait-s 1500` (tracked session proc_9dfdafd5235a).
  Canary output is redirected to raw/_pool_canary/, so it does not write gate data; all six gate lines queue after
  a PASS record. Known defect: launch ssh may report rc=3 while the detached job runs (NEEDS pool-canary).
