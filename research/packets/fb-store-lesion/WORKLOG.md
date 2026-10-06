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
- Gate canary: launch over ssh reported rc=3 (known NEEDS pool-canary false failure) but the detached job ran on
  wspool: id load_bearing_fraction_290e3be3f_wspool_20261006T020848, status rc=0 wall=529s, 0 Tracebacks,
  lb.json verdict=pass lb=0.0 exercised=1 deterministic=true store_lesioned=True error=None (seed 42 gate path).
  Recorded PASS (module, 290e3be3f, gate) in research/queue/pool_canaries.tsv -> pool_canary.sh has = rc 0.
- Six gate lines QUEUED 6/6 (pool_queue add --checked 'prereg research/findings/2026-10-05-fb-store-lesion-
  PREREGISTRATION.md at 290e3be3f ...'); within minutes 5 sat in pool.queue and 1 had dispatched. Next wake: when
  all six fbstore-290e3be/s{42,43,44,100,101,102} lb.json land, score once per prereg section 3:
  `python tools/fb_store_score.py --tag fbstore-290e3be --pin 290e3be3f08f0a0ebe7a364eed3410093ae9f5a0 --out
  research/findings/raw/_fb_store/verdict.json`, one re-run per not-DEFINED cell, then NEEDS-CLAUDE
  "[fb-store] RESULT".

## 2026-10-06 03:27-03:4x — SCORED: STORE-NOT-LOAD-BEARING (6/6 DEFINED)

- All six gate cells landed (s42 02:32, s43 02:42, s44 02:5x, s100/s101/s102 by ~03:45); no not-DEFINED cell, so
  no re-run was due (prereg section 3). Scored once:
  `python tools/fb_store_score.py --tag fbstore-290e3be --pin 290e3be3f08f0a0ebe7a364eed3410093ae9f5a0 --out
  research/findings/raw/_fb_store/verdict.json` -> rc 0, {"verdict": "STORE-NOT-LOAD-BEARING", "n_defined": 6,
  "n_reply_load_bearing": 0}; every seed DEFINED, load_bearing false, lesion_abstained false, engagement
  "store_lesioned", intact_answer == lesion_answer ("Sally will look in the basket for the marble.");
  precondition "cells DEFINED on 6/6 seeds" ok (measured=6 expect=6). Data + prov sidecar committed 3a48d7aa1.
- Engagement evidence on the GATE seeds is FLAG-LEVEL: each lesion cell records
  `false_belief_tom/store_lesioned = True` (env-flag echo at measurement); these cells carry no numeric
  belief_rates, so the order-of-magnitude late-window rate drop was shown on dev seeds 7-9 only. Label any
  report of gate engagement as flag-level.
- Prereg section 4's fixed reading row (mandated reporting, verbatim): "STORE-NOT-LOAD-BEARING (0 of 6 DEFINED
  seeds change the reply) | The store's attractor is not necessary on the immediate-question probe; the reply is
  carried by the host-gated write. Every load-bearing-fraction report lists `tom-false-belief` as a host-input
  lesion. Next method: a probe with simulated time between narration and question (the companion process this
  probe replaced with zero), on T18/T21." -> accordingly: this report lists tom-false-belief as a HOST-INPUT
  lesion (host-gated write carries the reply here).
- One MiMo Pro delegate (RULING models) reviewed prereg + verdict: CONFIRM, with caveats now folded in: (1)
  probe-scoped negative at REPLY level on the immediate-question probe, never "the store is not load-bearing" in
  general (dev 7-9 show the cut does move store dynamics); a method verdict, not a NO-GO (prereg section 5); (2)
  the two mandated items above (host-gated write; tom-false-belief host-input listing); (3) gate engagement =
  flag-level (this WORKLOG entry). NEEDS [fb-store] RESULT filed.
