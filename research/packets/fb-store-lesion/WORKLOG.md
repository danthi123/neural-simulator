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
