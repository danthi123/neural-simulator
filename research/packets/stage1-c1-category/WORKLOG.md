# WORKLOG — Stage-1 C1 category inheritance (branch `research/stage1-c1-category`)

Append-only. One line per step, newest last. The local model appends here; Claude reads this at inspection.

- 2026-09-30 09:45 Claude: worktree from main (f5e3732fe); `research/stage1-exit-gate` merged in (its C1 group and
  scorer are imported, unchanged). Record read: `before_you_build.sh` (top hits rung-3c, EMERGE-26, EMERGE-30) + RAG;
  the three findings and the rung-3c / EMERGE-26 runners read in full. Choice: rung-3c's learned member->context
  pathway with its hand-placed context->property projection replaced by a learned frame->frame association
  (prereg section 2). EMERGE-26's superordinate is a host dict literal (`SUPER = {...}`), unusable for `cat_disc`.
- 2026-09-30 10:00 Claude: organ-level dev sweeps (dev seeds 1-8, no gate seed). Bipartite member<->frame (IAC) variant
  with identical neurons: 3-hop read rates non-monotone in the learned-weight bound (mode-locking of identical
  assemblies; e.g. worm 0.000 at bound 40, 0.050 at 60, 0.000 at 90); OU 300 pA: every link grew at encoding
  (runaway). Chosen: member->frame + learned frame->frame, seeded per-neuron heterogeneity, bound 40: dev seeds
  1-8 canary `eat worm` 0.10-0.12 / `eat bug` 0.000; permuted `bug` 0.09-0.13 / `worm` 0.000; read lesion 0.000;
  lateral_learn 0.000; lateral read lesion keeps `worm` (~0.086, the link is written at encoding).
- 2026-09-30 10:25 Claude: code committed (f394633b3, 437a914f3, c081ff8a9): organ, wiring (default off), runner,
  tests (14 passed). Dev seed 7 real-handler arms queued locally behind `tools/mem_ok.sh 9` (the workstation was
  memory-contended by sibling dev runs).
- 2026-09-30 11:26 Claude: dev seed 7 `intact_a` (real handler) landed, 1142.8 s: every teach turn observed; `what does
  the canary eat` -> "perhaps the canary eats the worm [a guess from what I've learned -- not something I was
  taught]" (hypothesis_svo, recalled_svo null); organ worm 0.1192 / bug 0.0; identity controls and moat hold.
- 2026-09-30 11:48 Claude: dev seed 7 `lesion_inherit` landed, 1199.1 s: the direct probe abstains (lesion `all`,
  `lesion_held` true); the primary reply fields differ from intact on `p_canary_eat` only.
- 2026-09-30 12:08 Claude: organ dev artifact (dev seeds 1-8) + the two real-handler shards committed (a98a398d3);
  prereg committed alone (e3ef82263).
- 2026-09-30 12:09 Claude: dev seed 7 `permuted` landed, 1327.4 s: `what does the canary eat` -> "perhaps the canary
  eats the bug" (organ bug 0.1042 / worm 0.0); `what does the canary swim` recalls lake (taught in this arm).
  The remaining dev arms (`off`, `intact_b`, the three informational ablations) run behind `mem_ok.sh 9`.
- 2026-09-30 12:13 Claude: HANDOFF step 1 dry-run printed NEEDS-MERGE (main moved to 366d131f7); merged cleanly
  (9f138a93e), selftest OK, 22 passed, prereg byte-identical; GATE re-pinned to bb6705de6 and pushed.
- 2026-09-30 12:20 Claude: step 2: pool41 + pool42 provisioned at bb6705de6 (LTM OK, CORPUS OK, RUNNER OK on both).
  Step 3: 54 lines emitted (54 carry max_age_h=120) and added, 0 ADD-FAILED, queue depth 526. The HANDOFF's own
  step-3 count used `pool_queue.sh list`, which truncates commands (read 0); corrected to grep the queue file.
  Next for the local model: step 4.
- 2026-09-30 12:35 Claude: the remaining dev-seed-7 arms (`off`, `intact_b`, lateral_learn, lateral_read, draw) were
  not run: `mem_ok.sh 9` refused for over an hour (sibling dev runs held the workstation), and they are informational
  once the gate is queued. Launcher stopped; no dev process left running. Declared risk: the real-handler null was not
  checked at a dev seed (organ determinism was, 8/8).
- 2026-09-30 13:10 Claude (adversarial review of the packet): two defects fixed, nothing re-queued. (1) Scorer: a
  swallowed organ exception (an `error` in the `category_inherit` trace) read as FAIL "capability absent"; it now
  reads UNDEFINED ("not applied"), failing-first test, commit 8ff2547ed (25 tests pass, SELFTEST OK). (2) HANDOFF:
  section 0 sent a missing-shard re-queue through steps 1-3, whose step 3 re-adds all 54 lines (and, after a
  NEEDS-MERGE re-pin, at a second revision); steps 1-3 are now record-only and step 4 carries its own re-queue block
  (re-adds only the absent shards' lines, pinned to bb6705de6). Verified: the 54 queued lines pass the shape check and
  the gate-seed guard (prereg resolved), carry mem_ok 7 / memcap 9 / mem_gb=7 / max_age_h=120; both nodes hold the
  LTM bundle, the corpus, the runner and a venv at bb6705de6; queue position 469 of 522.
- 2026-09-30 14:27 Claude (review, dev seed 7 only, outputs kept out of the repo): (a) real-handler NULL: the runner's
  own `intact_b` arm at HEAD (989 s, peak RSS 6.8 GB under memcap 9) equals the committed `intact_a` (ce496d36e) on
  EVERY key of EVERY turn (sha256 of the sorted responses cec1434e9e879819 both); the declared risk is retired at
  the dev seed. (b) OFF byte-identity: the 16 cat_disc turns through `webapp.server.brain_chat` with the flag UNSET,
  main 366d131f7 vs this branch's HEAD: sha256 2290f578c98667f6 both, no `category_inherit` key. (c) Specificity at
  the dev seed: main vs flag-on `intact_a` differ only on `p_canary_eat` (full response minus the trace key); main
  equals the Stage-1 production-default dev shard (fac3f699c) on every key. (d) HEAD's organ in a fresh process
  reproduces the committed real-handler organ reads exactly (intact, permuted, lesion rates and the attribution);
  organ-only RSS about 0.5 GB.
- 2026-09-30 14:35 Claude (review): the 54 lines are no longer in `pool.queue`: a concurrent pool-maintenance task
  (pool venv lacked httpx; main 341e1490a/a4c9b5848, plus the pool-job-sandbox canaries) moved all 472 queued lines
  to `research/queue/pool.queue.held` around 14:00; all 54 C1 lines are there, unchanged. Releasing them is that
  task's job. Step 4's re-queue guard now also counts `pool.queue.held`, so a held line is never duplicated. The C1
  runner at bb6705de6 has no import gap against requirements-pool.txt (import_closure --queue on the held file).
- 2026-09-30 15:50 Claude (pool canary, pre-data fix): the held line seed 42 intact_a at bb6705de6, run on pool42
  through tools/pool_sandbox_selftest.sh (pool-job sandbox), printed 'GATE seed 42 refused' and '[stage1-c1 shard]
  worker FAILED rc=2': run_shard spawned --worker without --allow-gate-seed and main()'s guard applies to --worker, so
  none of the 54 lines could run. No line had been dispatched (no gate file at bb6705de6 on either node). Fixed in
  7762da223 (the shard hands --allow-gate-seed and --prereg to its worker; failing-first test fails at bb6705de6; 26
  passed, SELFTEST OK), merged origin/main b604085a6 at 6c2cb1362 (the merge touches no webapp/, sim/ or
  research/runners/ file; prereg byte-identical to PREREG_SHA; default anchor still False), pushed both remotes,
  provisioned 6c2cb1362 isolated on pool41+pool42 (LTM wikidata_100k + wikidata_core_15k, corpus, runner, venv check
  ok). Canary at 6c2cb1362 on pool42: rc 0, 16 turns, no errored turn, no category_inherit error, sources ['tiny-demo
  +LTM'], hash_ok, corpus present, wall 1667 s (the ~25-30 min planning figure holds on pool42). Its shard is a real
  gate shard at the pinned revision; the queued line re-runs it. New GATE row 6c2cb1362 in PINS.tsv; env.sh JOBS ->
  research/queue/requeue_stage1_c1_category_6c2cb1362.tsv (the same 54 seed x arm lines, max_age_h=168 in place of
  120). The bb6705de6 lines in pool.queue.held are retired, never released.
- 2026-10-07 02:1x Hermes (T7, HANDOFF docs/drafts/stage1-c1-inheritance/HANDOFF.md, steps 0-3; §4 order: T1's grid
  runs on the pool, so moved on). Step 0 READY (claude-review/c1-hop is an ancestor of main; HOP_FLAG present in
  c1_flip_dev_check.py). Step 1 already done (bde11b91a, 106 files tracked under gate/ incl. sidecars). Step 2 @
  DEV_SHA 2c2d559525e0508552143c2525a91270d130b1dd: pytest -q (3 files, -k "verdict or gate_ or env_parse") under
  nice -n 10 -> 12 passed, 30 deselected in 86 s; both selftests -> SELFTEST OK (organ-building tests belong to W,
  not run here). Step 3 provision via the server per RULING iw2-provision (NOT the HANDOFF's pool_provision.sh line,
  forbidden on S): touched research/queue/provision.req/2c2d55952... at 02:06 EDT; status state "running"
  (attempt 1/3, 7214 files/114 MiB + 278 MiB data -> pool41/42/40). Emitted research/queue/c1_flipdev_2c2d559.tsv
  = 96 lines, every why already carries mem_gb= and max_age_h= (no append needed); canary extract (--seed 1 --side
  on --group cat_disc) = 1 line, NOT yet queued (awaits state ok). Concurrent: T1's 144-line LDC grid at 9f625fd1c
  (mem_gb=2) dispatches ahead of the 7-9 GB C1 lines under FIRST-FIT -- deliberate, both inside max_age_h. Next:
  provision ok -> queue canary -> verify landing per the step-3 check (16 turns, flag "1", hop "1", trace on every
  turn, read hop True 30, flagged-worm guess) -> 3b emit 54 gate_hop + queue other 95 -> step 4 score -> step 5
  [c1-gatehop]/[c1-flipdev] RESULT lines. Unsure: none (no verdict written).
