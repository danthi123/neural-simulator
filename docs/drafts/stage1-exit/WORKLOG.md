# WORKLOG — Stage-1 exit gate (branch `research/stage1-exit-gate`)

Append-only. One line per step, newest last. The local model appends here; Claude reads this at inspection.

- 2026-09-29 20:50 Claude: worktree created from main (fac3f699c); record checked (`before_you_build.sh` +
  four RAG queries, one per criterion); runner, tests, prereg and HANDOFF drafted.
- 2026-09-29 21:15 Claude: dev smoke launched on dev seed 7 (word intact_a + lesion_recruit, val_heldout 2 words).
  Finding: a git worktree has no `data/corpus/` (git-excluded), so the one-brain cross-edge pool and the learned
  animacy lexicon degraded silently in the first word arms -> `corpus_present` recorded per arm and required by
  the aggregate's preconditions; the worktree's `data` linked to the primary checkout; word arms re-run.
- 2026-09-29 21:20 Claude: val_heldout 2-word smoke PASS list clean (build 153 s, first content turn 124 s,
  exposure ~8 s, feel ~2 s; sources tiny-demo +LTM; hash ok).
- 2026-09-29 21:45 Claude: word/lesion_recruit (dev seed 7, corpus present) landed: 811 s wall (build+first turn 183 s,
  teach ~23 s, probe ~12 s, two lazy-organ turns 365 s and 168 s); patch called 21x; every new-word probe abstains
  or mis-resolves (`what does it eat` -> the built-in cat/fish), `what does the dog chase` -> cat survives; source
  tiny-demo +LTM; hash ok. The lesion behaves as the prereg predicts. cat_disc/intact_a and val_novel/intact_a launched.
- 2026-09-29 21:55 Claude: word/intact_a (dev seed 7) landed: 886 s wall; bear->otter, otter->clam, both moats,
  pre-exposure abstain and dog->cat all as required; `what does it eat` resolved `it` to the built-in cat (the D6
  referent store is a build-time vocabulary). DESIGN DECISION before any gate seed: that sub-probe is REPORTED,
  never counted (`anaphora_it_eat`); prereg 4.2 says why. Lesion contrast on the two counted probes: differs on
  both, unchanged on the built-in fact and the moats.
- 2026-09-29 22:20 Claude: cat_disc/intact_a (dev seed 7) landed: 1045 s wall; eleven teach turns stored; both
  inheritance probes abstain (direct: recall path exercised, no block; generative: the open-ended branch fired and
  proposed nothing); identity controls and moat correct. The predicted C1 FAIL shape, instrument intact.
- 2026-09-29 22:30 Claude: lived/intact_a (dev seed 7) landed: 2813 s wall (the forced numpy BTSP write costs
  1075 s on the teach turn and 500-700 s on each later verified recall); every C3 check as required (recall after
  three distractors, the dAP disclosure with in_memory true, both pre-exposure abstains, the moat, dog->cat).
  Wart: "tell me about the sky" was ACQUIRED as [tell, about, sky]; distractors changed to questions before the
  prereg commit; FAILURE_LOG row added. ETA revised in the prereg (section 10) and the HANDOFF: ~95 job-hours.
- 2026-09-29 22:33 Claude: commit 1 = the prereg alone: 2a7097ccc.
- 2026-09-29 22:45 Claude: commit 2 was blocked by gates/attribution_required (the runner measured lesion vs
  intact arms and never subtracted the null). Fixed the cause: the aggregate now calls tools.lab.attributable_to
  per seed and lesion (treatment = primary-field differences intact_a-vs-lesion on the scored turns; control = the
  same count intact_a-vs-intact_b; C4 uses |r_intact - r_lesion| vs |r_a - r_b|), recorded as `attributable` in
  each lesion record; the aggregate's prints go to stderr so `--aggregate > report.md` stays clean.
- 2026-09-29 22:58 Claude: origin/main moved (serving-simplify merge, 5 commits) after commit 2; merged it in
  cleanly (4f6d9d7fb; FAILURE_LOG appends auto-merged), selftest/tests green, prereg byte-identical to
  PREREG_SHA; GATE_SHA re-pinned to the merge commit so pool_provision.sh accepts it.
- 2026-09-29 23:12 Claude: val_novel/intact_a (dev seed 7) landed: 442 s wall; the company moves the reply-visible
  mood (+0.068 after the glorp company, -0.061 after the blick company, read from the feel reply text), and each
  novel word ALONE holds the prior mood exactly (delta 0.0 on both probes) -> inherits False, the predicted C4b
  FAIL at production defaults (a non-lexicon word contributes zero by code); the E2 world-model surprise notice
  fired on each company flip. Instrument intact. All six dev arms now landed; HANDOFF section 2 PASS list holds.

- 2026-09-29 23:1x (Claude): GATE_SHA 4f6d9d7fb69124a4d16797a236ac388b67eb73c6 provisioned isolated on pool41+pool42 (LTM OK, CORPUS OK on both); 204 jobs queued, 0 ADD-FAILED; pool depth 551 after queueing. AWS lane NOT used (NEEDS-CLAUDE #13 ruled: the pool ETA of about Oct 2-5 lands inside the Oct 24 window and before the Pro downgrade).
- 2026-09-30 15:25 Claude (pool canary, pre-data fix): the 204 lines at 4f6d9d7fb never ran: a pool maintenance hold
  moved them to research/queue/pool.queue.held before any dispatch, and a canary of the same runner shape
  (stage1_c1_category at bb6705de6, pool42, in the pool-job sandbox) showed every gate shard dies rc=2: run_shard
  spawned --worker without --allow-gate-seed and main()'s guard applies to --worker. Fixed in 38fd327f3 (the shard
  hands --allow-gate-seed and --prereg to its worker; failing-first test fails at 4f6d9d7fb; 9 passed, SELFTEST OK),
  merged origin/main b604085a6 at c238ff325 (the merge touches no webapp/, sim/ or research/runners/ file; prereg
  byte-identical to PREREG_SHA), pushed both remotes, provisioned c238ff325 isolated on pool41+pool42 (LTM
  wikidata_100k + wikidata_core_15k, corpus OK, venv check ok). Canary at c238ff325: held line seed 42
  val_novel/intact_a through tools/pool_sandbox_selftest.sh on pool41: rc 0, 17 turns, no errored turn, sources
  ['tiny-demo +LTM'], hash_ok, corpus present, wall 740 s (1.7x the dev-seed-7 workstation 442 s; a stage2 dev shard
  ran 3.6x slower on pool42, so the section-5 ETA is optimistic). That canary's shard is a real gate shard at the
  pinned revision (the queued line re-runs it). GATE_SHA re-pinned to c238ff325; the 204 replacement lines (same seed
  x group x arm set, max_age_h=168) are research/queue/requeue_stage1_exit_gate_c238ff325.tsv.
- 2026-10-04 01:0x Hermes (T6 sections 6-7): section-6 count = **204/204** (exact filtered count; the 7 new
  stage1_exit files were s102 val_novel cells). `rsync` not installed on S (skipped, noted) → `cp -a` sync into
  the worktree, then `SIM_NO_PROVENANCE=1 $PY -m research.runners.stage1_exit_gate --aggregate research/findings/
  raw/_stage1_exit/gate --seeds 42,43,44,100,101,102 --out .../aggregate.json` → **AGG_RC=0**, report at
  `research/queue/stage1_exit_report.md`. Verdict: **Stage-1 exit FAIL** (C1 FAIL, **C2 PASS**, C3 FAIL, C4
  FAIL) — the expected prereg-section-9 measurement. Finding written per section 7's exact header + report block
  unchanged + section-8 fixed sentences: `research/findings/2026-10-04-stage1-exit-gate-RESULT.md`.
- Board line (Claude moves it): 2026-10-04 Stage-1 exit gate (first integrated stage-exit attempt): FAIL; C1 FAIL, C2 PASS, C3 FAIL, C4 FAIL (r per seed: 42 0.938, 43 0.946, 44 0.951, 100 0.947, 101 0.946, 102 0.944). Finding: research/findings/2026-10-04-stage1-exit-gate-RESULT.md.
- C2 PASS triggers section 10 #7 → NEEDS-CLAUDE filed on main (verify-go before any board/ledger word for the
  PASS; stage FAIL itself needs no NEEDS). `bash tools/push_both.sh` skipped (push reserved for Claude); commit
  is local on `research/stage1-exit-gate`.
- 2026-10-04 17:14 RULING s1-c4-void step 1 (re-provision): touched research/queue/provision.req/c238ff325a38e40a68b1f9751822630ff729377b
  per RULING iw2-provision -> /opt/host-status/sim-provision/ reports state=already (pool41/42 keep .provisioned_ok;
  pool40/wspool stale-guarded, optional). The ruling's confirm step fails: ssh pool41/pool42 'ls ~/derisk-pool/revisions/
  c238ff325*/research/findings/raw/_affect_learned_valence_map.json' = No such file on BOTH nodes, while S's main has the
  map. Per the ruling's order (confirm THEN re-queue): the 72 val_heldout+val_novel lines of
  research/queue/requeue_stage1_exit_gate_c238ff325.tsv are NOT queued (gate -> gate_c4rerun out-dir rewrite staged for
  when the map ships). NEEDS [s1-c4-void] filed on main; no code or queue touched.

- 2026-10-04 18:55 s1-c4-void continued (RULING s1-c4-topup): the owner's tools/pool/ship_runtime_inputs.sh top-up
  landed (50b949e21); confirm step PASSED - `ls .../c238ff325*/research/findings/raw/_affect_learned_valence_map.json`
  exit 0 on BOTH pool41 and pool42. Queued the 72 val_heldout + val_novel lines of
  research/queue/requeue_stage1_exit_gate_c238ff325.tsv (field 1 verbatim, out path under
  _stage1_exit/gate_c4rerun/, field 2 as the --checked reason): ok=72 fail=0, queue depth 105 -> 177.
  The TSV stays an untracked working file (agent main-commit backstop: queue TSVs are outside
  the routine classes). Next: cells land -> guarded-scorer step of RULING s1-c4-void.

## 2026-10-05 10:55 EDT - C4 rerun scored (guarded scorer) + affect audit

- 72 val lines (val_heldout + val_novel, 6 seeds) complete under
  research/findings/raw/_stage1_exit/gate_c4rerun (144 records). Scored:
  `python -m research.runners.stage1_exit_gate --aggregate .../gate_c4rerun
  --seeds 42,43,44,100,101,102` (AGG_RC=0) -> gate_c4rerun/aggregate.json.
- Verdict: val_heldout FAIL (6/6 correct, lesion_affect 6/6 reproduced, but
  permuted exposure tracked on 4 of 6 seeds); val_novel FAIL (0/6, capability
  absent on 6 of 6 seeds); per-seed intact r 0.805-0.836 defined, lesion r
  None (all-neutral guard -> UNDEFINED, not 0). Stage verdict reads UNDEFINED
  (INCOMPLETE): this dir carries C4 only, 132 other-group shards missing by
  design. This replaces C4's earlier UNDEFINED(instrument) reading (RULING
  s1-c4-void): the data are now defined and the verdict is FAIL.
- Audit (RULING s1-c4-void): docs/drafts/stage1-exit/AUDIT-affect-pool-results.md
  lists 702 pool results since 09-01 across 23 runners that import
  affect_production_organ (top: stage1_exit_gate 335, _lbf_affect_conditioned_mouth
  193, _affect_learned_vocabulary_derisk 67); NEEDS INFO filed for Claude to decide
  which are void.

## 2026-10-05 11:05 EDT - audit widened (RULING runtime-inputs)

- The [s1-c4-void] audit doc now has three sections, all report-only:
  A) affect_production_organ importers: 702 results / 23 runners;
  B) WKV mouth + e-prop head users (learned head default since 09-02):
  577 results / 58 runners; C) BRAIN_AFFECT_LEARNED_VOCAB references:
  403 results / 3 runners. Detection = prov runner field vs static source
  grep; 14,764 prov sidecars scanned since 09-01. Claude rules what is void.

## 2026-10-05 11:50 EDT - s1-c3-rerun started: P2 provisioned, A1.2 dev shards queued

- P2 = 7851dd220 (main HEAD at start). provision.req touched; state=ok on
  pool40/41/42/wspool (~5 min). before_you_build 'C3 lived rerun at P2: dev
  seed-7 shards must show lesion_lived_recall patch.calls>0 ...' rc=0, recorded
  to corpus_checks_shared.jsonl.
- Queued the two AMENDMENT-A1 A1.2 dev shards: seed 7, group lived, arms
  intact_a + lesion_recall, labels t_wolf_bite,p_wolf_bite,c_dog_chase, out
  research/findings/raw/_stage1_exit/dev_c3/s7/. Line shape cloned from the
  c238ff325 gate lines (mem_ok 7 1, memcap 9, --prereg, #checked); added 2,
  failed 0, depth 10. No --allow-gate-seed (dev seed).
- Next, after both land: require lesion_recall patch.calls > 0, its p_wolf_bite
  abstains or is not apple, and c_dog_chase still names cat; if calls == 0
  STOP + NEEDS per A1.2. Then --emit-pool-jobs 7851dd220, keep only --group
  lived (42 lines) with out-dir _stage1_exit/gate_c3rerun/, canary one line
  first, queue the rest with #checked naming AMENDMENT-A1, then score and
  RESULT + NEEDS [s1-c3-rerun].

## 2026-10-05 13:05 EDT - s1-c3-rerun: A1.2 dev check PASSED, 42 gate lines emitted, canary launched

- Both dev_c3/s7 shards landed (12:56). lesion_recall: patch.calls=3
  (lesion_lived_recall engaged; calls_total=10), p_wolf_bite answers
  "I don't know about that." abstained=True (not apple), c_dog_chase still
  names cat. intact_a answers "the wolf bites the apple" (capability
  present). All three A1.2 required criteria PASS; no STOP.
- Emitted with --emit-pool-jobs 7851dd220 --seeds 42,43,44,100,101,102
  (204 lines), kept the 42 --group lived (7 arms x 6 seeds), out-dir
  rewritten gate/ -> gate_c3rerun/ per A1.2. Lines at
  /opt/data/cache/scratch/s1c3_gate_lines.tsv (cmd<TAB>record).
- First gate-line add refused: no pool canary record for
  (research.runners.stage1_exit_gate, 7851dd220). Launched
  bash tools/pool_canary.sh run pool40 '<seed42 lesion_recall line>'
  --wait-s 5400 (pool40 holds the revision; verified over ssh). Known
  launch-hang defect NEEDS [pool-canary]: if it exits 3 while the job
  runs, poll the status file and record the verdict manually.
- Next: on canary PASS queue the remaining 41 with #checked naming
  AMENDMENT-A1; score gate_c3rerun after all 42 land; RESULT + NEEDS
  [s1-c3-rerun].

## 2026-10-05 14:10 EDT - s1-c3-rerun: canary PASS recorded, all 42 gate lines queued

- Canary stage1_exit_gate_7851dd220_pool40_20261005T130424 finished
  rc=0 wall=3006s (10 turns, seed42 lived lesion_recall): patch.calls=4,
  p_wolf_bite abstains, c_dog_chase names cat, error=None, pythonhashseed_ok,
  corpus_present, provenance stamped. The launcher's rc=3 was the known
  false-negative ssh launch (NEEDS [pool-canary]); detached job ran. Recorded
  PASS via pool_canary.sh record ... PASS pool40 gate (row 14:05).
- Queued all 42 AMENDMENT-A1 A1.2 C3 gate lines (7 lived arms x 6 seeds,
  42,43,44,100,101,102) from --emit-pool-jobs, out gate_c3rerun/, record cites
  AMENDMENT-A1: added=42 failed=0. Accounted: 5 running + 37 queued = 42.
- The canary's own artifact lives under
  raw/_pool_canary/stage1_exit_gate_7851dd220_pool40_20261005T130424/ and is
  NOT one of the 42; seed42 lesion_recall re-runs from the queue.
- Next: when all 42 land, --aggregate gate_c3rerun --seeds
  42,43,44,100,101,102; apply the A1.2 scorer rule; RESULT + NEEDS
  [s1-c3-rerun].

## 2026-10-05 16:10 EDT - s1-c3 mid-flight sanity at 10/42: seed-42 selective double-dissociation HOLDS
- Scanned the 10 landed gate_c3rerun results: error=None everywhere, pythonhashseed_ok, corpus present.
- `lesion_episodic` shows patch=null BY DESIGN (env-var lesion BRAIN_EPISODIC_LESION=1, stage1_exit_gate.py:214)
  -- an early patch.calls==0 alarm was my wrong assumption, not a defect.
- Seed-42 selectivity (the C3 re-measure's core claim): lesion_episodic abstains p_wolf_discuss ("I don't
  recall us discussing wolf") but still answers p_wolf_bite; lesion_recall (patch lesion_lived_recall,
  calls=4) abstains p_wolf_bite ("I don't know about that") but still answers p_wolf_discuss; intact_a
  answers both. Each lesion cuts its own probe and spares the other -- the double dissociation attempt 1
  could not show.
- No queue action needed; 10/42 landed, 17 running, 25 queued (as of 16:06).

## 2026-10-05 21:25 EDT - s1-c3-rerun aggregate: C3 PASS-WITH-CAVEAT (verify-go)

All 42 gate shards landed (7 lived arms x 6 seeds, pin 7851dd220). `.venv/bin/python -m research.runners.stage1_exit_gate --aggregate research/findings/raw/_stage1_exit/gate_c3rerun --seeds 42,43,44,100,101,102` (AGG_RC=0) -> lived/C3 primary PASS 6/6: lesion_recall 6/6 + lesion_episodic 6/6 lesion effect, reproduced 6/6, permuted correct 6/6; attribution 12/12 cells 100.0% (treatment +3/+2, control +0). Stage line unchanged: UNDEFINED (INCOMPLETE) - C1/C2/C4 shards absent from this dir by design (Missing shards 150); across the record C1 INHERITANCE FAILED, C2 PASSED, C4 FAILed on defined data.

verify-go (one MiMo Pro delegate per RULING models; 7 lenses: reproducibility/power, gate-cheat, control-integrity, instrument-trust, selectivity/mass, process+prereg+pin, scope/stage): PASS-WITH-CAVEAT. Survived: read-only re-run of the scorer's own aggregate() reproduced the table; prereg bar 4.3 unchanged (no post-data weakening; A1.2 strictly harder); all 42 shards git_sha 7851dd220 git_dirty=false pythonhashseed_ok corpus_present; double dissociation verified seed-by-seed 6/6 (lesion_recall cuts only p_wolf_bite - patch calls=4/30, 9 other labels byte-identical incl. c_dog_chase naming cat; lesion_episodic env-lesion cuts only p_wolf_discuss, spares p_wolf_bite); permuted arm names PEAR 6/6; pre-exposure probes abstain 6/6; intact_a==intact_b on 10 labels x 6 seeds; no floor/ceiling (instrument moves both ways).

CAVEAT (carried into the RESULT line): (1) criterion-scoped - C3 PASS does not move the stage verdict, which stays UNDEFINED (C1 FAILED, C4 FAILED, C2 passed); (2) the 100%-attributable 'control' is intact_a-vs-intact_b, the run-to-run null - it means 'not run noise', not 'no other co-varying cause'; (3) the scorer gates the PASS on intact/lesion/repeat/permuted controls but does not gate the cross-probe sparing or the attribution figure (reported only; both satisfied on 6/6). The 16:10 monitoring false alarm stands resolved: lesion_episodic patch=null is by design (stage1_exit_gate.py:214 env-var lesion).

2026-10-06: Board synced per RULING s1-c3c4-read, including a backfill of the never-executed RULING
board-session6 — Stage-1 attempt-1 verdict stamped in docs/.vikunja_sync; witness/BG-seam/host-share
notes appended to board #232/#237; new task #244 "Redesign the feeling test's scrambled-values control
(Claude)" (p3, due 2026-10-24, needs owner sign-off). The two "Re-run the Stage-1 ..." tasks the ruling
closes do not exist on the board (board-session6's add-task step was never executed); the non-existence
is recorded in the marker rather than a create-then-close.
