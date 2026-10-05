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
