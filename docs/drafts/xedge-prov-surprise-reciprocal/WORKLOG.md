# Draft: xedge-prov-surprise-reciprocal — WORKLOG

## 2026-10-04 10:3x — §5 step 1: choice and why (local model)

Section 4 had nothing to move (duties done; all other tasks blocked or landing-bound), so §5 (Idle: self-directed
draft research) applies — ONE draft at a time.

**Chosen topic:** close the missing reciprocal of the surprise <-> source_provenance pair: declare
**source_provenance -> surprise** (the provenance verdict feeding back into the surprise read) as a declarative
`CrossEdge` through the same generic `onebrain_crossedge_gate.run_gate` used by the prior reciprocal
(`2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md`).

**Why this one (first-open rule):**
- `GAP_CLOSURE_MISSION.md` CURRENT STATE (10-02) has no idle-draft next-item (its "Next command" = runbook §4),
  so the fallback is the master roadmap's literal "next to queue": `docs/plans/2026-07-23-MASTER-DEVELOPMENT-
  ROADMAP.md:46` — *"NEXT (queued, not yet built): another reciprocal pair from the same connectome
  (surprise->source_provenance is still one-direction only), OR the production wire-in rung"* — the wire-in rung
  is a default flip (Claude-reserved), so the reciprocal pair is the driver-open candidate.
- The one-direction fact verified in code today: `research/runners/_onebrain_integration_surprise_episodic_
  crossedge.py:41` declares `merge_organs([SURPRISE_LITE, source_provenance], wire=True, cross_edges=[surprise
  ->prov...])`; no source_provenance->surprise row exists (grep of all `CrossEdge(` rows).
- Precedent class: functionally-related areas connect reciprocally, not all-to-all (Magrou 2024; Gamanut 2018;
  Theodoni 2020 — quoted in the provenance<->self_schema reciprocal GO's §1); R1 (comprehension<->d6 WM) and R4's
  second half are the two completed pairs before this.

**Preference check (§5 step 1):** integration-first-leaning (a learned edge whose lesion is tested; stays OFF by
default — the ON-by-default production wire-in stays Claude's, recorded). Stage-1 (10-24) relevance: indirect
(source-monitoring/surprise sit beside the C1-C3 seams); marked honestly.

**Record research done (step 2, first pass):** `before_you_build.sh` recorded (corpus log line; pointers:
`_source_monitor_attractor_competition_scoping.md`, `_source_monitor_heterosynaptic_encoding_scoping.md`,
`_route_b_onebrain_seam_scoping.md`, ...); `rag_search.py ... --corpus all` top-5 = the reciprocal-GO finding
(4 hits) + the cross-edge framework gap-analysis (already-closed, no Phase-2 layer to build); read the
reciprocal-GO finding §1 (rationale + the generic-gate declaration recipe: 1-row CrossEdge + train_fn + read_fn +
two conditions) and the phase-DESIGN R3 entry (surprise -> episodic/provenance encoding gate, Lisman-Grace;
`docs/BURN_DOWN_LIST.md:146`).

**Next (step 3):** PLAN.md with one MiMo Pro delegate (question, biology, dev seeds 7 + 1-15 only,
outcome-to-reading table BEFORE any run), then a second delegate tries to refute it against CLAUDE.md's
non-negotiables; revise until it approves. Then step 4 build in `$WT/xedge-prov-surprise-reciprocal` on
`research/draft-xedge-prov-surprise-reciprocal` (new `_xedge_*` runner copy; never edit sim/, a registered runner,
a scorer or a gate), run per RULING compute-placement (pool/workstation, never this server >10 min), step 5 =
PROVISIONAL finding + verify-go + routine-merges + NEEDS "[draft-...] RESULT: ...". A draft never claims a GO;
gate seeds / preregs / scorers / default flips / sim/ stay Claude's.

## 2026-10-04 10:5x — §5 step 3: PLAN drafted, refuter verdict REVISE (8 fixes queued)

MiMo Pro delegate #1 (deleg_15ee3fb0) returned the full PLAN; written to PLAN.md (question, biology, dev seeds
[7,1,4,9,12,15], 7-row outcome-to-reading table, sign constraint: amplification edge prov_perceived -> surprise
+ attenuation as control read only, variant B pre-declared).
MiMo Pro delegate #2 (deleg_dc6a7b54) = refuter verdict **REVISE**, 8 required changes: (1) qualify
'load-bearing' as 'trace-level' (TERMS.md 09-29); (2) row 1 must not spend the governed word 'closed'; (3) fix
miscite Fremaux & Gerstner (three-factor) for the two-factor symmetric rule (grounding: sim/config.py
hebbian_symmetric, Kandel 6e Ch 54); (4) mandate the 2026-09-02 C2 per-neuron-state reset (MergedPool
_PER_NEURON_STATE) between reads + threshold-restored smoke check + counterbalance/order-invariance THIS run;
(5) row 3 must not assert a cause it cannot establish (bank honest negative if instrument clean); (6) add
lesion persistence-at-read check (slots read exactly 0 at measurement, learning OFF); (7) split JHL 1993
confidence tag (loop framing = precedent's extension, not JHL's); (8) pre-declare INTACT_FLOOR rule now
(never set post-smoke). Refuter verified the sign constraint TRUE in code (sim/bridge.py:1240-1257 clip) and
everything else compliant.
Next wake: revise PLAN.md per the 8 fixes, re-run refuter until APPROVES, then step 4 (build). No run yet.

## 2026-10-04 11:2x — §5 step 3: PLAN revised per refuter's 8 fixes

Applied all 8 to PLAN.md: (1) 'load-bearing' qualified 'trace-level' at every occurrence (probe, rows 2-3,
§6 check 3); (2) row 1 no longer spends the governed word 'closed' (reciprocal feedback loop at the synapse
level, NOT a closed capability, default-OFF wire-in=Claude); (3) plasticity citation fixed to the TWO-factor
symmetric rule grounding (sim/config.py hebbian_symmetric ~869-873; Fremaux & Gerstner kept only as the
three-factor family this rule is the cousin of); (4) risk 4 rewritten: MANDATORY Port-A reset via
MergedPool._PER_NEURON_STATE (onebrain_merge_framework.py:246-250) + smoke threshold-restore check +
read-order COUNTERBALANCED THIS RUN; (5) row 3 reworded (only 'not carried by the declared edge'; instrument
checks first, bank honest negative if clean); (6) §6 check 3 gains persistence-at-measurement (slots read
exactly 0 at every lesion re-read, learning OFF); (7) JHL confidence split (loop framing = precedent's
extension, not JHL's); (8) INTACT_FLOOR rule PRE-DECLARED (0.5 x smoke seed-7 delta, frozen before 6-seed
queue, never revisited).
Standing duties: T0 clean (flip 9q+8r draining, c1 54q, b2d 43q, iw1-R 2q+8r, stage2 8q); no new NEEDS/rulings.
Next: refuter re-run until APPROVES, then step 4 build.

## 2026-10-04 11:4x — refuter round 2: REVISE (3 fixes) applied

Round-2 refuter (deleg_a71022d6) confirmed 7/8 original fixes land correctly; 3 remaining: (a) row 5's bare
'load-bearing' -> 'trace-level load-bearing' (TERMS stated-result site); (b) §8 fail-gate didn't cover new
item (e) -> added: (e) failure at smoke = no floor freeze, no queue, report per row 6; (c) risk-4 line span
stale (246-250 -> 254-259, verified by refuter in code). All 3 applied. Next: round-3 refuter for APPROVES.

## 2026-10-04 11:5x — refuter round 3: APPROVES — §5 step 3 COMPLETE

Round-3 refuter (deleg_b1d38151): APPROVES, zero required changes. Verified via git show fa119e951 + live
reads: all 3 round-2 fixes landed (row 5 'trace-level load-bearing'; §8 dual fail-gate coherent with row 6;
_PER_NEURON_STATE span 254-259 = real definition site, 4 fields present). Non-negotiables hold; W2 line-length
clean. One pre-existing cosmetic dangling paren at L179 noted as out-of-scope (non-semantic).
Refutation arc: round 1 REVISE (8 fixes, deleg_dc6a7b54) -> applied -> round 2 REVISE (3 fixes,
deleg_a71022d6) -> applied -> round 3 APPROVES (deleg_b1d38151). Planning delegates: 3 this session
(total session delegates now 9).
STEP 3 DONE. Next wake = step 4: build `research/runners/_xedge_prov_surprise_reciprocal.py` in
$WT/draft-xedge-prov-surprise-reciprocal on research/draft-xedge-prov-surprise-reciprocal (new file copy,
sim//registered/scorers/gates untouched), embed the outcome table in the runner docstring first, then smoke
seed 7 on pool/workstation per compute-placement.

## 2026-10-04 12:3x — §5 STEP 4: BUILD (branch research/draft-xedge-prov-surprise-reciprocal)

Worktree $WT/draft-xedge-prov-surprise-reciprocal off main; branch research/draft-xedge-prov-surprise-reciprocal.
Runner research/runners/_xedge_prov_surprise_reciprocal.py (516 lines): reshaped from the 2026-09-01
reciprocal precedent (_onebrain_crossedge_provenance_to_selfschema — generic GATE_SPEC/run_gate/byte-off
harness) onto template-2's calibrated SURPRISE+source_provenance organ pair (_onebrain_integration_surprise_
episodic_crossedge — mismatch-trial drives, CUE_PA=2000, LR=0.15, HMAX=20, N_EPISODES=150, Port-A
_EXTRA_RESET_ARRAYS). Edge FLIPPED to the draft's declared direction: CrossEdge prov_perceived -> surprise
(gate provper_to_surprise), full-mask (no concept blocks on the prov side). Train = CONTRADICT trial +
ctx_perceived + perceived-exemplar content, Hebbian learning on; read = same trial + verdict HELD through the
window, N_READS=8, surprise-region windowed mean rate; delta = perceived - generated (sign +1).
Docstring = table-first: all 7 outcome rows embedded, MUST-NOT-CLAIM block, sign constraint + variant B,
dev seeds 7,1,4,9,12,15 as the --seeds default (smoke = seed 7 only; gate seeds never run here).
PROVISIONAL-only strings (tags PROVISIONAL-SUPPORTS / NOT-SUPPORTED / smoke indicator; payload key
provisional_supports; probe xedge_prov_surprise_reciprocal). INTACT_FLOOR = 0.005 PLACEHOLDER with the
pre-declared smoke rule (0.5 x smoke delta) in the comment + printed by --smoke. Smoke adds the
order-invariance check (a->b->a, drift must be <=1e-12; FAIL => do NOT queue). Selftest = read-isolation
fails-in-its-failing-direction guard (seed 7). Clean-zero control checked per seed (perc rate < 0.005).
Verified: py_compile OK; import-only check OK (GATE/SPEC/expected/docstring seeds confirmed; no pool build,
no seeds run). NOT YET RUN — smoke/selftest next, and per RULING compute-placement they do NOT run on this
server: queued to the pool (numpy CPU) next wake.
