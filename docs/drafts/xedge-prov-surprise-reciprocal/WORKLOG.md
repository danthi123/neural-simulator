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
