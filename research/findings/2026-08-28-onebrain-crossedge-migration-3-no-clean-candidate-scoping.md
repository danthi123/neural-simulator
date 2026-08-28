---
type: finding
status: scoping
date: 2026-08-28
mechanism: onebrain-crossedge-migration-3-candidate-search
lane: onebrain-integration
builds_on:
  - research/findings/2026-08-28-onebrain-r4-declarative-crossedge-migration-GO.md
  - research/findings/2026-08-27-onebrain-completeness-audit.md
  - research/findings/2026-08-27-onebrain-production-integration-SCOPING.md
  - research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md
  - research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md
  - research/findings/2026-08-27-affect-tone-spiking-mouth-flip-confirmed-GO-6seed.md
---

# THIRD declarative-CrossEdge production migration — read-only candidate search finds NO clean, non-redundant, wiring-ready pair; every named candidate is blocked by something OTHER than "needs re-expression" (a read crux, a missing structural seam, or a mechanism that never used `CrossEdge` at all)

**One-line:** re-checked all four candidates named as the R4 migration's own ranked-next (R3 surprise->episodic,
d5_episodic, affect->tone/mouth, R2), plus a fifth surfaced during this search (R3-v3 DA-credit-gated
d6->comprehension), against the primary sources (findings + the framework's own `GROUP_A`/`GROUP_A_DEFERRED`
registry + the live `spiking_mouth_recall_prod.py`/`onebrain_merge_framework.py` code). **None qualifies.** Per
the task's own stop condition, this is the deliverable for Phase 1: no code was written, no flag was added: the
next rung on THIS specific lane (a third learned faculty-pair edge onto the declarative `CrossEdge` framework) is
genuinely not available yet; the productive next steps live in three OTHER lanes (a read-instrument fix, a
structural-seam build, and an owner-gated flip), enumerated in §6.

## 1. R2 (three-factor rule) — pre-excluded per the task brief; re-confirmed, not re-litigated

<!--derived-->

Confirmed against `research/findings/2026-08-27-onebrain-integration-R2-threefactor-selforganized.md` and the R4
finding's own §1: R2 upgrades the plasticity RULE on the IDENTICAL `d6_multiref_wm -> comprehension` region pair
R1 built — the same pair already flipped default-ON in production (`fe1911f2`,
`2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.md`). Not a second faculty-pair edge; out of
scope for "migrate a new pair" as the task brief already states.

## 2. R3-v3 (DA-credit-gated cross-edge) — a fifth candidate found during this search, and ALSO the same R1 pair

<!--derived-->

Not named in the task brief, but surfaced while tracing the framework's `GROUP_A` registry and the completeness
audit (`research/findings/2026-08-27-onebrain-completeness-audit.md:76-78`), which lists THREE framework-GO
edges as of 2026-08-27: R1, R3-v3, R4 — and states verbatim: **"R1 and R3-v3 are the SAME organ pair under two
rules."** R3-v3's own finding
(`research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md`) confirms this directly: "the
DA-credit-gated cross-edge (**d6 WM referent -> comprehension** sel_agent/sel_patient role competition)" —
literally R1's region pair, with a spiking dopamine-population credit signal substituted for R2's host-scalar
reward and R1's plain two-factor Hebbian rule. Ruled out for the identical reason as R2: it is a third
LEARNING-RULE variant of the one pair already in production, not a second pair. (This also means the project's
"R" numbering is NOT globally sequential — R1/R2/R3-v3 name three rule-variants of ONE pair; "R3" in the R4
finding's own ranked-next list names a DIFFERENT edge, `surprise -> source_provenance/episodic`, which the R4
finding itself labels "R3 (surprise->episodic/provenance)". Recorded here so the collision does not cause
re-derivation later: **there is no "R5"; the two "R3"s are distinct sub-arcs that happen to share a prefix.**)

## 3. R3 (surprise -> source_provenance/episodic encoding gate) — confirmed a READ crux, not a wiring gap

<!--derived-->

Read `research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md` in full (not just the R4
finding's summary). Confirmed directly from the primary source:

- The edge's CONSTRUCTION already uses the declarative framework — it is `CrossEdge`'s SECOND pre-existing
  consumer (`_onebrain_integration_surprise_episodic_crossedge.py:218`), predating even R4's migration. There is
  nothing to re-express.
- F1/F3/F4/emergence/lesion-recovers-migration are clean 6/6 on the current committed artifact
  (`_onebrain_integration_surprise_episodic_crossedge_6seed.json`).
- The blocker is F2 (vary-then-lesion): the raw delta misses its pre-registered floor on 6/6 seeds even after the
  one dose-response fix tried (`hebbian_learning_rate` 0.05->0.15), AND — the decisive point — the F2 lesion
  CONTROL itself fails its own precondition on 5/6 seeds (`delta_lesion` does not fall under 34% of
  `delta_intact`), which is why `tools/verdict.py`'s precondition framework correctly reports **UNDEFINED, not a
  validated negative**. An undefined read-fidelity crux is a different kind of work (fix or replace the
  instrument) than "wire an already-characterized edge into production" — exactly the distinction the task brief
  drew in advance ("if the candidate's blocker is the read, it is NOT a clean wiring-migration; skip it").
- Also confirmed non-redundant with board #129 (`source_provenance`'s own perceived-vs-generated mechanism) per
  the R4 finding's own check — this edge externally FEEDS one existing input of that mechanism rather than
  duplicating it. That part of the candidate is genuinely clean; the F2 read is what blocks it, not redundancy.

**Verdict: skip, confirmed. Ranked #1 in §6 as the next step for a DIFFERENT lane (instrument fix), not this
one.**

## 4. d5_episodic — confirmed NOT wiring-ready: no cross-edge exists to migrate, and the framework itself says why

<!--derived-->

Checked the primary source directly, not just the R4 finding's summary: `onebrain_merge_framework.py`'s own
`GROUP_A_DEFERRED` registry (`:2148-2149`) carries d5_episodic's exact deferral reason, verbatim: `"Heavy own-pool
— a ~2000-neuron CA3 with two-compartment apical dendritic-dAP + slow-NMDA reverberation + BTSP formation.
Group-C own-pool + apical/NMDA-slow seam."` It is absent from `GROUP_A`/`REGISTRY` entirely — `grep
'OrganDescriptor(key=' onebrain_merge_framework.py` lists 7 organs (`causal_whatif`, `comprehension`,
`self_schema`, `source_provenance`, `curiosity`, `prospective_memory`, `d6_multiref_wm`); `d5_episodic` is not
among them. `d5_episodic_production_organ.py` (the LIVE production organ, already default-ON via
`BRAIN_EPISODIC`, per its own module docstring) confirms it independently: it builds its own standalone
`EpisodicDapMemory` bridge, conversation-scoped, with no `merge_organs`/`CrossEdge`/`OrganDescriptor` import at
all.

There is consequently no bespoke, already-F1-F4-characterized d5_episodic cross-edge anywhere in the repo to
re-express declaratively (unlike R1 and R4, which both started as a working bespoke edge that only needed
re-expression). Building one from scratch would mean designing + validating a NEW learned mechanism first — the
completeness audit's own ranked-next list makes the same call: `"Migrate d5_episodic (Group-C own-pool seam) so
the memory hub can accept cross-edges (R3/R5 target it)"` is listed as a PREREQUISITE step, separate from and
prior to any cross-edge question. Confirms the R4 finding's own characterization ("no evidence found of this
having started; still open, needs its own multi-bridge/apical-dendrite seam before any cross-edge question
applies").

**Verdict: skip. Not wiring-ready — blocked on a structural prerequisite, ranked #3 in §6.**

## 5. affect -> tone/mouth — confirmed it does NOT route through `CrossEdge`/`OrganDescriptor` at all

<!--derived-->

The R4 finding flagged this "ambiguous... not verified this pass." Resolved here by reading the actual mechanism
(`research/findings/2026-08-27-affect-tone-spiking-mouth-flip-confirmed-GO-6seed.md` +
`spiking_mouth_recall_prod.py`, `mouth_tone_marker`, `_apply_mouth_mood_tone`) and grepping the live code:
`grep -n "merge_organs\|CrossEdge\|OrganDescriptor" research/runners/spiking_mouth_recall_prod.py` returns
**zero matches**. The mechanism is a **rate-vs-rate spiking READ** (`mouth_tone_marker`, its own small bridge,
independent of whatever composer/pool backs the chat turn) whose scalar output selects a punctuation marker
(`'!'`/`'.'`) via host string logic (`_apply_mouth_mood_tone`) — a neural-signal-driven READOUT modulating a
host articulation choice, not a learned cross-region PLASTIC SYNAPSE connecting two `OrganDescriptor` pools the
way R1/R3-v3/R4/R3-surprise all are. It is architecturally a different class of integration (closer to the
project's other "faculty DRIVES the reply" couplings, e.g. #84/#85 in the 2026-08-25 session, than to a
`CrossEdge`). It also is not a migration TARGET in any sense — it is already production default-ON
(`BRAIN_SPIKING_MOUTH_MOOD`, `_MOUTH_MOOD_DEFAULT_ON = True`) via this alternate mechanism, confirmed 6/6 GO
plus a real-production-composer-kind existence check in the cited finding. There is nothing here for the
declarative `CrossEdge` framework to migrate.

**Verdict: not applicable to this framework at all; already shipped by a different, already-validated mechanism.
Not ranked in §6 — no action implied.**

## 6. Ranked next (for the LANES this search actually surfaced work in — none of them is "migrate edge #3")

<!--derived-->

1. **Fix or replace the F2 read/lesion-control instrument for the surprise->source_provenance edge (§3).** The
   construction is already declarative and 6/6 clean on every OTHER arm; only the read is undefined. This is the
   single most promising residual found this session — closing it would make R3-(surprise) the genuine third
   migration-ready edge, but the work is instrument design, not wiring.
2. **The companion-process read-determinism residual the R4 finding logged** (§4(c) of
   `2026-08-28-onebrain-r4-declarative-crossedge-migration-GO.md`, `research/FAILURE_LOG.md` 2026-08-28,
   `NOT-GATEABLE` pending design) — worth checking whether it also affects the ALREADY-flipped R1 edge's live
   reads.
3. **Build d5_episodic's own multi-bridge/apical-dendrite seam** (§4) — a genuinely new, larger piece of work
   (Group-C own-pool integration into the framework) that must land BEFORE any cross-edge to/from d5_episodic is
   even askable. Not started; no evidence of scoping beyond the one-line deferral reason.
4. **R4's outer default-ON flip** (`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA`) — owner-gated per the R4 finding and this
   task's own brief; ready whenever that decision is made. Not a new migration.
5. **R2/R3-v3's rule upgrade folded into the already-flipped R1 production edge** — a refinement, not a new pair;
   lowest priority of the five, per the R4 finding's own ranking.

No `sim/` file touched. No `webapp/server.py` edit. No new flag added. No branch of `CrossEdge`/`merge_organs`
changed. This finding is read-only research: primary sources checked directly (not just prior findings'
summaries), each verdict traced to a grep/file/line, consistent with `feedback_read_own_substrate_before_theorizing`.

## Pre-existing artifacts grounding this scoping's per-candidate claims

<!--derived-->

This session ran no new experiment; every quantitative claim above restates a value already committed in one of
these artifacts (backing the findings cited inline in §1-§5), read directly, not re-derived:
`research/findings/raw/_onebrain_integration_r2_threefactor_selforganized_6seed.json` (R2, §1),
`research/findings/raw/_onebrain_integration_r3v3_functional_drive_6seed.json` (R3-v3, §2, `n_go`/per-seed
`delta_agent_intact`),
`research/findings/raw/_onebrain_integration_surprise_episodic_crossedge_6seed.json` (R3-surprise, §3, the F2
precondition-failure fractions),
`research/findings/raw/_onebrain_declarative_crossedge_r4_repro_6seed.json` and
`research/findings/raw/_onebrain_xedge_selfschema_production_declarative_6seed.json` (R4, referenced for
comparison in §2/§6),
`research/findings/raw/_spiking_mouth_recall_soak.json`,
`research/findings/raw/_affect_tone_spiking_mouth_fix_verify.json`, and
`research/findings/raw/_affect_tone_mood_onebrain_composer_kind_check.json` (affect->tone/mouth, §5, its 6/6 GO
+ real-composer-kind check).

## Files consulted (all read directly, not RAG-paraphrased)

`research/findings/2026-08-28-onebrain-r4-declarative-crossedge-migration-GO.md` ·
`research/findings/2026-08-27-onebrain-completeness-audit.md` ·
`research/findings/2026-08-27-onebrain-production-integration-SCOPING.md` ·
`research/findings/2026-08-27-onebrain-integration-R2-threefactor-selforganized.md` ·
`research/findings/2026-08-27-onebrain-integration-R3v3-functional-drive-GO.md` ·
`research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md` ·
`research/findings/2026-08-27-affect-tone-spiking-mouth-flip-confirmed-GO-6seed.md` ·
`research/runners/onebrain_merge_framework.py` (`GROUP_A`, `GROUP_A_DEFERRED`, `REGISTRY`) ·
`research/runners/d5_episodic_production_organ.py` · `research/runners/spiking_mouth_recall_prod.py` ·
`GAP_CLOSURE_MISSION.md` · `bash tools/before_you_build.sh "next declarative cross-edge migration after R4
self_schema source_provenance"` (RAG corpus check; the tool's own run auto-logs its query, per its stdout).

Functional read-outs only; no phenomenal-experience claim.
