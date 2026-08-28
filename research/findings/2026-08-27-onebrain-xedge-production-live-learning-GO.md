---
type: finding
status: live
date: 2026-08-27
mechanism: onebrain-xedge-production-live-learning
lane: one-brain/integration/production
artifacts:
  - research/findings/raw/_onebrain_xedge_live_learning_6seed.json
runner: research/runners/onebrain_xedge_production.py
builds_on: research/findings/2026-08-27-onebrain-xedge-production-frozen-GO.md
---

# One-brain PRODUCTION integration PART 2 — the d6-WM->comprehension cross-edge now GROWS from an IN-BRAIN
self-supervised credit signal (0.05 -> ~16.7, bounded) AND CLOSES the sub-decision caveat: the live-learned edge
FLIPS the real production repair-role DECISION with the held WM referent, lesion-attributable — GO 6-seed

**One-line:** PART 1 wired a FROZEN pre-grown cross-edge that drove the read but was SUB-DECISION (never changed the
turn's content). PART 2 removes both limits. (1) LIVE-LEARNING: the edge starts near-zero (W0=0.05) and GROWS
through use from an IN-BRAIN, self-supervised credit signal — comprehension's OWN confident spiking sel resolution
drives `teach_*` (three-factor, DA-coincidence-gated, bounded by `stdp_w_max`), NOT a host ground-truth schedule and
NOT a pre-grown weight. (2) CAVEAT CLOSED: on a content-ambiguous transitive the cross-edge RESOLVES the role via
R3-v3's validated balanced (content-cancelled) read, so the real production `repair_target` role DECISION now FLIPS
agent<->patient with the held WM referent (the clarification wording changes) and REVERTS under cross-edge lesion.
6-seed GO (42/43/44/100/101/102): edge grows 0.05 -> ~16.7 following the per-seed RANDOM role assignment, bounded
(F3), and the repair decision flips 5/5 ambiguous items INTACT, 0/5 LESIONED, on every seed. Default-OFF,
byte-identical-off.

Artifact: `research/findings/raw/_onebrain_xedge_live_learning_6seed.json` (n_go 6/6).

## (1) LIVE-LEARNING — the credit signal, and why this one

**Chosen: comprehension's OWN confident sel resolution as the self-supervised teacher.** Per learning turn: hold a
WM candidate pool, present role-resolving discourse, READ the brain's own spiking resolution (`amb_read` sel
margin), and IFF the comprehension is CONFIDENT (|margin| > conf) drive `teach_{the-role-the-brain-resolved}` — the
DA-coincidence population then credits `w{held}->sel_{resolved}`. No host ground-truth label anywhere; the credit
VALUE and DIRECTION are both read off the substrate's own spikes. Justified over the D2-surprise alternative
because it is the TARGET organ's own success signal (the WM->comprehension edge is credited by comprehension
succeeding) — the most direct "organs learn to drive each other" coupling, and it reuses an organ already on the
live path. This closes the scoping's "no live credit signal in production" blocker with an in-brain signal.

**Growth (6 seeds, from the cited artifact's `grow_traj`).** The edge rises from 0.05 to ~16.7 <!--derived--> on the
CORRECT `w{p_agent}->sel_agent` / `w{p_patient}->sel_patient` edges, following the per-seed RANDOM role assignment
(anti-cheat): seed 42 p_agent=w0 -> `w0->A`=16.779; seed 43 p_agent=w1 -> `w1->A`=16.643; seed 44 p_agent=w2 ->
`w2->A`=16.846 (seeds 100/101/102 land the same way at their own random assignment, max weight ~16.7-16.85
<!--derived-->). Every seed (6/6): `grew_both=True`, `bounded_F3=True` (all weights <= `stdp_w_max`=20, no runaway),
`n_credited`=80/80. The un-credited p_ctrl pool's edges stay at 0.05 exactly. Growth is SELECTIVE though not clean:
the correct edge (~16.7) dominates the same-pool wrong-role edge (~4-6 <!--derived-->, partial cross-growth) by
~3x — declared, not hidden.

## (2) CAVEAT CLOSED — the WM now changes a real conversational DECISION

The PART-1 drive was sub-decision because the production judge reads `|a0-a1|` / net-lean, which partly cancel a
symmetric bias and sit far from their boundaries. The fix: on a content-ambiguous transitive (the repair/abstain
path — content did not resolve WHICH referent plays which role), the cross-edge RESOLVES the held referent's role
via R3-v3's VALIDATED balanced `amb_read` (drive both animacy cue directions equally so CONTENT cancels to ~0; the
cross-edge is then the ONLY thing that signs the margin). role = sign(wm_margin - baseline), baseline = a
non-candidate control-hold (no grown edge). That becomes the `repair_target` role.

LOAD-BEARING on the OUTPUT (through the real production `repair_target`, 6 seeds, from the artifact): varying the
held WM referent (p_agent-pool vs p_patient-pool) FLIPS the repair role agent<->patient on 5/5 ambiguous items per
seed (`decision_flips_intact`=5 every seed); lesioning the cross-edge collapses the balanced margin to baseline ->
the content role stands, 0/5 flips (`decision_flips_lesioned`=0 every seed). So the WM state demonstrably CHANGES the turn's content (the
clarification the brain asks) and the change VANISHES on lesion — the "organs drive each other in the live brain"
milestone, not a hollow observer.

## Verification summary (6 seeds 42/43/44/100/101/102, numpy CPU; through the REAL production organ path)

| criterion | result |
|---|---|
| edge GROWS from W0=0.05 via in-brain self-supervised credit | **PASS** 6/6 (`grew_both`) |
| bounded by `stdp_w_max` (F3, no runaway) | **PASS** 6/6 (`bounded_F3`; max ~16.85 < 20) |
| grows on the per-seed RANDOM role edges (anti-cheat) | **PASS** 6/6 |
| CAVEAT: repair DECISION flips with the held referent | **PASS** 5/5 items/seed INTACT, 6/6 seeds |
| lesion-attributable (flip vanishes) | **PASS** 0/5 items/seed LESIONED, 6/6 seeds |
| byte-identical-off (`BRAIN_ONEBRAIN_XEDGE` unset) | **PASS** — standalone, `_wm_resolved_role`->(None,None), no `wm_resolved` field |
| moat | comprehended not flipped on well items (PART 1); the WM only touches the repair role on content-ambiguous items |

The FROZEN PART-1 edge closes the caveat by the SAME mechanism (5/5 flip / 0/5 lesion), so the caveat-close is
independent of how the edge was grown.

## What is built + guarded

`research/runners/onebrain_xedge_production.py`: `BRAIN_ONEBRAIN_XEDGE_LEARN` (default OFF, only with
`BRAIN_ONEBRAIN_XEDGE` on) routes the pool build to `grow_live_selfsupervised` (R3Pool at W0=0.05, gate open ->
grow from in-brain credit -> freeze) instead of the PART-1 frozen host-schedule grow. The caveat-close lives in
`comprehension_production_organ.repair_target` via `_wm_resolved_role`, which calls the pool-bound VALIDATED
`amb_read` (`shared.xedge_amb_read`) — reused, not reimplemented (a hand-rolled balanced read was NOT actually
balanced: baseline -1.2, wrong scale; fixed by delegating to the proven read). Every path is guarded by the
xedge-pool marker attrs -> byte-identical when shared=None / flag off / no referent held.

## Declared residuals (honest — the next rungs, not deferred)

1. **The training CURRICULUM is host-directed** (which discourse is presented, which WM referent is held while it
   resolves) — legitimate environment/teacher-scaffold territory (CLAUDE.md's brain-based-only boundary), the same
   class as R3-v3's teacher schedule and the AGENT_CUE/PATIENT_CUE currents. What became in-brain is the credit
   VALUE + DIRECTION (comprehension's own spiking verdict); the curriculum has not.
2. **Growth runs over a BUILD-TIME multi-turn curriculum, not yet during real per-turn chat.** The mechanism (an
   in-brain credit signal growing the edge over a multi-turn sequence) is demonstrated; wiring per-turn plasticity
   into the live `brain_reply` loop (open the candidate gate for one credited window per confident-comprehension
   turn, freeze between) is the named next rung.
3. **Semantic referent->pool binding is host-directed** (positional live focus) — carried unchanged from PART 1 /
   R3-v3's host-chosen candidate topology.
4. **Partial cross-growth**: the same-pool wrong-role edge grows to ~4-6 (vs the correct ~16.7); the correct edge
   dominates ~3x but the selectivity is not perfect.

## Verdict

**GO (6-seed, n_go 6/6):** the one-brain cross-edge now GROWS from the substrate's OWN activity (an in-brain,
self-supervised credit signal — comprehension's confident resolution), from W0=0.05, bounded (F3), following the
random role assignment; and the live-learned edge CLOSES the sub-decision caveat — it flips the real production repair-role
DECISION with the held WM referent, lesion-attributable, so the WM state genuinely changes the conversation. This is
the emergent goal: organs learning to drive each other in the live brain. Default-OFF (`BRAIN_ONEBRAIN_XEDGE` +
`BRAIN_ONEBRAIN_XEDGE_LEARN`), byte-identical-off, NO autonomous flip-to-default (owner-gated). Functional
read-outs only; no phenomenal-experience claim.
