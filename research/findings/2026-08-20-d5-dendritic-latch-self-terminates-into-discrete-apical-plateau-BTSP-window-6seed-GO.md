---
type: finding
status: contributing
date: 2026-08-20
mechanism: dendritic-plateau-coincidence-burst
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_d5_latch_self_termination_derisk.py — a genuine D5 EpisodicDapMemory store, its
  own persistent dendritic-dAP latch READ, with a runner-side Ecker-b spike/dAP-triggered adaptation current added to
  the D5 apical compartment (cp_v_apical); snapshot/restore harness so adapt-ON vs adapt-OFF differ ONLY by the gate;
  tools.verdict block + attributable_to + the completion/self-termination/specificity/lesion teeth.
runner: research/runners/_gap5_d5_latch_self_termination_derisk.py
external: NO-EXTERNAL-NEEDED — composes two in-repo GO mechanisms (D5's dendritic-plateau latch +
  [[swr-sequence-replay]]'s self-terminating envelope); the biology (a dendritic Ca-activated K / AHP / SK current
  self-terminating a plateau; BTSP's instructive signal IS the plateau, Bittner & Magee) is banked in
  research/biology/dendritic-plateau-coincidence-burst.md.
artifacts:
  - research/findings/raw/_d5_latch_selfterm/seed42.json
  - research/findings/raw/_d5_latch_selfterm/seed43.json
  - research/findings/raw/_d5_latch_selfterm/seed44.json
  - research/findings/raw/_d5_latch_selfterm/seed100.json
  - research/findings/raw/_d5_latch_selfterm/seed101.json
  - research/findings/raw/_d5_latch_selfterm/seed102.json
  - research/findings/raw/_d5_latch_selfterm/summary_6seed.json
---
# GO (6/6): D5's persistent dendritic latch SELF-TERMINATES into a discrete apical-plateau window — completion + specificity preserved — the bounded BTSP-eligibility window for learn-through-use

Artifact: research/findings/raw/_d5_latch_selfterm/summary_6seed.json (6/6 GO) · the per-seed
research/findings/raw/_d5_latch_selfterm/seed42.json + seed43.json + seed44.json + seed100.json + seed101.json + seed102.json.

**One line.** Step-2 of the D5 learn-through-use arc. Step-1 (NO-GO,
[[2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read]]) showed AdEx
soma recurrence can't complete a real ~14-25-cell D5 store, so D5's per-cell dendritic-dAP latch stays the READ — but
that latch is PERSISTENT (KIR-latched apical UP), and learn-through-use / BTSP needs a DISCRETE, self-terminating
window. This shows the persistent latch **CAN be made to self-terminate into a discrete apical-plateau transient
WITHOUT losing peak completion or specificity** — 6/6-seed GO, adversarially verified (4 lenses CONFIRMED + an
independent from-scratch rerun, no confound).

## The mechanism (NO `sim/` edit — a runner-side current on D5's OWN apical compartment)
A dendritic **Ca-activated K / AHP / SK-style adaptation current** on `bridge.cp_v_apical`: per-cell `w` increments by
`b_adapt` on every step the apical is in the UP/plateau state, decays with `tau_w`, and feeds back as
`dv_adapt = -(dt/tau_apical)*w`. It accumulates until it pushes the apical below the KIR bistable band → the plateau's
self-regenerating `sustain` sigmoid (`sim/kernels.py::fused_coincidence_plateau`) collapses → the latch drops to DOWN;
the cue is already gone, so it rests silent = a bounded transient. Attached exactly where the organ's own
`_reset_apical_latch` writes `cp_v_apical`; guarded by `if b_adapt > 0.0`, so **adapt-OFF is byte-identical to the
untouched persistent latch** (adapt-OFF IS the lesion control). Biology: the plateau's Ca2+/NMDA basis is
[[dendritic-plateau-coincidence-burst]]; the self-terminating transient is the [[swr-sequence-replay]] envelope.

## The 6-seed verdict (GO 6/6)
<!--derived-->
Same store, snapshot-restored (so ON vs OFF differ ONLY by the gate), dog adapt-ON (b_adapt=0.8, tau_w=150) vs
adapt-OFF (= the lesion): **completion PRESERVED** — dog-ON peak == dog-OFF peak (win_peak 0.857 seed42, 1.0 the other
five; the gate shortens DURATION, not peak). **SELF-TERMINATES** — ON term_ratio 0.0 (rests silent) vs OFF 1.0
(persistent, flat at max the full 500 ms window), transient up_ms 105-196. **SPECIFIC** — never-formed cat-ON ~0.0
(the gate only subtracts). **100% attributable to the gate** (`attributable_to` ON vs OFF-lesion = 1.0; the OFF lesion
reverts to the persistent latch). dog_size 14/24/15/25/24/25 across seeds — the mechanism holds in BOTH the small
(14-15) and larger (24-25) regimes.

## Honest scope — read before citing "reactivation" (the load-bearing caveats)
<!--derived-->
1. **It is a DENDRITIC APICAL-PLATEAU event, NOT somatic spikes.** `soma_peak` is 0-0.083 across all 6 seeds —
   essentially zero somatic output. The result is "SWR-like" ONLY in its bounded-transient ENVELOPE (ignite →
   co-activate → collapse → rest-silent over ~105-196 ms), NOT in the somatic population ripple-firing that DEFINES a
   biological SWR. The correct scope: an apical adaptation current converts D5's persistent KIR-latched apical-UP into
   a discrete, self-terminating **dendritic-plateau window** without reducing peak completion or specificity. This is
   biologically defensible AS the BTSP instructive signal precisely because BTSP's plasticity trigger IS the dendritic
   plateau / Ca2+ (Bittner & Magee), NOT the soma spike — so the step-3 write-back consumer must read the apical
   plateau, and this doc says so rather than implying somatic assembly reactivation.
2. **The gate's actuator and the read-out share `cp_v_apical`** — so a bare term_ratio→0 is guaranteed by
   construction. What UPGRADES it from "clamped read-out" to "genuine bistable termination" is the **rest-silent,
   no-re-ignition trace**: seed42 ON declines SMOOTHLY 0.857→0.714→0.571→0.143→0 over ~120 ms (a membrane trajectory
   crossing threshold cell-by-cell, not a step-clamp), then stays pinned at 0.0 for the remaining ~375 ms with NO
   re-ignition even though `w_adapt` (~100) is decaying with tau_w=150 ms. A clamp proportional to `w_adapt` would
   relax and re-ignite; permanent collapse to rest = a genuine KIR/self_regen bistable flip. That is the mechanism's
   real physiological content.
3. **seed43 cat_on=0.10** sits exactly at the absolute specificity threshold (a tighter CAT_MAX=0.09 flips this one
   seed). It is a PRE-EXISTING STORE-level false-positive (present in cat_OFF too, and in native D5 `recall('cat')`),
   gate-blameless; the ratio half of the specificity test passes ~10:1 (dog 1.0 ≫ cat 0.1).

## Adversarial verification + next
Four independent lenses (harness-manufacture, completion/read-legitimacy, causality/specificity, verdict/gate-
readiness) all returned CONFIRMED with no confound; the synthesizer independently reran the runner from scratch (a
build the 6-seed sweep never saw) and reproduced GO on all four teeth. Plasticity is frozen in the read bridge
(hebbian/stdp/structural/btsp/bdsp=False) so weight drift cannot confound. NEXT (step-3, the learn-through-use
CAPABILITY): drive BTSP potentiation gated ON this apical-plateau window and verify that USING a memory (recall →
window → write-back) STRENGTHENS it specifically + boundedly + monotonically with use — the bounded window is exactly
what prevents the runaway a persistent latch would cause. (Agent-built; parent verified the 6/6 + all four teeth +
sim/-untouched from the artifacts, ran the 4-lens adversarial workflow, and banked with the corrected scale + the
mandatory apical-vs-soma scope.)
