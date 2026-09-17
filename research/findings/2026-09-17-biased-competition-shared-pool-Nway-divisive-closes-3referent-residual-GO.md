---
type: finding
status: verified
date: 2026-09-17
mechanism: biased-competition N-way competition topology — a single SHARED inhibitory pool (star, not complete graph)
  driven by ALL sel accumulators and inhibiting all of them (competition_mode="shared_pool"), so pooled suppression
  self-scales with the live competitor count; replaces the pairwise sel_FS_X->sel_Y!=X cross-inhibition that could not
  hold 2-ref and 3-ref on one fs_to_sel scalar
integration_faculty: selective-attention-biased-competition
lane: language + focus (scaffold-retirement follow-on)
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO. The pairwise cross-inhibition (fs_to_sel_weight=7.0, the stable basin centre from the 2026-09-16 graded-bias
  fix) left the de-risk's 3-REFERENT scale probe at 4/6 while the 2-ref GO-arm was 6/6 — a documented residual (pairwise
  suppression does not scale to 3+ rivals on one scalar). The shared-pool divisive topology (Wang 2002 / Wong-Wang 2006
  shared inhibitory pool; Carandini-Heeger divisive normalization) reaches the runner's FULL GO bar SIMULTANEOUSLY at
  6 seeds: go_arm 6/6, three_ref 6/6, lesion-breaks 6/6, moat-intact 6/6, seed100-roll-closes 6/6. Additive + DEFAULT
  "pairwise" (byte-identical-off, SHA256-proven on cp_neuron_firing_thresholds + synapse pre/post/weights). This closes
  the 3-referent residual. Landed default-off; the production FLIP (shared_pool as the default WTA) is a follow-on that
  must first re-verify the 2-ref production tests + the integrated /api/brain-chat battery under shared_pool.
runner: research/runners/_phaseB_biased_competition_graded_derisk.py (--competition-mode shared_pool)
artifacts:
  - research/findings/raw/_biascomp_3rival_6seed.json
external: Wang 2002 (NMDA recurrent decision network, shared inhibitory pool); Wong & Wang 2006 (J Neurosci, the
  recurrent-attractor WTA — recorded in the lane external log with its DOI); Carandini & Heeger 2012 (divisive
  normalization) — the pooled-FSI motif generalized to N channels. NO-EXTERNAL-NEEDED beyond these (a topology change
  on an already-grounded WTA).
builds_on:
  - research/findings/2026-09-16-graded-bias-regression-FIXED-designed-cross-inhibition-restored-fs7.md
---

# Biased-competition shared-pool N-way divisive inhibition — closes the 3-referent residual (GO 6/6)

The 2026-09-16 graded-bias fix restored the 2-referent GO-arm to 6/6 (fs_to_sel_weight 5.0->7.0) but left an honest
residual: the de-risk's 3-REFERENT scale probe stayed at 4/6, because the pairwise cross-inhibition (each FS pool
inhibits every other referent) cannot hold both the 2-ref pair case and the 3-ref case on one fs_to_sel scalar — a
saturated biased winner (~0.5 sel-pool ceiling) cannot pull the 1.3x margin over a strong rival resisting pairwise
suppression at N=3.

## The mechanism (additive, default "pairwise" = byte-identical)

`competition_mode="shared_pool"` is a genuinely different topology (a star, not a complete graph): ONE common
inhibitory population driven by ALL sel accumulators and inhibiting all of them. The pooled suppression SELF-SCALES
with the live competitor count — an added rival raises the common inhibition it itself receives — while the biased
winner's large content-bias current keeps it above the pool. This is the shared-inhibitory-pool / divisive-normalization
motif (Wang 2002; Carandini-Heeger) applied to the disjoint per-referent accumulators.

## Result (from research/findings/raw/_biascomp_3rival_6seed.json)

All at 6 seeds, `competition_mode: shared_pool`: `go_arm_seeds` 6, `three_ref_seeds` 6, `lesion_breaks_seeds` 6,
`moat_intact_seeds` 6, `seed100_roll_closes_seeds` 6 — the runner's FULL GO bar (go>=6 AND lesion==n AND moat==n AND
three_seeds==n) is met. Byte-identical-off proven (SHA256 of cp_neuron_firing_thresholds + synapse pre/post/weights
matches git HEAD at the "pairwise" default).

## Scope + follow-on

Landed on main default-off (commit fae67ca7d). The 3-referent residual documented in the fs7 finding is now closed at
the de-risk level. The production FLIP — making shared_pool the DEFAULT WTA in `MultiTurnAgent` — is a reviewed
follow-on: it changes the WTA topology for every biased-competition turn, so it must first re-verify the 2-ref
production tests (test_multireferent_biased_competition / graded / byte-identity) AND the integrated /api/brain-chat
no-regression battery under shared_pool (GPU/local + AWS, post-gaming). A wall defers a METHOD, not the capability —
here the capability (N-referent selective attention) is now de-risked GO.
