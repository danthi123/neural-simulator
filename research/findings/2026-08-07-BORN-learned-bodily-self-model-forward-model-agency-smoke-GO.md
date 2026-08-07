---
type: finding
status: live
date: 2026-08-07
mechanism: learned-bodily-self-model-forward-model-agency
lane: A-affect
---

# Lane A · SELF-SCHEMA (BORN adoption): a LEARNED bodily self-model — a Hebbian/Oja forward model + a neural reafference-cancellation comparator produce a self/other AGENCY signal, PARENT-VERIFIED 6-seed GO (2026-08-07)

BORN (landscape-survey CLOSEST-1) adopted onto the self-schema lane: a LEARNED forward model (efference -> predicted
sensory feedback) plus a NEURAL comparator that emits a self/other agency signal from PREDICTED-vs-ACTUAL feedback —
the "mirror-test" correlate (recognise self-generated, predictable sensory consequences vs externally-generated,
unpredicted ones). Runner: `research/runners/_born_learned_self_model_derisk.py`; artifact:
`research/findings/raw/_born_learned_self_model_smoke.json`. This is a 1-seed SMOKE (spot-checked identical on seeds
42/43/100); the parent runs the 6-seed validation before any generalization claim.

## Re-anchor — what our self-schema foundation ALREADY computed, and the genuine un-built step
Our lane already had (i) the DR-3 SELF-SCHEMA region (2026-07-23 GO): reads its own attention/confidence/authorship
internal state; and (ii) the AGENCY/AUTHORSHIP tag (2026-08-01 GO): a FIXED-structure corollary-discharge comparator
answering "was an efference-copy MARKER present?". BOTH lack a LEARNED forward model — the authorship-tag's own honest
scope names "self-organized/learned wiring" as the un-built follow-on. The genuine BORN step, built here: the forward
model is LEARNED (action_k -> predicted_k grown by co-fire, NOT hand-wired), and agency = "does the ACTUAL feedback
match what my forward model PREDICTED this action would cause?" — NOT "was a marker present?". This is DECISIVE on the
DECOUPLED case (efference present but the world's feedback is wrong): a presence/marker detector says SELF (wrong);
the forward-model comparator says LOW agency (right) — the Blakemore-Frith perturbed-reafference result.

## Mechanism (brain-based; reuse-by-import; NO `sim/` edit)
ONE numpy Izhikevich `SimulationBridge`, 21 regions (K=4 identities x {action, predicted, sensory, resp, rinh} + a
shared predicted-layer inhibitor). Body/world (legit host): the action (efference) the body emits + the actual sensory
feedback the world returns are external drive. Everything between is neurons/synapses.
- FORWARD MODEL: `action_i -> predicted_j` all-to-all, Hebbian-plastic (zero init), Oja's rule
  (`cfg.hebbian_oja=1.0`). Trained by co-firing the efference `action_k` with a reafferent teacher on `predicted_k`;
  Oja's INPUT-DEPENDENT fixed point (`w* = <a x>/<a^2>`) grows the SELECTIVE diagonal and leaves the off-diagonal near
  zero (measured diag 27.9 / off 0.01). Frozen before test.
- COMPARATOR = REAFFERENCE CANCELLATION (Blakemore-Wolpert-Frith; the "can't tickle yourself" circuit, FIXED
  structure — the banked scope, as the affect/authorship GOs banked a hand-wired attractor): `sensory_k -> resp_k`
  excites the sensory response; `predicted_k -> rinh_k -> resp_k` (gaba) cancels it. A matched prediction attenuates
  the self-caused reafference. Agency read-out = LOW residual `resp` population rate (NOT a host abs(pred-actual)).
- PREDICTED-LAYER LATERAL INHIBITION (competitive normalization, a companion process real cortex runs): a shared
  `pwta` pool so only the strongly-driven diagonal prediction fires — this kills the off-diagonal FM leak that
  otherwise partially cancels the decoupled response (see build note below).

## Conditions + result (1-seed smoke; means over the balanced identity x condition x REPS trials)
<!--derived from research/findings/raw/_born_learned_self_model_smoke.json (`means` + `per_seed`)-->
| metric | value | reads as |
|---|---|---|
| FM diagonal / off-diagonal | 27.9 / 0.01 | forward model learned the SELECTIVE action_k->predicted_k mapping |
| resp SELF / EXTERNAL / DECOUPLED | 0.018 / 0.047 / 0.047 | prediction cancels the self reafference; decoupled un-cancelled (no leak) |
| agency AUC (self vs not-self) | 1.000 | discriminates self-caused from not-self-caused feedback (chance 0.5) |
| self-vs-DECOUPLED AUC | 1.000 | efference present but wrong feedback -> LOW agency |
| PRESENCE detector self-vs-decoupled | 0.562 | our existing foundation CANNOT tell them apart |
| contingency self-vs-EXTERNAL AUC | 1.000 | identical content, self vs external -> different agency |
| RANDOM forward model, self-vs-decoupled | 0.50-0.60 | learning REQUIRED (see anti-cheat b) |

- SELF (action_k + sensory_k): predicted_k cancels resp_k -> resp 0.018 -> HIGH agency.
- EXTERNAL (sensory_k only): no prediction -> resp 0.047 -> LOW agency.
- DECOUPLED (action_k + sensory_j, j!=k): predicted_k cancels the empty resp_k; resp_j un-cancelled -> resp 0.047 ->
  LOW agency DESPITE the efference being present.

## Anti-cheats — all outcomes
- (a) SELF-vs-OTHER CONTINGENCY: HOLDS. Identical sensory content (sensory_k) reads HIGH agency as SELF (action_k
  present) and LOW as EXTERNAL (no action) — self-vs-external AUC 1.000 — so it is a self-model, not a sensory
  detector. And DECOUPLED reads LOW despite the efference being present — agency requires prediction-match, not
  action-presence.
- (b) LEARNING-REQUIRED: HOLDS. A RANDOM all-to-all forward model predicts EVERY identity -> cancels indiscriminately
  -> it behaves like the presence detector: it can nail external (why its POOLED auc is ~0.75, not chance) but on the
  DECISIVE self-vs-decoupled axis it collapses to 0.50-0.60 while the learned model is 1.000. So the identity-specific
  (decoupled) agency is carried by the LEARNED synaptic mapping, not the fixed comparator structure.
- (c) NEURAL COMPARATOR: HOLDS. Agency = residual `resp` population spike rate from a spiking cancellation circuit,
  NOT a host abs(predicted-actual) formula.
- (d) 6-SEED: ✅ VERIFIED (parent, seeds 42/43/44/100/101/102) — see the PARENT-VERIFIED section below.
- (e) MAPPING-SPECIFIC (parent-added anti-cheat): HOLDS. A PERMUTED-SELECTIVE forward model (a derangement
  `action_i -> predicted_{perm[i]}`, perm[i]!=i, at the learned diagonal magnitude — SELECTIVE but MIS-mapped)
  ALSO fails self-vs-decoupled (mean AUC 0.339, below chance) while the correctly-learned model is 1.000. So it is
  the CORRECT learned mapping, not merely any selective forward structure + the cancellation/lateral-inhibition
  machinery, that carries the agency signal.

## Head-to-head vs the existing foundation
The learned forward-model comparator does self-vs-DECOUPLED at AUC 1.000; a PRESENCE/marker detector (agency := was
efference present?) — the mechanism our existing authorship-tag foundation implements — scores 0.562 (chance), because
SELF and DECOUPLED both have the efference present. The LEARNED forward model adds exactly the agency-discrimination
the foundation lacked.

## Build notes (measured, load-bearing)
- The comparator is CANCELLATION, not a coincidence AND-gate, because of a substrate property: the synaptically-driven
  `predicted` pool fires SYNCHRONOUS volleys (which a fast FS interneuron reads cleanly) while the tonically-driven
  `sensory` pool fires ASYNCHRONOUSLY -> a symmetric summation AND-gate saturated on the predicted volley alone (a
  single input drove `match` supra-threshold regardless of weight).
- Oja's rule was necessary: the plain rate-window Hebbian rule's fixed point is `w* = hebbian_max_weight`, INPUT-
  INDEPENDENT (config.py:408), so the off-diagonal drifts to the cap and there is no selectivity (measured diag 44.6 /
  off 31.8); Oja's input-dependent fixed point gave diag 30 / off 4.5, then the lateral-inhibition companion process
  drove the off-diagonal leak to ~0.01.
- The predicted-layer LATERAL INHIBITION was the decisive fix for the DECOUPLED axis: without it the weak off-diagonal
  FM made `predicted_j` fire enough to partially cancel the decoupled response (resp_dec 0.036 < external 0.041,
  self-vs-dec AUC ~0.80); with competitive normalization only the diagonal prediction fires, decoupled = external
  (0.047), self-vs-dec AUC -> 1.000.
- Post-inhibitory rebound: strong gaba + a long test window accumulates rebound spikes in `resp` (self resp INVERTED
  to 0.096 at a 90-step window) -> the window is kept moderate (40 steps).

## Honest scope (the deliverable's boundary)
- The FORWARD MODEL is LEARNED by synapses (Hebbian/Oja co-fire). The COMPARATOR is FIXED-structure (hand-wired
  reafference-cancellation), the banked scope. Named follow-ons: a self-organized comparator; a reward-gated
  (three-factor) forward model; the reafferent teacher replaced by a real body/sensor loop.
- numpy is the CPU backend (real spiking Izhikevich), not a host shortcut. STDP is off because its eligibility never
  applies without reward on this bridge (weight stays 0, per the `_D_sparse_heteroassoc` note) — the co-fire learning
  is Hebbian/Oja.

## ✅ PARENT-VERIFIED (6-seed) + the added mapping-specific control
<!--derived from research/findings/raw/_born_learned_self_model_6seed.json (`per_seed` + `means`)-->
The parent ran the full 6-seed validation (42/43/44/100/101/102, numpy-CPU, 48.6s) AND added a decisive third
anti-cheat (the permuted-selective control, arm (e) above). **VERDICT: GO (6-seed).** Every seed is identical to
the smoke:

| metric (mean over 6 seeds) | value | reads as |
|---|---|---|
| FM diagonal / off-diagonal | 27.5 / 0.01 (~2900x) | forward model learned the SELECTIVE mapping, all 6 seeds |
| resp SELF / EXTERNAL / DECOUPLED | 0.018 / 0.047 / 0.047 | prediction cancels the self reafference; decoupled un-cancelled |
| agency AUC (self vs not-self) | 1.000 (6/6) | discriminates self-caused from not-self-caused |
| self-vs-DECOUPLED AUC | 1.000 (6/6) | efference present but wrong feedback -> LOW agency |
| PRESENCE detector self-vs-decoupled | ~0.52 (0.500-0.562) | the existing foundation is at CHANCE on this axis |
| contingency self-vs-EXTERNAL AUC | 1.000 (6/6) | identical content, self vs external -> different agency |
| RANDOM forward model, self-vs-decoupled | 0.518 | learning REQUIRED (collapses to chance on the decisive axis) |
| **PERMUTED-selective, self-vs-decoupled** | **0.339** | **mapping-specific: a mis-mapped selective FM ALSO fails** |

**Honest caveat on the `attributable_to` ⛔ flags.** The random-FM and presence-detector controls sit AT CHANCE
on self-vs-decoupled (0.518 / 0.528); because `attributable_to` compares raw AUCs against 0 (not against the 0.5
chance level), it reports ~52% as "present in the control" and prints a ⛔. That is an artifact of feeding an
AUC (chance=0.5) to a helper that assumes chance=0 — the controls are genuinely at chance (they FAIL), which is
the intended result. The PERMUTED control (0.339, BELOW chance) has no such ambiguity: it attributes 66% cleanly.
All three controls fail the decisive axis while the learned model is 1.000, so the agency discrimination is
attributable to the CORRECT LEARNED MAPPING. (A tighter future read would pass AUC-0.5 to `attributable_to`.)

**⇒ the SECOND concrete adoption win from the 2026-08-07 landscape survey (BORN, CLOSEST-1):** a LEARNED bodily
self-model with a mirror-test agency correlate, on the self-schema lane — the learned forward model adds the
agency-discrimination (self vs perturbed-reafference) that our fixed presence/marker foundation lacked, 6-seed,
brain-based, with three anti-cheats (contingency, learning-required, mapping-specific) all holding. Named
follow-ons (unchanged): a self-organized comparator; a reward-gated (three-factor) forward model; the reafferent
teacher replaced by a real body/sensor loop.

## Run
```
SIM_BACKEND=numpy python -u -m research.runners._born_learned_self_model_derisk --smoke                       # 1-seed
SIM_BACKEND=numpy python -u -m research.runners._born_learned_self_model_derisk --seeds 42 43 44 100 101 102  # 6-seed
```
