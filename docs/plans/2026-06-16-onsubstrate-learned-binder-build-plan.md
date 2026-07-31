---
type: plan
status: live
date: 2026-06-16
---

# On-substrate learned binder — incremental build plan (2026-06-16, CYCLE 102)

**Goal:** realize the de-risked LEARNED role-filler binder (additive bind + linear unbind, ON/OFF rate coding,
trained by a brain-faithful local rule) as actual spiking neurons/synapses on the `SimulationBridge` — the
genuine "step 3," replacing the fixed vector-symbolic-algebra bind.

## What the cheap-first arc established (numpy, the design these steps realize)
- The bind is **additive**: `bound = nonlinearity(role·W_R + filler·W_F)`; unbind is linear (`concat(bound,
  role·W_RP)·W_U`). Generalizes systematically on the stream codes (held-out 0.889 vs floor 0.0).
- **Read noise** (finite populations) doesn't break it (held-out flat to n_per≈11).
- The nonlinearity must be **ON/OFF opponency** (a signed value → two non-negative rate channels): a single
  non-negative rate collapses it (0.083); ON/OFF restores it (0.806, beats the signed tanh baseline).
- The learning rule can be **brain-faithful and local**: feedback alignment (a fixed random feedback matrix,
  no weight transport) matches/beats exact backprop (seed-42 0.917 vs 0.833) — i.e. local plasticity + a
  broadcast teaching signal, no backprop. [#2c finalizing; bundled-facts finalizing.]

## The architecture on the bridge
Every signed quantity is carried by an ON/OFF population pair (the substrate's standard, used in the NEF
cleanup / FHRR / biologization). Populations (rate-coded; the project's population code carries graded values
at ~94% fidelity, CYCLE 91):
- `role_on/role_off`, `filler_on/filler_off` (D_in each) — inputs; drive = relu(±code)·scale.
- `bind_pos/bind_neg` (D_h each) — the bind hidden layer. `bind_pos` current = role·W_R + filler·W_F (wired:
  role_on→+W_R, role_off→−W_R, filler_on→+W_F, filler_off→−W_F); fires ∝ relu(h) = the ON channel. `bind_neg`
  wired with negated weights → fires ∝ relu(−h) = the OFF channel. (bound = [bind_pos, bind_neg] rates.)
- `roleh_on/roleh_off` (D_h each) — role·W_RP for the unbind path.
- `out_pos/out_neg` (D_in each) — the unbind readout; current = concat(bind_pos,bind_neg,roleh)·W_U; the
  estimate = out_pos − out_neg rates → cleanup vs the codebook (the validated spiking NEF cleanup).

## Incremental de-risk sequence (each a GATE; stop/localize on a NEGATIVE)
1. **Forward, fixed weights (GPU).** Inject the numpy-trained W_R/W_F/W_RP/W_U as fixed pathways via
   `inject_explicit_wiring`. Drive role/filler ON/OFF, run, read `out_pos−out_neg`, cleanup → recall. GATE:
   on-bridge recall ~ the numpy binder (the spiking forward preserves the bind). Risk: rate-regime tuning
   (drive scale so neurons are in the ~linear rate band, not silent/saturated) — reuse the stream-cortex
   population-read calibration.
2. **Bundled forward (GPU).** Same, but drive 3 superposed facts (agent+verb+object). GATE: who/what recall ~
   the numpy bundled result. (Gated on bundled-facts numpy GO.)
3. **Local-rule training (GPU) — the novel core.** Replace fixed weights with plastic pathways; add a teaching
   signal (the target filler driven as a population) + an error population (target − out) + a FIXED RANDOM
   feedback pathway (error → bind hidden, feedback alignment) + eligibility-gated three-factor plasticity (the
   bridge has eligibility traces + three-factor). Train on single pairs; GATE: held-out systematicity >> floor
   (the substrate LEARNS to bind via local plasticity). This is the project's credit-assignment frontier; #2c
   de-risks the principle (per-output error broadcast, not a scalar reward — distinct from the documented
   global-scalar-feedback-fails wall).
4. **Integrate (GPU).** The on-bridge learned binder + the (already-biologized) cleanup + no-confab moat on the
   stream-cortex codes → the full who/what conversation with a LEARNED, on-substrate bind.

## Anti-cheats (carry the numpy protocol's four onto the bridge)
Leakage-free train/held-out splits; shuffled-label control (→ chance); memorization-floor (lookup table);
lesion the learned weights (→ collapse). Plus: the teaching signal must be ABSENT at test (no target leak);
report the spiking vs numpy gap honestly.

## Reuse (file pointers)
`inject_explicit_wiring` (bridge.py:2273), `set_pathway_weights` (:2866); the ON/OFF read + NEF cleanup
(`research/findings/raw/_spiking_cleanup_nef.py`); eligibility/three-factor (bridge.py ~6745); the population
read calibration (the stream-cortex runners); `OnOffRateBinder` / `FeedbackAlignmentBinder` (the numpy
reference for the weights + the expected held-out). Build CuPy/GPU; numpy tiny-smoke only.

Start at step 1 once the two CPU de-risks land. Keep each step a committed, anti-cheated GATE.
