# Full FHRR-on-bridge feature — plan + layer-(a) complex-synapse-bind de-risk design — 2026-06-05

> **For Claude:** owner-FUNDED + greenlit "proceed with the full FHRR-on-bridge feature now" (after the RF-on-bridge
> de-risk GO, `2026-06-05-rf-on-bridge-derisk-GO.md`). De-risk-driven, TDD, controller-verified protected diffs,
> flagged. Protected `sim/` edits in scope for the RF/FHRR substrate ONLY; frozen bars / no-confab moat never weakened.

**Goal:** the production conversational composer computes FHRR composition on the SimulationBridge's resonate-and-fire
(RF) phasor neurons — clearing the opponency rate-coded SNR wall for real (the opponency does not exist in the phasor
algebra) and gaining the F=3 two-attribute resonator the ±1 Hadamard scheme provably cannot do.

**Status:** the RF substrate (RESONATE_AND_FIRE model: rotate Z=re+i·im, Im zero-crossing spike = phase) + phase
readout + bind/unbind/bundle via external `rf_kick` + the full composer task at parity = DONE (de-risk GO). What
remains is to make the bind/unbind happen THROUGH the bridge (not external injection) and to recode the production
composer onto it.

## Three layers (de-risk-driven; each gated GO before the next)
- **(a) complex-synapse bind [NEXT — designed below].** The bind/unbind happen through synapses carrying the operand
  phasor, not external `rf_kick`. A continuous complex-state recurrent path for RF neurons.
- **(b) recode the production composer.** Replace the rate-coded ±1 Hadamard bind/unbind in
  `research/runners/core_sim_composition.py` (+ the parser/dialogue in `brain_conversational_agent.py`) with the RF
  phasor bind/unbind through complex synapses. Concept codes become phasor vectors (random phases); cleanup becomes
  phase-cosine similarity (already the FHRR reference's cleanup).
- **(c) re-validate the full capability matrix** (who/what/abstain/negation/clauses/dialogue) on the RF composer at
  parity with the current rate-coded bars — the GATE for declaring the opponency cleared. A regression here is a
  reportable finding (the measured cost), not hidden.

## Layer (a) design — complex-synapse bind (the next de-risk)
**Mechanism (Frady-Sommer 2019, the resonate_fire_fhrr.py reference's conceptual basis):** a resonate-and-fire
*network* computes with continuous complex states + complex weights; spikes are the phase readout. Synaptic input to
neuron i is the complex matvec `u_i = Σ_j W_ij·z_j`; the bind `phasor_a·phasor_b` is `a` passing through a synapse
whose complex weight is `b` (complex multiply = magnitude product + phase sum). The current de-risk computed
`a·b` in numpy and injected it via `rf_kick`; layer (a) computes it through a complex synapse on the bridge.

**On-bridge realization (the protected extension):**
- RF neurons hold complex state `z = re + i·im` (already: `v=re`, `u=im`).
- A complex weight set for RF connections: two real CSR matrices `W_re`, `W_im` (or reuse `cp_connections` for the
  real part + a parallel `cp_rf_w_im`). Lazily allocated; only for RESONATE_AND_FIRE bridges.
- In the RF step branch, add the complex synaptic input to the rotated state:
  `u_re = W_re@re − W_im@im;  u_im = W_re@im + W_im@re;  re += u_re;  im += u_im` (complex matvec via two real
  sparse matvecs on the backend). The rotation + zero-crossing readout are unchanged.
- An `rf_set_complex_weights(pre, post, phasor)` helper to install a complex synapse (weight = the operand phasor).

**Why continuous-state, not spike-driven:** FHRR phase arithmetic is in the *complex states*, not spike events; the
zero-crossing spike is the readout. So the RF synaptic path reads presynaptic *states* `z_j` (re/im), not
`cp_prev_firing_states` — distinct from the rate-coded synapses (untouched).

### Layer (a) de-risk TDD gates (`tests/test_rf_complex_synapse.py` + `_rf_complex_synapse_probe.py`)
1. **Single complex synapse = bind:** pre RF neuron state = phasor `a`; one complex synapse weight = phasor `b`;
   run; the post RF neuron spikes at phase `a+b` (circular err < 0.03). (unbind: weight = `conj(b)` → `a−b`.)
2. **Bundle through synapses:** several pre phasors → one post via unit complex synapses → post resonates the sum.
3. **Composer task through synapses (the de-risk GATE):** the loads-2/3/5 task with bind/unbind realized via
   `rf_set_complex_weights` (no `rf_kick` of precomputed products) → accuracy ≥ 0.80 + abstention, parity with the
   reference. GO → layer (b). NEGATIVE → report the on-bridge complex-synapse wall (reopen with owner).

**Honest risk:** the continuous complex recurrent dynamics may need stability tuning (the `Σ W z` term can grow/decay
across the resonate window; the reference's single-neuron `rf_resonate` sidesteps this by injecting the kick once).
The de-risk's job is to find whether a stable on-bridge complex-synapse bind exists at parity. If the recurrent term
destabilizes the phase readout, fall back to a spike-time-triggered complex kick (deliver `spike_phasor·W` at the
presynaptic spike) — a second realization to de-risk before declaring NEGATIVE.

## Sequencing / scope
Layer (a) is the next concrete arc (de-risk → if GO, the layer-(a) integration). Layers (b)/(c) are scoped after
(a) GO (their design depends on (a)'s realization). Each protected diff is minimal, guarded, flagged, controller-
verified, both-remotes. The current rate-coded composer stays the production path until (c) re-validates at parity —
no capability regression ships silently.
