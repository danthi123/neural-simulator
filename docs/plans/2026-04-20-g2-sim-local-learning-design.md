---
type: plan
status: live
date: 2026-04-20
---

# Design: G2 — Sim-local plasticity bends the learning curve

**Date:** 2026-04-20 (same day as G1 GO; user asleep, autonomous continuation)
**Status:** Approved (operator pre-authorized autonomous progression)
**Scope:** Gate G2 — show that a plasticity rule *inside the sim* (not external gradient/logistic regression) demonstrably improves task performance across training epochs.

---

## 1. Context

G1 closed GO with 71.3% mean test accuracy using a 264-neuron reservoir and an **external** `sklearn.LogisticRegression` readout. The sim was a static feature extractor. That satisfies the G1 gate but defers the question: *does the simulator's own plasticity make the task easier?*

G2 answers that directly: same reservoir, same external readout, but STDP is turned on for the input→hidden projection. If STDP is producing better features, the learning curve *will bend upward epoch-over-epoch*. If it doesn't, STDP either isn't helping or is hurting — both are publishable findings.

## 2. Alternatives

### A. STDP on input→hidden (CHOSEN)
- Minimal change from v3.
- Reservoir (hidden→hidden) stays fixed — keeps the feature-extraction baseline unchanged.
- STDP on ~1920 input→hidden synapses.
- Measurement: retrain LogReg each epoch and report test accuracy.

### B. R-STDP with reward channel
- Rewarding-correct / punishing-wrong on each example via `current_reward_signal`.
- Requires a frozen-then-updated LogReg to generate the reward signal.
- More complex, and the sim's eligibility trace is unsigned (noted in G1.v1 findings), which makes signed-reward learning less clean.

### C. Structural plasticity
- Let synapse growth/pruning happen during training.
- Harder to isolate a clean learning curve; a good G3/G4 lever.

**Selected A** because it's the most direct test of "does STDP make features better?" and the smallest delta from the v3 baseline.

## 3. Architecture

### 3.1 New capability: per-synapse plastic mask

The sim's STDP kernel currently applies to all synapses with fired neurons. For G2 we need to keep hidden→hidden frozen (they're the reservoir; we don't want the reservoir's dynamics to drift) while STDP fires on input→hidden.

Add a `cp_synapse_plastic_mask` bool array of shape `(nnz,)` to `SimulationBridge`:
- `None` by default → behaves as if all synapses are plastic (back-compat; existing experiments unaffected).
- Non-`None` → STDP only writes back weight updates where the mask is True.

Implementation:
- `SimulationBridge.__init__` sets `self.cp_synapse_plastic_mask = None`.
- `inject_explicit_wiring(plan, output_inhibitory_indices=None)` now also builds the mask from each population's `plastic` flag and stores it.
- STDP update path in `_run_one_simulation_step` gates its weight write on the mask.

Existing experiments don't set this mask → no regression.

### 3.2 G2 runner

New module `research/runners/g2_runner.py`, closely parallel to `g1_v3_runner.py`:

- Same 264-neuron reservoir topology and per-pulse stimulus calibration.
- Plastic input→hidden (STDP on), fixed hidden→hidden.
- `CoreSimConfig.enable_stdp = True`, symmetric `stdp_a_plus = stdp_a_minus = 0.008` (lower than defaults, no LTP bias, to avoid the runaway potentiation we saw in G1.v1).
- `stdp_w_max = 2.0`, `stdp_w_min = 0.0`.

Per-epoch loop:
1. Extract train features (activate STDP during presentation).
2. Extract test features (STDP still on — we're measuring what the sim naturally does).
3. Train fresh LogReg on train features; report train/test accuracy.
4. Log per-epoch: train_acc, test_acc, weight stats, mean hidden rate.

Run 8 epochs per seed. Compare epoch 0 vs epoch 7 test accuracy across 3 seeds.

### 3.3 Success criteria

**GO:**
- Mean(epoch_7_test_acc) > Mean(epoch_0_test_acc) + 5 percentage points across 3 seeds, AND
- The increase is directional: epoch_7 ≥ epoch_0 in at least 2 of 3 seeds, AND
- Test accuracy at epoch 7 stays above the G1 baseline of ~55% (so STDP isn't catastrophically destroying features).

**NO-GO:**
- Mean test accuracy decreases across epochs → STDP is hurting features.
- Test accuracy at epoch 7 < 40% (below or near chance): STDP is destroying the reservoir's information content.

**PARTIAL:**
- Flat curve (change < 3 pp). STDP isn't changing anything meaningful. Document and try G2.b (R-STDP) or G2.c (structural).

## 4. Risks

| Risk | Mitigation |
|------|------------|
| STDP saturates weights uniformly (G1.v1 pathology) | Symmetric A+/A−; w_max = 2.0 cap; monitor weight histogram per epoch |
| Features become less class-discriminative | Test accuracy is the ground truth; if it drops we NO-GO |
| Runtime explosion from STDP per step | STDP is already benchmarked in the sim; 264 neurons × ~10K synapses ≈ fast |
| Plastic-mask implementation bug | Add a targeted unit test asserting only masked synapses update |

## 5. Deliverables

- `sim/bridge.py`: `cp_synapse_plastic_mask` support in STDP path + `inject_explicit_wiring` builds the mask.
- `tests/test_plastic_mask.py`: unit test — inject wiring with some plastic=False; run STDP-triggering step; assert those weights unchanged.
- `research/runners/g2_runner.py`: runner + multi-epoch loop.
- `tests/test_g2_runner_smoke.py`: smoke test.
- `research/findings/2026-04-20-g2.md`: results + verdict.
- CHANGELOG update.

## 6. Branch decision

G2 uses only existing simulator primitives (STDP, explicit wiring, reservoir). No departure from the biological-science foundation. **Stays on `main`.**

If a later gate (e.g., G5 sensorimotor) requires something the core sim genuinely can't do (multi-compartment neurons, custom learning rule kernels), that work goes on a branch.
