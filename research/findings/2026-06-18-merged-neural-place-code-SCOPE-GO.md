# Merged nav-critic self-org place-code afferent — SCOPE-GO (composition), δ-probe pending

**Date:** 2026-06-18
**Roadmap:** TRUE-ONE-BRAIN spike-ification item #5
**Status:** SCOPE-GO (composition + moat validated, numpy CPU). The short δ-lift GPU probe is the
follow-on (see "What's next").

## Goal (item #5)

Replace the merged "one brain" nav critic's HOST-Gaussian `vs_place_context` position afferent with the
self-organized spiking `place` code (`neural_place_selforg`). **Dual value:**
- (a) **Breadth win** — retires a host shortcut: the critic's position code becomes self-organized spiking
  place cells (`place_sensors → place` threshold-WTA + `place_fs` FS-PING), the egocentric landmark sensors
  the only host entry point.
- (b) **Possible δ-lift** — the self-org `place→striosome_value` pathway is a `coincidence_detector` fired by
  the FS-PING gamma volley. It fires the MSN-D1 critic from the LEARNED code WITHOUT the position-blind
  up-state bootstrap that capped the CYCLE-211/214 value-train δ at ~1.3 (the `enable_convergent_upstate` A1
  floor). So it *could* lift the r−V gap above that cap.

An honest NEGATIVE (self-org place doesn't compose, doesn't lift δ, or needs the dendritic substrate) is a
valid deliverable.

## What was built (Step 1 — the builder edit)

`research/runners/nav_conv_merged_bridge.py`, additive + default-off (byte-identical when off):

1. New kwarg `nav_critic_place_selforg: bool = False` on `build_merged_nav_conv_bridge`. When True (and
   `co_resident_nav_critic=True`), it forwards `neural_place_selforg=True` into `build_bg_brain_regions`,
   switching the critic to the `if enable_neural_critic and neural_place_selforg:` branch
   (`g11_bg_runner.py:1175` regions / `:1783` pathways) — the self-org place pool + the plastic coincidence
   `place→striosome_value` pathway — instead of the host-Gaussian `vs_place_context`
   (`g11_bg_runner.py:1841` `elif enable_neural_critic:`).
2. **Mutual-exclusivity assert** against `nav_critic_convergent_upstate`: the up-state arm is
   `vs_place_context`-specific (the self-org branch has no `vs_place_drive` region/pathway), AND
   `g11_bg_runner.py:3853` HARD-GATES `enable_convergent_upstate` OFF whenever `neural_place_selforg` is on
   (the position-blind A1 floor caps grading). Co-requesting both is a config error → loud assert.
3. Threaded the same kwarg through the `MergedNavConvAgent` constructor so the agent-level moat check can
   build the self-org critic.

Diff: 28 insertions / 2 deletions (the 2 deletions are the two signature lines extended with the trailing
kwarg). The `else`/non-self-org path and every existing default are byte-unchanged.

## Composition smoke (numpy CPU) — PASS

`build_merged_nav_conv_bridge(seed=42, co_resident_nav_critic=True, nav_critic_place_selforg=True)` builds.
47 regions on ONE `SimulationBridge`. Verified:

| Check | Result |
|---|---|
| `place_sensors` region present | ✅ (60 neurons) |
| `place` region present | ✅ (200 neurons, self-org hippocampal pool) |
| `place_fs` region present | ✅ (24 neurons, FS-PING) |
| `striosome_value` region present | ✅ (80 neurons, MSN-D1 critic) |
| `snc` + `reward_us` present | ✅ |
| host-Gaussian `vs_place_context` **ABSENT** (replaced) | ✅ |
| conv slices present (`parse_conj`, `parse_role`, `dlpfc_wm`, `cortex_ctx`) | ✅ |
| `place` array-disjoint from `parse_role` / `dlpfc_wm` | ✅ |
| `striosome_value` array-disjoint from `parse_role` | ✅ |
| plastic `place→striosome_value` pathway present | ✅ |

## Moat check (numpy CPU, via `MergedNavConvAgent`) — PASS

With the self-org place critic co-resident:
- `hear('dog go north')` → `what_does('dog','go')` = `'north'` (correct retrieval)
- `what_does('river','look')` = `None` (no-confab abstention — never stored)
- **MOAT PASS** — the self-org place afferent + the DA-over-`snc` modulator do NOT perturb the
  parser/conversational comprehension.

## Verdict

**SCOPE-GO.** The self-organized spiking `place` critic composes on the merged "one brain" bridge: the
host-Gaussian `vs_place_context` is retired for self-organized spiking place cells, the conversational
slices stay array-disjoint, and the no-confab moat is intact. This alone is a TRUE-ONE-BRAIN breadth win
(one more host shortcut → neurons/synapses).

## What's next (the δ-lift GPU probe — NOT yet run)

The load-bearing question for value (b): does the self-org `place→striosome_value` coincidence-volley fire
the MSN-D1 critic AND yield a δ=r−V gap ABOVE the CYCLE-212 ~1.3 cap (the position-blind-bootstrap ceiling)?

Protocol to port (NESTED CLOSURES in `g11_bg_runner.py` run_seed — not importable, must be mirrored, as the
`vs_place_context` value-train build did):
- STEP-1 `_run_place_selforg` (`g11_bg_runner.py` ~self-org runner): self-organize the place fields.
- STEP-2 `_run_place_value_training` (`:5716`): DA-gated STDP grows V on `place→striosome_value`.
- Stage-B `_run_stage_b_smoke` (`:5893` self-org branch): the RPE battery / δ measurement.
- Anti-cheat: lesion `place→striosome_value` → δ must collapse; moat intact.

**AUDIT FLAG (from the prompt + g11 comments):** the self-org place code is CuPy-non-deterministic
(transpose-SpMV atomic scatter — `sim/bridge.py` ~5771/5812 coincidence/GABA_B matvec; research
`2026-06-10-N9-placecode-reproducibility-robustness-research.md`). The volley strength (hence the critic
rate) varies 28–118 Hz run-to-run. `enable_critic_homeostasis` (the `critic_only` mask) is the in-tree
mitigation (intrinsic per-region homeostasis defends a target critic rate). If reproducibility blocks a
clean δ measurement even with homeostasis, a `sim/` determinism edit on those matvec sites would be needed —
flag a byte-level diff for controller review BEFORE making it.

## Files

- `research/runners/nav_conv_merged_bridge.py` — the builder edit (this commit).
