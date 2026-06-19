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

### Pathway-level verification (numpy CPU) — the self-org critic is structurally CORRECT, not just region-present

| pathway | plastic | plasticity_gate | transmission_gate | coincidence_detector |
|---|---|---|---|---|
| `place_sensors → place` | True | `landmark_to_place` | — | False |
| `place → place_fs` (FS-PING excite) | False | — | — | False |
| `place_fs → place` (FS-PING GABA_A) | False | — | `place_fs_gate` | False |
| `place → striosome_value` (the value afferent) | True | `value_input` | — | **True** |
| `striosome_value → snc` (the δ=r−V GABA_B route) | False | — | `critic_snc_window` | False |
| up-state arm `vs_place_drive → striosome_value` | **ABSENT** | — | — | — |

This is the decisive structural difference from the `vs_place_context` value-train critic: the self-org branch
has **NO up-state arm**. The cold MSN-D1 is fired by the FS-PING-synchronized place volley through the
`coincidence_detector=True` `place→striosome_value` pathway, which is **position-SELECTIVE** (the learned place
code), not the position-BLIND dense up-state that floored the CYCLE-211 δ at ~1.3. So the δ *can* lift past that
cap — this is exactly the mechanistic premise of item #5's value (b), now confirmed wired on the merged bridge.

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

**Engineering note for whoever picks this up — the δ probe for #5 is a DIFFERENT protocol from the
`vs_place_context` value-train (`_merged_navcritic_valuetrain.py`), NOT a 2-kwarg swap.** The `vs_place_context`
template drives the place code as a host-rendered Gaussian **directly into the afferent region**
(`_vs_drive → cp_external_input_current[vs_place_context]`) — but for the self-org critic the place code must be
**self-organized FIRST** (STEP-1) and the drive enters via `place_sensors` (bearing+distance render), with the
critic fired by the FS-PING **coincidence volley** through `place→striosome_value`, not a Gaussian current. So
the three STEP closures must be ported in their **self-org form**, not the `vs_place_context` form.

**KEY SHORTCUT FOUND (use this):** `run_g11` in `g11_bg_runner.py` ALREADY has a standalone `stage_b_smoke`
path (`:5874-5899`) that runs `_run_place_selforg` → `_run_place_value_training` → `_run_stage_b_smoke` and
**returns the δ measurement, exiting before the nav loop** — i.e. the exact STEP-1+STEP-2+Stage-B δ probe is
already implemented for the STANDALONE g11 bridge (validated by `n9_place_graded_critic_stage2_derisk.py`). The
cheapest merged-bridge port is therefore EITHER (a) a thin runner that calls the three self-org closures'
*logic* against the merged bridge (mirroring how `_merged_navcritic_valuetrain.run_value_train` ported
`_run_place_value_training`), OR (b) — much cheaper if it works — confirm the standalone g11 Stage-B δ on the
self-org critic is unchanged when the conv slices are appended, since they are array-disjoint with zero
out-edges into the critic (the composition smoke above already proves disjointness). Option (b) is the
recommended cheap-first next move: run `n9_place_graded_critic_stage2_derisk` (or `run_g11 --stage-b-smoke
--neural-place-selforg`) standalone for the baseline δ, then the merged equivalent, and compare. Anti-cheat:
lesion `place→striosome_value` (or the `striosome_value→snc` GABA_B `cp_gabab_synapse_mask`) → δ must collapse;
moat intact (already PASS above).

**AUDIT FLAG (from the prompt + g11 comments):** the self-org place code is CuPy-non-deterministic
(transpose-SpMV atomic scatter — `sim/bridge.py` ~5771/5812 coincidence/GABA_B matvec; research
`2026-06-10-N9-placecode-reproducibility-robustness-research.md`). The volley strength (hence the critic
rate) varies 28–118 Hz run-to-run. `enable_critic_homeostasis` (the `critic_only` mask) is the in-tree
mitigation (intrinsic per-region homeostasis defends a target critic rate). If reproducibility blocks a
clean δ measurement even with homeostasis, a `sim/` determinism edit on those matvec sites would be needed —
flag a byte-level diff for controller review BEFORE making it.

## Files

- `research/runners/nav_conv_merged_bridge.py` — the builder edit: `nav_critic_place_selforg` kwarg on
  `build_merged_nav_conv_bridge` + `MergedNavConvAgent`, the mutual-exclusivity assert, `neural_place_selforg`
  forwarded into `build_bg_brain_regions`. Additive, default-off = byte-identical.
- `research/runners/g11_bg_runner.py` — UNCHANGED (consumed): `:1175`/`:1783` self-org region+pathway branches,
  `:3853` the `enable_convergent_upstate` hard-gate, `:5874-5899` the standalone `stage_b_smoke` δ path,
  `--neural-place-selforg` / `--stage-b-smoke` CLI flags.
- `research/runners/_merged_navcritic_valuetrain.py` — UNCHANGED; the `vs_place_context` value-train TEMPLATE
  the δ-probe port would mirror (but in the self-org form — see the engineering note above).
- The numpy CPU smoke evidence (transient logs, not committed): `research/findings/raw/_smoke_place_selforg*.log`.

## Verdict (one line)

**SCOPE-GO** — the self-org spiking `place` critic composes correctly on the merged "one brain" bridge
(regions + pathways + `coincidence_detector` + moat all verified, host `vs_place_context` retired, no up-state
arm), retiring a host position-code shortcut. The δ-lift GPU probe (value (b)) is the cheap follow-on, with the
recommended cheapest-first path (re-use the standalone g11 `stage_b_smoke`, compare standalone-vs-merged δ on the
disjoint self-org critic) documented above.
