---
type: biology
id: sacramento-selfpredicting-microcircuit
mechanism: A learned SST/PV-like INTERNEURON microcircuit whose plastic apical-cancellation weight LEARNS to predict and cancel the top-down feedback, so the pyramidal APICAL dendrite is SILENT when the network is already correct and carries a true prediction error otherwise -- the credit signal is a LEARNED error, not a FROZEN fixed-random projection that never zeroes
status: hypothesis
last_verified: 2026-09-09
current_status: "MECHANISM IMPLEMENTED IN-ENGINE (2026-09-09), decisive 6-seed run QUEUED (not yet a result). The committed additive/default-off cfg.enable_selfpredicting_interneuron makes the ENGINE, inside SimulationBridge._run_one_simulation_step, PROJECT the interneuron rate through the PLASTIC substrate weight cp_spi_wpi to form the apical cancellation cp_bdsp_int_drive AND UPDATE cp_spi_wpi by the local Sacramento self-prediction rule (sim/dendritic_plasticity.selfpredicting_interneuron_update) toward the fixed feedback cp_spi_Y. Construct-smoke PASS (numpy): cp_spi_wpi learns in-engine, the engine forms int_drive, transport-free (AST), byte-identical when off, cfg.seed controls the substrate. SCOPE, and it is load-bearing: the LEARNED interneuron microcircuit is NOT fresh -- it is already tested (a) at RATE (2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO: earned apical-silence, but accuracy-INDISTINGUISHABLE from fixed-FA) and (b) on-bridge as a RUNNER-SUPPLIED host cancellation (2026-08-18-gap4-microcircuit-expander-6seed-NOTGO; the sibling _gap4_onbridge_spiking_selfpredict skeleton whose deep-hidden arc reached 'the crux was never askable' -- a READ-regime foreclosure where even the copied-weight ceiling could not fit). The genuinely-untested distinction, named by 2026-09-09-gap4-dendritic-urbanczik-senn-read-snr-clean-NO-GO, is LEARNING the cancellation IN-ENGINE on the substrate (this build) rather than supplying it from the runner. dendritic/burst/BDSP READ-side deep credit is already tested-negative (2026-07-22-gap4-real-issue-NOT-dendrites); this binding is the FEEDBACK-SIGNAL side, not another read lever."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "action potentials can backpropagate"
    note: "the physical basis for a per-neuron dendritic teaching signal: the somatic spike back-propagates INTO the apical dendrite, so the apical compartment has a LOCAL copy of the neuron's own output -- the residual the interneuron's cancellation is measured against (no non-local error transport). Same substrate the Urbanczik-Senn binding assumes."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "regenerative depolarization, referred to as an nmda"
    note: "active apical conductances (NMDA / voltage-gated Ca2+) give the apical compartment its own regenerative response that a top-down feedback drives -- the depolarization the SST/PV interneuron cancellation SUBTRACTS, so the RESIDUAL apical is the clean prediction error (silent when the interneuron's learned prediction matches the top-down)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "voltage-gated ca2+ channels"
    note: "the ionic machinery of the two-compartment apical dendrite (Kandel Fig 13-17) -- the substrate the learned interneuron microcircuit rides on; the cancellation gates whether this apical Ca2+ event is expressed (a controllable, learnable inhibitory drive), not a fixed one."
implemented_by:
  - sim/config.py
  - sim/dendritic_plasticity.py
  - sim/bridge.py
  - research/runners/_gap4_selfpredict_interneuron_inengine_derisk.py
findings:
  - research/findings/2026-09-09-gap4-dendritic-urbanczik-senn-read-snr-clean-NO-GO-second-read-lever.md
  - research/findings/2026-07-22-gap4-real-issue-NOT-dendrites-and-timing-FIRST-CLASS-deep-research.md
---

# The gap#4 credit signal is a LEARNED interneuron self-prediction, not a frozen fixed-random projection

**The claim the code must respect.** Sacramento, Costa, Bengio & Senn (2018, "Dendritic cortical microcircuits
approximate the backpropagation algorithm", NeurIPS 2018 / arXiv:1810.11393) show a biologically-plausible
microcircuit in which pyramidal APICAL dendrites receive top-down feedback AND a cancelling projection from a
population of lateral inhibitory (SST/PV-like) INTERNEURONS. The interneuron's plastic weight W^PI *learns* (by a
local rule, from the pyramidal cell's own lateral input) to PREDICT and cancel the top-down feedback. At the
self-predicting fixed point the cancellation matches the top-down, so the apical dendrite is **silent when the
network's own prediction is already correct**, and the *residual* apical carries a true prediction ERROR when it is
not -- which the feedforward synapses read as their credit signal. This is the fix for the defect the project's own
record root-caused: our credit apical carried a **FROZEN fixed-random projection of the raw output error that never
falls to zero when the network is already correct** (`2026-07-22-gap4-real-issue-NOT-dendrites`), so it keeps
nudging the weights even when correct. Urbanczik & Senn (2014) supply the companion local rule the pyramidal cell
uses (a neuron's dendrite predicts its own somatic spiking; see `urbanczik-senn-dendritic-prediction`); Sacramento
2018 adds the *learned interneuron* that makes the apical silent-when-correct, closing the loop without weight
transport.

## What is newly established here, and what is explicitly NOT (do not overclaim)

**Implemented, and smoke-verified (numpy).** The learning is now carried IN-ENGINE on the substrate:
`sim/dendritic_plasticity.selfpredicting_interneuron_update` is the local, transport-free Sacramento Eq.9 update
`dW^PI = lr * outer(int_rate, int_rate @ (Y_fixed - W^PI))`, and the guarded block in
`SimulationBridge._run_one_simulation_step` (reached only when `enable_selfpredicting_interneuron` and the runner
has installed the small logical `cp_spi_*` arrays) forms the apical cancellation `cp_bdsp_int_drive` from the plastic
`cp_spi_wpi` and applies that update each credit step. The construct-smoke confirms `cp_spi_wpi` changes in-engine,
the engine (not the runner) forms `int_drive`, the update reads no forward weight (AST), the path is byte-identical
when the flag is off, and `cfg.seed` controls the substrate.

**NOT established (the decisive question, queued).** Whether learning the cancellation IN-ENGINE beats the FROZEN
fixed-random feedback baseline on held-out deep-credit accuracy. The learned microcircuit is already tested and did
NOT separate from fixed-FA at RATE (`2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO`) and was a
NOT-GO as a RUNNER-SUPPLIED cancellation on-bridge (`2026-08-18-gap4-microcircuit-expander-6seed-NOTGO`); the
deep-hidden on-bridge instrument is moreover READ-regime-foreclosed ("the crux was never askable" -- the copied-weight
ceiling itself could not fit). So the in-engine build is the pre-registered RANK-1 lever, but the decisive 6-seed
GPU run must clear an INTERPRETABILITY precondition first: the transport-ceiling arm must exceed chance (else the
verdict is UNDEFINED, not a false NO-GO). An honest negative is the deliverable.

## What this entry cannot catch

No `constraints_config`: the microcircuit's requirements are structural (a two-compartment apical, a plastic
interneuron cancellation projection, a local self-prediction rule reading only activities), not a numeric config
default that `biology_check --config` can compare by equality. The guard against a host shortcut is the runner's
LESION arms -- FREEZE `cp_spi_wpi` (silence cannot be earned), apical-lesion, shuffle-target, shuffle-error, and the
AST no-forward-weight check on the credit path -- which are part of the result, not this binding.
