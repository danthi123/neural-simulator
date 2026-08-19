---
type: biology
id: urbanczik-senn-dendritic-prediction
mechanism: A neuron's DENDRITE predicts its own SOMATIC firing; the local mismatch (soma - dendritic prediction) IS the teaching error that drives synaptic change -- no separate error unit and no host error formula
status: established
last_verified: 2026-08-19
current_finding: research/findings/2026-08-19-neural-error-onbridge-GO.md
current_status: "6-seed GO, now ON THE LIVE BRIDGE. The 'was I wrong?' teaching error that corrects the brain's word choices is the read-out neuron's own soma-minus-dendrite mismatch (shipped urbanczik_senn_update). Established first in numpy (2026-08-19-neural-error-population-GO.md, NEURAL=0.964), then delivered through the live SimulationBridge's per-synapse reward/eligibility channel (2026-08-19-neural-error-onbridge-GO.md, NEURAL-onbridge=0.929=97% of the host on-bridge error, 5/6 seed parity), so the production learning loop USES the neural error. NO sim/ edit (routed runner-side into the already-present cp_per_synapse_reward_override array; production path byte-identical to main). Silencing the dendritic self-prediction OR the somatic teaching OR mis-addressing the error each collapses on-bridge learning to chance (attribution 0.95-0.97). Needs a K=16 population read for the short on-bridge budget's spike-count SNR. Retires the host err = est - target formula on the production path."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "action potentials can backpropagate"
    note: "the physical basis for the soma-vs-dendrite comparison: the somatic spike back-propagates INTO the dendrite, so the dendritic compartment has a LOCAL copy of what the soma did -- the signal the U-S rule compares its dendritic prediction against (no non-local error transport)"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "backpropagate from the cell body"
    note: "same event named from the somatic end -- the back-propagating AP is what makes 'the dendrite knows the soma fired' a structural property of one neuron, not host bookkeeping"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "regenerative depolarization, referred to as an nmda"
    note: "active dendritic conductances (NMDA / voltage-gated Ca2+) give the dendrite its own regenerative response = a genuine dendritic PREDICTION signal that can differ from the soma, so the mismatch is non-trivial"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "voltage-gated ca2+ channels"
    note: "the ionic machinery of the dendritic compartment's own (predictive) activity -- Kandel Fig 13-17, the two-compartment substrate the U-S rule assumes"
implemented_by:
  - sim/dendritic_plasticity.py
  - research/runners/_neural_error_population_derisk.py
  - research/runners/_neural_error_onbridge_derisk.py
findings:
  - research/findings/2026-08-19-neural-error-population-GO.md
  - research/findings/2026-08-19-neural-error-onbridge-GO.md
---

# The teaching error is a neuron's own somato-dendritic mismatch, not a host subtraction

**The claim the code must respect.** Urbanczik & Senn (Neuron 81:521-528, 2014, "Learning by the Dendritic
Prediction of Somatic Spiking", PubMed 24507189) show a local, biologically-plausible rule in which a neuron's
DENDRITIC compartment learns to PREDICT its own somatic firing, and the plastic weights change in proportion to the
mismatch `(somatic_rate - phi(dendritic_voltage))`. The subtraction that yields the error is done inside one
neuron by comparing two of its own compartments -- made physically possible because the somatic action potential
**back-propagates** into the dendrite, giving the dendrite a local copy of what the soma actually did. There is no
separate error unit and no non-local error transport; the error is intrinsic to the cell. Mikulasch, Rudelt,
Wibral & Priesemann (Trends Neurosci 46:45-59, 2023, PubMed 36577388) frame this as the general point: prediction
errors are computed locally in dendritic compartments, not in separate units.

**Why this row exists.** The project's read-out that learns the brain's word choices was trained by a HOST formula
`err_j = est_j - target_j` -- a documented BRAIN-BASED-ONLY shortcut (the *brain* was not computing the error, the
host bookkeeping was). This binding is the biological warrant for removing that formula: the same corrective error
can be the read-out neuron's own soma-minus-dendrite mismatch, with the teacher entering only as a somatic nudge
(a legitimate env/teacher scaffold) and the dendrite predicting the estimate through the plastic weights.

## What is actually established here, and what is NOT

**Established (`_neural_error_population_derisk.py`, 6 seeds).** On the role-filler word-acquisition task, the
read-out trained by the shipped `urbanczik_senn_update` soma-vs-dendrite mismatch matches the exact host-error
read-out on held-out generalization (see `current_status`), and every anti-cheat collapses: silencing the
dendritic self-prediction (pin the dendrite so it no longer predicts the soma), silencing the somatic teaching
nudge (soma == dendrite -> mismatch == 0), and mis-addressing the error across outputs (scramble) each drop
learning to floor. The neuron's own error, not a residual host formula, drives the plasticity.

**NOT established here.** This binding covers the ERROR-SOURCE only (HOW the corrective signal is computed). It
does NOT touch the mouth/word-readout READ-REGIME (a separate frontier), and it does NOT claim the two-compartment
credit-assignment result for DEEP/hidden layers -- that is the separate, previously-characterized dendritic
credit question. The task here is a single plastic read-out layer.

## What this entry cannot catch

No `constraints_config`. The U-S rule's requirements are structural (two compartments; a soma-driving teacher; a
dendritic prediction through the plastic weights), not a numeric config default that `biology_check --config` can
compare. A runner could satisfy every anchor here and still compute the error with a host subtraction -- the guard
against that is the runner's LESION arms (silence the dendrite / silence the teacher / scramble), which are part
of the result, not this binding.
