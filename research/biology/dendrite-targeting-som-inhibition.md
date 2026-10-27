---
type: biology
id: dendrite-targeting-som-inhibition
mechanism: Somatostatin (SOM / Martinotti-like) interneurons with facilitating input synapses inhibit the DISTAL dendrite, where an IPSP stops a local dendritic depolarization (a plateau) that somatic inhibition cannot reach
status: specified
last_verified: 2026-10-27
current_finding: research/coordination/cs-race-f1i-PACKET.md
current_status: "ENGINE SUPPORT BUILT, NO RESULT. The default-OFF sim/ edit (cfg.enable_apical_gaba_routing + receptor='gaba_a_apical') routes a pathway's GABA_A spikes to a bounded apical conductance instead of somatic g_i; flags-off byte identity and the per-step bound are pinned by tests/test_apical_gaba_routing.py. Whether this inhibition ends an older word's held plateau (cs-race f1i C7) and whether facilitation is NECESSARY (the NOFAC arm) are unmeasured."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "dendritic IPSP is more effective in preventing local"
    note: "Kandel 6e Ch. 13, Fig. 13-16: the axodendritic IPSP is the one that stops LOCAL dendritic depolarization; the axosomatic IPSP is the one that stops action potentials. The compartment, not the transmitter, sets which job inhibition does."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "activate slowly in response to an excitatory input and"
    note: "somatostatin interneurons target distal dendrites, activate slowly, and their IPSPs grow with repetitive activation (synaptic facilitation); parvalbumin cells are the fast, depressing, perisomatic counterpart"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "require several pulses before they begin to"
    note: "Buzsaki, Rhythms of the Brain Ch. 3: dendrite-targeting interneurons have facilitating input synapses (a high-pass filter); at a low pyramidal rate inhibition stays perisomatic, at a high rate it shifts to the dendritic domain"
  - path: ~/Projects/sim-catalog/references/textbooks/okeefe-nadel-cognitive-map/OKeefe-Nadel-1978-HippocampusCognitiveMap.txt
    anchor: "considered it to be an active IPSP in the dendrites"
    note: "O'Keefe & Nadel 1978, after Fujita & Nakamura 1961: an apical-dendrite IPSP inhibited the dendritic spike evoked by Schaffer-collateral stimulation"
  - path: doi:10.1073/pnas.95.9.5323
    anchor: "Differential signaling via the same axon of neocortical pyramidal neurons"
    note: "Markram, Wang & Tsodyks 1998 PNAS 95:5323 (PMID 9560274), read in full text (PMC20259) on 2026-10-27: pyramid -> interneuron facilitating connections, 7 convergent pairs in L2/3-4, U = 0.049 +/- 0.037, D = 399 +/- 295 ms, F = 1797 +/- 1247 ms. The paper does NOT classify the target interneurons (no 'bitufted'); the SOM-like reading rests on the facilitation."
constants:
  E_gaba_a_mV: -75.0
  tau_gaba_a_ms: 10.0
  stp_U_pyr_to_interneuron: 0.049
  stp_tau_rec_ms: 399.0
  stp_tau_facil_ms: 1797.0
implemented_by:
  - sim/bridge.py
  - sim/config.py
  - tests/test_apical_gaba_routing.py
findings:
  - research/findings/2026-07-21-gap5-2assembly-selective-inhibition-family-NEGATIVE-points-to-dendritic.md
  - research/findings/2026-07-30-gap5-plateau-termination-is-INTRINSIC-and-the-machinery-already-exists.md
---

# Inhibition has a compartment: a dendritic plateau is ended at the dendrite

**The claim the code must respect.** Where an inhibitory synapse lands decides what it can stop. Kandel's
Fig. 13-16 states the asymmetry directly: an axosomatic IPSP is the stronger brake on action potentials, while an
axodendritic IPSP is the stronger brake on a local dendritic depolarization. Somatostatin-expressing (Martinotti-like)
interneurons are the cells that target distal dendrites. Their input synapses facilitate, so they are recruited by
sustained high-rate pyramidal firing and much less by low-rate firing (Buzsaki Ch. 3: a high-pass filter). An apical
IPSP blocks a dendritic spike (Fujita & Nakamura 1961, via O'Keefe & Nadel).

**Why the engine needed it.** Before 2026-10-27 every inhibitory spike in the engine landed in the somatic `g_i`, and
the BDSP apical compartment (`cp_v_apical`) received no synaptic conductance at all. The apical -> soma coupling is
one-way and positive-only, so no inhibitory cell could pull an apical plateau down. The record met this compartment
gap twice: somatic selective inhibition could not gate an apical plateau attractor (07-21), and GABA_B as a plateau
terminator was undefined because `I_gabab` reads the somatic voltage (07-30).

**What the engine now does (default OFF).** With `cfg.enable_apical_gaba_routing` and a plan pathway tagged
`receptor="gaba_a_apical"`, that pathway's spikes are removed from the somatic `g_i` matvec and increment a per-neuron
apical conductance with the GABA_A time constant. The BDSP apical update applies it against the APICAL driving force
`(E_i - v_apical)` as a bounded exponential-Euler step, so v_apical reaches E_i at most and never passes it at any
conductance (the 2026-09-29 explicit-Euler overshoot family is not reintroduced). The somatic attenuation of an
axodendritic IPSP is taken as total, which is an idealization: Kandel's figure shows a small but non-zero somatic IPSP.

**What is NOT claimed.** No run has used the routing. That SOM-like inhibition ends an older plateau while the newer
one survives (cs-race f1i C7), that the held trace does not recruit its own eraser (C8), and that facilitation is
necessary (the NOFAC arm) are all open. The SOM cells in f1i use the probe's default intrinsic type, not the LTS type
(branch B2), and their wiring is a declared scaffold whose neural replacement is inhibitory plasticity.
