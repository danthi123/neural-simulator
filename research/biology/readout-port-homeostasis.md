---
type: biology
id: readout-port-homeostasis
mechanism: Each LIF unit of a readout class population regulates its OWN operating point from its own activity -- a multiplicative synaptic-scaling factor on all of its afferents plus an intrinsic-excitability threshold -- so that its firing-rate distribution over experienced (training) inputs approaches a set point (Triesch-rule intrinsic plasticity toward an exponential rate distribution with a fixed mean), learned on training exposures and then frozen; this replaces the fixed host constants (read_gain, read_bias) that left the port's drive saturated or silent.
status: de-risking
last_verified: 2026-09-24
current_finding: research/findings/2026-09-24-vision-readout-port-homeostasis-intrinsic-plasticity-PREREGISTERED.md
current_status: "PRE-REGISTERED (2026-09-24), not yet scored. Built as the companion process the attention-gated-soft-fbgain port was missing: on a non-evaluation dev seed the port's pre-spike class drive sat at trial-independent offsets of hundreds with a trial SD of ~2, so two class populations were rectified silent and two were pinned at the refractory ceiling (count SD exactly 0) -- a constant output. `--port-homeostasis ip` gives every unit its own learned gain + threshold; the decisive 6-seed run and its bands are fixed in the pre-registration."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Neuronal Excitability Is Plastic"
    note: "EXTERNAL. Kandel PNS-6e (Ch. on voltage-gated channels): the channel complement that sets a neuron's firing rate is not fixed but changes with the neuron's own activity; on a longer time scale increased network activity decreases the excitability of individual neurons -- 'a homeostatic feedback system'. The intrinsic-threshold half of this mechanism."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "a homeostatic feedback system"
    note: "EXTERNAL. Same passage; the regulated variable is the neuron's own activity, which is what the rule's error term reads (the unit's own realized spike counts). Synaptic scaling itself is Turrigiano et al. 1998 (Nature 391:892, doi:10.1038/36103, PMID 9495341 -- multiplicative scaling of all of a neuron's inputs with its activity, 'may help to ensure that firing rates do not become saturated'); intrinsic-excitability regulation Desai, Rutherford & Turrigiano 1999 (Learn Mem 6:284, PMID 10492010); the gain+threshold rule toward an exponential rate distribution Weber & Triesch 2008 (Neural Comput 20:1261, doi:10.1162/neco.2007.02-07-472) after Triesch 2005 -- read via PubMed, not locally anchored."
  - path: research/runners/_vision_lindiscrim_readout_derisk.py
    anchor: "INTRINSIC PLASTICITY + SYNAPTIC SCALING of every class-population LIF unit"
    note: "LOCAL. The implementation (`_learn_port_homeostasis`, `_ip_port_current`), its declared host shortcuts, and the `--ip-lesion-update` lesion."
implemented_by:
  - research/runners/_vision_lindiscrim_readout_derisk.py
findings:
  - research/findings/2026-09-24-vision-configural-binding-spiking-feedback-divisive-gain-control-readout-NEUTRAL-6seed.md
  - research/findings/2026-09-24-vision-readout-port-homeostasis-intrinsic-plasticity-PREREGISTERED.md
---

# A readout neuron is not handed its operating point

The class-population port of the configural-binding readout received its drive through two fixed host
constants (`read_gain`, `read_bias`). Whenever the upstream drive moved (here: the per-class top-down gain
multiply leaves trial-independent offsets of hundreds of current units), nothing moved the port with it, and
the LIF units sat outside their dynamic range -- silent or at the refractory ceiling -- on every trial.

Real neurons run two homeostatic processes alongside every synaptic change: synaptic scaling (all afferents
of a cell scaled together, in proportion to their strength, as a function of the cell's own firing) and
regulation of intrinsic excitability. Both are driven by the cell's OWN activity, and both act slowly
relative to a single stimulus. The rule used here (Triesch's intrinsic-plasticity gradient rule) adapts a
gain and a threshold so the unit's firing-rate distribution over its experienced inputs approaches an
exponential with a fixed mean -- maximal output entropy at a fixed metabolic cost -- which is exactly the
property a constant-output port lacks.

**Protocol.** Learned on TRAIN exposures only (labels never read), frozen for every evaluation read. The
set point (mean firing fraction 1/n_classes of the refractory ceiling) is fixed a priori, not fit.
