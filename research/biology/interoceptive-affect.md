---
type: biology
id: interoceptive-affect
mechanism: A simulated interoceptive BODY-STATE (energy/comfort + arousal) is read by spiking interoceptive afferent populations that project SYNAPTICALLY onto the affect attractor, so the physiological condition of the body CAUSES the felt core-affect (valence + arousal) — Damasio somatic-marker / Craig interoception, realized on the P0.3 opponent NMDA attractor.
status: established
last_verified: 2026-08-19
current_finding: research/findings/2026-08-19-embodied-affect-interoception-GO.md
current_status: "BRAIN-BASED GO (6/6 seeds, numpy-CPU, NO sim/ edit). A minimal simulated body-state (homeostasis h in [0,1] = satiety/comfort; arousal a in [0,1] = heart-rate/sympathetic tone) enters the brain ONLY as an interoceptive afferent CURRENT (the body->sensory boundary, i_scale ~200 pA) onto three spiking interoceptive relay pools (intero_comfort/discomfort/arousal, Izhikevich RS). Each projects SYNAPTICALLY (AMPA, gated by intero_out) onto the reused P0.3 affect pools (intero_comfort->affect_vplus, intero_discomfort->affect_vminus, intero_arousal->affect_arousal). The felt state is the attractor's OWN read: mood = rate(V+)-rate(V-), felt_arousal = rate(affect_arousal). RESULT: a comfortable body -> POSITIVE valence (mood +0.08), a distressed body -> NEGATIVE valence (-0.08) (swing 0.16, ordered corr 0.84); an aroused body RAISES felt arousal (swing 0.08). LOAD-BEARING DISSOCIATION: cutting the interoceptive->affect synapses (intero_out=0) collapses BOTH channels to exactly 0.000 while the interoceptive pools STILL encode the body (corr 0.99) — the body signal is present but can no longer reach the feeling. A silence control (zero afferent current) agrees. CHARACTERIZED BOUNDARY: the affect reads the body as a BISTABLE SIGNED SWITCH (valence: comfort/distress; arousal: on/off ignition, gradedness Pearson 0.70), NOT a graded valence x arousal circumplex — the SAME P0.3 line/bump-attractor + dendritic surpass. The gradedness limit does NOT weaken the embodiment claim: the CAUSATION (body -> correct signed feeling) and the interoception dissociation are clean 6/6."
sources:
  - path: "doi:10.1038/nrn894 (Craig 2002, Nat. Rev. Neurosci. — 'How do you feel? Interoception: the sense of the physiological condition of the body')"
    anchor: "interoceptive afferents report the physiological condition of the body to the insular cortex, where it becomes the substrate of subjective feelings"
    note: "EXTERNAL (recorded for local addition). The canonical interoception->feeling pathway: lamina-I/vagal afferents carry the body's homeostatic state (energy, temperature, cardiorespiratory) via the thalamus to the (posterior->anterior) insula, forming the neural image of the body that IS the felt core-affect. Our intero_comfort/discomfort/arousal pools projecting synaptically onto the affect attractor are the spiking realization of the afferent->insula->affect read."
  - path: "doi:10.1098/rstb.1996.0125 (Damasio 1996, Phil. Trans. R. Soc. B — the somatic-marker hypothesis)"
    anchor: "bodily/somatic states mark options and percepts with an affective value that biases cognition and choice"
    note: "EXTERNAL. The functional claim de-risked here: a feeling has a BODILY cause (a somatic marker), not merely a lexical one; the body's physiological state is what colours a percept good or bad. Our comfort->V+ / discomfort->V- projection makes the body-state the cause of the affect sign."
  - path: "doi:10.7554/eLife.04811 (Keramati & Gutkin 2014, eLife — homeostatic regulation as reward)"
    anchor: "an interoceptive deficit current drives a hypothalamic drive population whose activity defines the homeostatic setpoint error"
    note: "EXTERNAL. The precedent for the body->sensory current boundary used here (AgRP/POMC deficit drive, catalog O.05/O.06): the homeostatic-drive GO (2026-06-17) already injected the body's energy deficit as an interoceptive afferent current at i_scale ~300 pA — this de-risk reuses that boundary and routes it onward, synaptically, into the affect attractor."
implemented_by:
  - research/runners/_embodied_affect_interoception_derisk.py
findings:
  - research/findings/2026-08-19-embodied-affect-interoception-GO.md
---

# Embodied affect — a simulated body-state causes the neural feeling

**What is measured.** A bounded first interoception->affect coupling: a minimal simulated body-state (satiety/comfort
+ arousal) is encoded by three spiking interoceptive relay pools and projected SYNAPTICALLY onto the reused P0.3
affect opponent-NMDA attractor. Sweeping the body-state moves the neural affect state in the correct direction
(comfort -> positive valence, distress -> negative valence, arousal -> raised felt-arousal), and cutting the
interoceptive->affect synapses decouples the feeling from the body while the pools keep encoding it. 6/6 seeds,
numpy-CPU, no `sim/` edit (an additive default-off `extra_regions`/`extra_pathways` seam on `AffectStateBrain`).

## Why this is brain-based, and where the boundary sits

The body VARIABLES are host — the body is a legitimate host interface, exactly like the world; they enter the brain
ONLY as an interoceptive afferent current (the same body->sensory boundary the homeostatic-drive GO used).
Everything from that current onward is neurons/synapses: the interoceptive pools FIRE in proportion to the body
signal (corr 0.99), and their SYNAPSES drive the affect attractor, whose recurrent NMDA dynamics settle the felt
state. The body->affect map is never computed in Python — asserted at runtime (the affect pools receive zero direct
external current every step; the felt state is read only as rate(V+)-rate(V-) off `cp_firing_states`).

## The characterized boundary (not a config constant)

No `constraints_config` is bound: the operating point (i_body ~200 pA, intero->affect weight ~10) is an EMPIRICAL
calibration on this substrate, not a biology-REQUIRED constant. The affect attractor reads the body as a BISTABLE
SIGNED SWITCH, not a graded circumplex — the same P0.3 latch limit whose surpass (a graded line/bump attractor with
adaptation eviction / the dendritic substrate) is already named. That gradedness limit is orthogonal to the
embodiment claim proven here (body -> correct signed feeling + interoception load-bearing, clean 6/6).
