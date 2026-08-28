---
type: biology
id: affective-marker-lateral-inhibition-wta
mechanism: A continuous felt-affect signal (the #81 graded-affect ladder's own mood/felt-arousal read) is projected as a topographic population code onto a small pool of excitatory marker-coding assemblies; each assembly recruits its own fast-spiking-interneuron sub-pool that cross-inhibits every OTHER assembly (mutual/reciprocal lateral inhibition), so the assembly whose spiking rate wins the resulting competition SELECTS the discrete affective expression marker — the SELECTION is a spiking winner-take-all race, not a host dict lookup on a pre-binned level.
status: established
last_verified: 2026-08-28
current_finding: research/findings/2026-08-28-affect-marker-spiking-wta-derisk.md
current_status: "DE-RISK GO (6/6 seeds, numpy-CPU, NO sim/ edit). Additive, default-OFF (BRAIN_AFFECT_MARKER_SPIKING). Byte-identical-off, load-bearing (mood sweep across all 6 registers selects the matching marker, 36/36 rows), lesion-vanish (cutting the felt-state->assembly projection collapses every read to no clean winner -> an honest no-lead turn, 36/36 rows), and a shuffle anti-cheat (mis-routing which physical assembly receives which register's tuning drive changes the reported marker on 30/36 (seed,level) pairs, matching the ~1/6 expected fixed-point rate of a random 6-permutation) all PASS. CHARACTERIZED BOUNDARY: near a register's tuning-curve boundary (measured at mood=+0.069, almost exactly equidistant between two adjacent centers) the circuit is honestly LESS decisive than the old hard host threshold it replaces -- it reports 'no clean winner' rather than force a pick. This is a property of reading a graded population code, not a defect, but it means byte-identical parity with the old `_LEAD_WORD` table is claimed only for the flag-OFF path."
sources:
  - path: "doi:10.1126/science.3749885 (Georgopoulos, Schwartz & Kettner 1986, Science 233:1416 -- 'Neuronal population coding of movement direction')"
    anchor: "the movement direction is predicted by the vector sum of the preferred-direction contributions of the population of directionally-tuned motor cortical cells"
    note: "EXTERNAL. The population-vector / labeled-line coding precedent this circuit's topographic drive follows: a continuous quantity (here, felt valence, not movement direction) is recovered from / read out through a bank of narrowly-tuned units, each maximally driven near its own preferred value. Our 6 marker-assembly Gaussian tuning curves over the #81 ladder's mood axis are the same principle applied to a continuous affect dimension instead of a motor direction."
  - path: "doi:10.1037/h0077714 (Russell 1980, J. Pers. Soc. Psychol. 39:1161 -- 'A circumplex model of affect')"
    anchor: "affect concepts fall in a roughly circular order in a two-dimensional bipolar space, the coordinates of which can be roughly described as pleasure-displeasure and degree of arousal"
    note: "EXTERNAL. The justification for treating discrete affect WORDS (Wonderful/Gladly/Sure/Hm/Honestly/Frankly) as graded REGIONS of one shared continuous valence axis rather than separate faculties -- the 6 marker-assembly tuning centers are placed along the SAME mood axis the #81 Koulakov/Goldman ladder already implements (at each register's existing #84 mood-bin midpoint), not a newly invented axis."
  - path: "research/biology/interoceptive-affect.md"
    anchor: "A simulated interoceptive BODY-STATE"
    note: "LOCAL. The upstream #81/#84 mechanism this circuit consumes (the felt mood/felt-arousal read off cp_firing_states) is unchanged by this de-risk; this entry documents only the NEW marker-SELECTION step downstream of that read."
  - path: "research/runners/bg_action_selection_production_organ.py"
    anchor: "D1->GPi direct-path disinhibition, the GPe/STN indirect path"
    note: "LOCAL. The mutual/reciprocal cross-inhibition motif this circuit generalizes from N=2 channels (the 6-seed flip-soak GO'd SPEAK-vs-STAY-SILENT basal-ganglia action selector) to N=6 marker assemblies -- a lighter-weight FSI-mediated lateral inhibition (Grossberg 1973's on-center/off-surround competitive-network motif; Douglas & Martin 2004's canonical cortical microcircuit) rather than a full basal-ganglia loop, proportionate to 'a small pool of marker-coding assemblies'."
implemented_by:
  - research/runners/_affect_marker_wta_derisk.py
  - research/runners/_affect_marker_wta_verify.py
findings:
  - research/findings/2026-08-28-affect-marker-spiking-wta-derisk.md
---

# The affective expression-marker SELECTION is a spiking lateral-inhibition WTA, not a host dict lookup

**What is measured.** The #84 affect-drives-chat coupling's final step — which word ("Wonderful"/"Gladly"/"Sure"/
"Hm"/"Honestly"/"Frankly") the reply leads with, given the #81 ladder's felt mood — is now available as a genuine
spiking competitive-selection circuit, additive and default-OFF. Six small excitatory marker assemblies, each with
its own cross-inhibiting fast-spiking-interneuron sub-pool, receive a topographic Gaussian-tuned drive from the
CONTINUOUS felt mood; the assembly that wins the resulting lateral-inhibition race names the marker. 6/6 seeds:
byte-identical when off, load-bearing when on (varying mood changes the selected marker, tracking the intended
register), lesion-vanishing (cutting the felt-state→assembly projection collapses every read to "no clean winner"
→ an honest no-lead turn, not a silent revert to the host table), and shuffle-anti-cheated (mis-routing which
physical assembly gets which register's drive changes the reported marker, proving the selection reads off which
assembly actually won, not a formula blind to the wiring).

## Why this is brain-based, and where the boundary sits

The felt-state VARIABLES (mood, felt_arousal) are unchanged — they are the #81 ladder's own neural read, already
established (`interoceptive-affect.md`). What was host before this de-risk was the STEP that turns that read into
a discrete surface token: a Python dict keyed on a pre-binned integer level. That step is now, additively, a real
competitive network: the drive computation (mood → per-assembly current via a Gaussian tuning curve) is the ONLY
host arithmetic remaining, exactly analogous to how an existing BG action-selector organ computes a per-candidate
SALIENCE bias in Python before handing it to the striatal race — the COMPETITION itself (which assembly ends up
firing) is neurons and synapses on a real `SimulationBridge`, read via `cp_firing_states`, never a host argmax
over a formula's output.

## The characterized boundary (not a config constant)

No `constraints_config` is bound: the dead-margin (0.05) and the tuning-curve widths (`MOOD_SIGMA=0.02`,
`AROUSAL_SIGMA=0.02`) are EMPIRICAL calibrations on this substrate (an intact separation of ~0.15–0.17 measured
6/6 seeds gives ample headroom over the margin), not biology-REQUIRED constants — the topographic CENTERS,
however, are placed by construction at the pre-existing #84 mood-bin midpoints, not independently tuned. The
circuit is honestly LESS decisive than the hard host threshold it replaces exactly at a tuning-curve boundary
(measured at mood=+0.069); this is a property of population coding, not a defect, and does not weaken the
load-bearing / lesion / anti-cheat results, which are all measured well away from that boundary or explicitly
tolerant of the ambiguous case (part (B)'s representative moods are the bin MIDPOINTS, not the boundaries).
