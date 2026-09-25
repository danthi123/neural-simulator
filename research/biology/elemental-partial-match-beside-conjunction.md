---
type: biology
id: elemental-partial-match-beside-conjunction
mechanism: A strict two-input coincidence (conjunction) unit is never the only route from its inputs to the cell that reads it -- the same single inputs also reach that cell weakly (the passive, unamplified component a lone input still delivers), so a partial match (one of the two inputs) still casts a weaker vote instead of nothing, and every active input is represented somewhere
status: de-risking
last_verified: 2026-09-25
current_finding: research/findings/2026-09-25-lexicon-closed-class-frame-junction-dev-amendment3-elemental-s7-s42-NOT-READY.md
current_status: "DEV CHECK at seeds 7 and 42 (not evaluation seeds): NOT READY. The ungated elemental edge (AMENDMENT 3, flag BRAIN_LEARNED_REFERENT_JUNCTION_ELEMENTAL, default OFF) restored decisions: silent-NON 0.263 / 0.175, under the 0.30 bar, and held-out-noun recall R3 0.75 at seed 7. But on the queried battery words it carried 81% / 68% of the afferent drive, so it decided every admitted closed-class or adjective word (G2 24 / 9 mismatches), and removing the conjunction left the parse unchanged (G3'a fails, G3'b passes). The premise that it is the weaker vote by construction is measured false here. Next rung: one cell-wide normalisation across both edges, or Marr's gated elemental vote."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "dendrites may switch between passive"
    note: "Kandel ch. 13: '... dendrites may switch between passive and active integration depending on the precise timing and strength of synaptic inputs.' The SAME synapses are integrated two ways: actively (NMDA-dependent, supralinear) when they coincide, passively when they do not. A lone input is not deleted; it is integrated passively."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the distal dendrites usually produce only a very small"
    note: "Kandel ch. 13: a lone distal input produces only a very small depolarizing response at the soma, and enhances firing when paired with proximal input. SMALL, not zero: the unamplified single-input component is weaker than the coincidence-amplified one. The frame-junction point neuron delivers exactly zero for a lone afferent (the constant hyperpolarizing I_TONIC_J + a somatic threshold) -- the passive component was replaced by zero."
  - path: ~/Projects/sim-catalog/references/textbooks/cerebellum-marr/Marr-1969-cerebellar-cortex.txt
    anchor: "mossyfibremustwithhighprobabilitybeincludedinatleastonecodon"
    note: "Marr 1969 section 4.1.1 (the PDF text has no spaces): every active mossy fibre must with high probability be included in at least one codon. A conjunctive (codon) representation must not drop an active input. The strict (-1,+1) junction layer drops every one-sided occurrence (36-37% of the teacher curriculum's presented occurrences at dev seeds 7 and 42, frame_composition_s7_s42.json)."
  - path: ~/Projects/sim-catalog/references/textbooks/cerebellum-marr/Marr-1969-cerebellar-cortex.txt
    anchor: "codonsizeshoulddependontheamountofmossyfibre"
    note: "Marr 1969 section 4.0: the codon size (how many coincident inputs a granule cell needs) should depend on the amount of input -- sparse input must fall back to smaller codons. The same design principle as the elemental pathway: when the full conjunction is not available, a smaller (single-input) code must still carry the input. Marr's own implementation (a Golgi-cell regulator, 'inhibition increasing with increasing size of input') is the GATED alternative considered and not built in AMENDMENT 3."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "based either on each individual cue or on the combined"
    note: "Kandel ch. 52 (Fig. 52-11, Duncan et al. 2018): an outcome can be predicted from each individual cue (elemental) or from the cue configuration (configural); both strategies are used, and hippocampus-striatum interactions are 'sometimes competitive and sometimes cooperative'. Behavioural precedent that an elemental route runs beside the configural one rather than being replaced by it."
implemented_by:
  - research/runners/lexicon_frame_junction.py
findings:
  - research/findings/2026-09-24-lexicon-closed-class-frame-junction-PREREGISTRATION.md
  - research/findings/2026-09-25-lexicon-closed-class-frame-junction-dev-amendment2-s7-s42-NOT-READY.md
  - research/findings/2026-09-25-lexicon-closed-class-frame-junction-dev-amendment3-elemental-s7-s42-NOT-READY.md
---

# A conjunction needs an elemental partner, or partial evidence goes silent

**The wall this answers.** The frame-junction lexicon routes heard context to the noun/non-noun category pools
ONLY through two-input AND units, one per (word before, word after) pair. Round 2 made the AND nearly perfect
(0 and 1 violations in 10,000 junctions at dev seeds 7 and 42) and the lexicon stopped admitting most closed-class
words. The cost: 60% (seed 7) and 40% (seed 42) of the heard non-noun battery words left BOTH pools silent, against
11% and 18% for the single-offset v2 lexicon at the same seeds, and held-out-noun recall (route R3) fell to 0.25 at
seed 7. `research/findings/2026-09-25-lexicon-closed-class-frame-junction-dev-amendment2-s7-s42-NOT-READY.md`.

**The companion process (the wall-reframe question).** What does the real system run alongside a strict
coincidence detector, that the build replaced with a constant? In a pyramidal neuron the coincidence detector is
a thin dendritic branch, and a lone input on it is still integrated passively: a small somatic response, not none
(Kandel ch. 13). The point-neuron junction replaced that passive component with zero. Marr's cerebellar theory
states the same requirement at the population level: a conjunctive code must still represent every active input,
so sparse input needs a smaller code (section 4). And at the behavioural level, elemental and configural
predictions coexist (Kandel ch. 52).

**The mechanism (AMENDMENT 3).** Beside the junction edge, the two frame afferents each junction reads
(FR(-1,a), FR(+1,b)) also project straight to the category pools through an ELEMENTAL edge. It uses v2's own
frame->category synapse constants (start weight, Oja rate and normalisation), unscaled, and learns jointly with the
junction edge under the same teacher curriculum. The junction edge keeps its drive-matching boost (S = 165.1,
because junctions fire sparsely); the elemental edge gets none (its afferents are not sparse). So the elemental
vote is the unamplified, weaker component, and its weight relative to the conjunction is fixed by constants that
were already frozen. Nothing new is tuned.

**What the dev check measured (seeds 7 and 42).** The "weaker" premise did not hold. On the queried battery words
the elemental edge carried most of the afferent drive (81% and 68%). The boost S was measured on untrained
curriculum presentations, and after training the junctions carry little on these words. The elemental vote fixed
recall and silence, and it also re-admitted v2's closed-class and adjective errors. The companion process this
points to next is a shared synaptic budget: one normalisation per postsynaptic neuron across both edges, instead of
one per edge. Heterosynaptic competition would then move weight toward whichever input predicts the cell's
activity.

**What this does not claim.** A biological ratio of passive to NMDA-amplified somatic response is not derived or
matched: the relative weight is inherited from the two pathways' existing constants. The two routes are two
populations in a point-neuron circuit, not two integration modes of one dendrite. The elemental vote is not gated:
it also votes on complete frames, where it re-adds some of the single-offset evidence the conjunction replaced.
That is the precision risk AMENDMENT 3's dev check measures. Marr's gated alternative (a Golgi-like regulator that
silences the single-input code whenever the full conjunction is available) is recorded as the next rung if this
costs precision.
