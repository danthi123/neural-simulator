---
type: biology
id: dendritic-plateau-coincidence-burst
mechanism: Distal (long-range) + proximal (local) coincidence in a pyramidal dendrite triggers a plateau potential, whose somatic read-out is a BURST
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-25-gap4-forward-representability-SURPASSED-ON-BRIDGE-coincidence-plateau-reliable-expander-6seed-GO.md
current_status: "LINEAR DECODABILITY ONLY — scoped 2026-07-31, and it does NOT carry to depth. The expander raises held-out LINEAR decodability (0.611 +/- 0.047, reproducibility 1.000, 6/6 seeds; non-expanding control 0.352, label-shuffle 0.247) and in the same move DESTROYS the task's depth requirement: the 2-hidden ceiling collapses 0.9599 -> 0.6914 and depth-separation is lost in 6/6 seeds (graded read 5/6), so the deep-credit arms sit at or BELOW their own 1-hidden floor there. Measured with the graded codon as a control, so it is the EXPANSION, not the binarization. The inference that this gives deep credit 'features to shape' is RETIRED; the CREDIT half remains a POWERED NO-GO."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "from more distant brain areas arrive at the distal tips"
    note: "the anatomical segregation the whole credit story rests on: LONG-RANGE input lands distal, LOCAL input lands proximal"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "when a distal stimulus is paired with"
    note: "the plateau requires the CONJUNCTION -- distal alone produces only a very small somatic response"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "of dendritic spike called a plateau potential, which"
    note: "names the event and its ionic basis (voltage-gated Ca2+ channels + NMDA receptors); Kandel Fig 13-17C, adapted from Larkum et al. 1999"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "three or more spikes at rates as high as 100 Hz"
    note: "the somatic read-out of a plateau is a BURST -- the quantity a burst-multiplexed credit signal would carry"
constants:
  burst_min_spikes: 3
  burst_rate_hz_upto: 100
implemented_by:
  - research/runners/_gap4_plateau_expander_probe.py
  - research/runners/_gap4_credit_on_expanded_forward_derisk.py
findings:
  - research/findings/2026-07-25-gap4-forward-representability-SURPASSED-ON-BRIDGE-coincidence-plateau-reliable-expander-6seed-GO.md
  - research/findings/2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md
---

# The plateau is a CONJUNCTION detector, and its output is a burst

**The claim the code must respect.** A cortical pyramidal neuron receives *long-range* input on its distal
(apical) dendrite and *local* input proximally. Distal input alone "usually produce[s] only a very small
depolarizing response at the soma." Paired with a proximal stimulus, the backpropagating spike summates with the
distal EPSP into a **plateau potential**, and when that plateau reaches the soma it triggers a **burst of three or
more spikes at up to 100 Hz**. Two input streams in, one qualitatively different output signal out. That is the
biological warrant for using top-down/bottom-up coincidence as a credit-carrying signal — it is a *structural*
property of the neuron, not something a learning rule has to discover.

**Why this row is tagged with it.** gap#4 is "deep multi-layer directed credit for accuracy". Every named surpass
in the ledger row — burst-multiplexed credit, the Sacramento self-predicting microcircuit, learned feedback —
assumes this two-compartment conjunction exists to carry the instructive signal.

## What is actually established here, and what is NOT

**Established (6 seeds, `_gap4_plateau_expander_probe.py`):** the coincidence-plateau read used as a fixed
nonlinear EXPANSION lifts held-out linear decodability from the characterized 0.34 boundary to **0.611 ± 0.047 at
reproducibility 1.000** (6/6 seeds 0.556–0.704). The controls are load-bearing: a non-expanding control at the
same input width gives 0.352 (+0.26 attributable to expansion), label-shuffle gives 0.247 (≈ chance), and a
pool-silence lesion degenerates to reproducibility 0.000.

**NOT established:** that credit *learns* on top of it. The credit half is a **POWERED NO-GO** on the sparse
point-neuron forward — the idealized weight-transport ceiling fails to learn to 40 epochs, root-caused to
φ′-vanishing (injected apical 15.7 → 1.97 → 0.01, ~1600× attenuation over depth at E ≈ 0.04) plus a tonic-pinned
hidden representation. `PlateauExpander` is defined in one file and, as of this entry, imported by exactly two:
its own probe and `_gap4_credit_on_expanded_forward_derisk.py`, which was written to run the combination. **No
result from that combination is on the record yet**, and that runner's own header warns its learner is a
one-layer `fit_lin`, so a positive there reads as forward-representability, not as deep credit.

## What this entry cannot catch

No `constraints_config`. Nothing in the two runners is a *numeric* config whose value the biology pins — the
plateau's requirements are structural (two input compartments, a conjunction read), and `biology_check --config`
only compares numeric defaults. A runner could satisfy every number here and still drive one compartment.
