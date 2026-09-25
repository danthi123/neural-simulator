---
type: biology
id: homeostatic-scaling-relative-strength
mechanism: Synaptic scaling is a slow, per-neuron homeostat -- over a day or two of changed activity a neuron multiplies ALL of its excitatory synaptic inputs by one factor, each in proportion to its initial strength, so its firing returns toward a set-point while the relative strengths of its inputs (and so the memories they carry) are preserved; it keeps Hebbian learning from saturating a neuron and does not equalize memories
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "plasticity is a circuit mechanism that endeavors to"
    note: "ch.49 (critical-period plasticity after monocular deprivation): homeostatic plasticity is a circuit mechanism that endeavors to maintain a steady level of input to neurons; loss of drive from the closed eye leads to a compensatory increase in drive from the open eye."
  - path: "PMID:9495341"
    anchor: "increases or decreases the strength of all of a neuron's synaptic inputs as a function of activity"
    note: "Turrigiano, Leslie, Desai, Rutherford & Nelson 1998, Nature 391:892 (doi 10.1038/36103), abstract read via PubMed 2026-09-25: the scaling 'apparently affected each synapse in proportion to its initial strength'; after blocking inhibition, amplitudes decreased 'over a 48-hour period'; scaling may help 'stabilizing synaptic strengths during Hebbian modification and facilitating competition between synapses'. Already cited as external context in research/biology/readout-port-homeostasis.md."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# Scaling is per neuron and keeps the ratios; the store's version is per memory and removes them

**The constant this replaces (read in the code, 2026-09-25).** `OneBrainComposer.apply_homeostatic_scaling`
(`research/runners/one_brain_composer.py`), on by default through `BRAIN_DA_ENCODING` and
`BRAIN_DA_ENCODING_SUBSTRATE`, runs on an idle tick after new facts were stored
(`webapp/continuous_engine.py` `consolidate_substrate_homeostasis`). It senses each block's readout activity and
rescales that block alone: a weak engram up to the set-point (at most x4), a strong one part of the way down
(`ratio ** 0.25`, at least x0.34). Each block owns its own D readout units (trig+1..trig+D), so a rule that is
per-neuron in the tissue becomes per-memory here: it removes every importance difference below the set-point and
compresses the ones above it (the chat's per-write gain is floored at the set-point, so today the compression of
strongly written, important facts is the live effect). It also runs on the first idle tick after a write, where
the tissue's scaling develops over tens of hours.

**What the real system runs.** One multiplicative factor per neuron over all of its inputs, proportional to each
synapse's strength (Turrigiano 1998), so a neuron that carries several memories keeps their order while its total
drive is regulated (Kandel ch.49). Memories share neurons; that sharing is what makes the per-neuron rule
order-preserving.

## How the code is expected to bind to it (design only, nothing implemented)

Design doc section 3 (how the pass interacts with the ledger and the gates, and the arms it stays on in) and Step 2(c):
once Step 2 allocates facts to compartments whose units they share, the homeostat senses the compartment's units and
scales all of their engrams by one factor. Until then the per-engram pass stays on in every arm (production default)
and a REPORTED lesion arm measures what it costs the what-matters gates. No `constraints_config`.

⚠️ **Provenance honesty.** The Kandel anchor is in the local corpus. Turrigiano et al. 1998 was read as an abstract
through PubMed on 2026-09-25.
