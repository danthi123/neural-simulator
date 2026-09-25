---
type: biology
id: prp-competition-and-locality
mechanism: The proteins that make a potentiation last are limited and local -- tagged synapses compete for them (new strong potentiation is maintained at the expense of older, weaker potentiation), capture works best within one dendritic branch and falls off with distance and time, and synapses differ in how ready they are to potentiate -- so the part of a trace that lasts is a graded fraction set by competition, locality and synaptic heterogeneity, not one all-or-none switch per memory
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Although a single tetanus on its own induces"
    note: "ch.54, Frey & Morris: a weak tetanus alone gives only early LTP but gives late LTP when delivered within 2-3 h of a strong one on the same neurons -- the capture the pair already models (reused anchor, see research/biology/sleep-replay-tag-capture.md)."
  - path: "PMID:9020359"
    anchor: "The synaptic tag decays in less than three hours"
    note: "Frey & Morris 1997, Nature 385:533 (doi 10.1038/385533a0), abstract via PubMed 2026-09-25: the persistence of LTP 'depends not only on local events during its induction, but also on the prior activity of the neuron'."
  - path: "PMID:21170072"
    anchor: "the induction of synaptic potentiation creates only the potential for a lasting change in synaptic efficacy, but not the commitment to such a change"
    note: "Redondo & Morris 2011, Nat Rev Neurosci 12:17 (doi 10.1038/nrn2963), abstract via PubMed 2026-09-25: other activity before or after induction decides whether the change persists; tag setting and LTP expression are dissociable."
  - path: "PMID:15603743"
    anchor: "the induction of additional protein synthesis-dependent long-term potentiation for a given set of postsynaptic neurons occurs at the expense of the maintenance of prior potentiation on an independent pathway"
    note: "Fonseca, Nagerl, Morris & Bonhoeffer 2004, Neuron 44:1011 (doi 10.1016/j.neuron.2004.10.033), abstract via PubMed 2026-09-25: 'competitive maintenance' under limited protein synthesis. The model's PRP pool is never depleted by capture, so nothing competes. Already named as the registered response to an NR NO-GO in the sleep-replay-capture prereg Amendment 7 (branch research/pair-production-path-arms)."
  - path: "PMC3032443"
    anchor: "the efficacy of this facilitation decreases with increasing time between stimulations, increasing distance between stimulated spines and with the spines being on different dendritic branches"
    note: "Govindarajan, Israely, Huang & Tonegawa 2011, Neuron 69:132 (doi 10.1016/j.neuron.2010.12.008), abstract via PubMed 2026-09-25: capture of weak stimulation by strong is branch-local; 'stimulated spines compete for L-LTP expression if stimulated too closely together in time'; late LTP is biased to spines within a branch (clustered plasticity)."
  - path: "PMID:16791146"
    anchor: "local translational enhancement, along with synaptic tagging and capture, facilitates the formation of long-term memory engrams"
    note: "Govindarajan, Kelleher & Tonegawa 2006, Nat Rev Neurosci 7:575 (doi 10.1038/nrn1937), abstract via PubMed 2026-09-25: the clustered-plasticity model -- engrams form through weight changes among synapses within a dendritic branch."
  - path: "PMC3323981"
    anchor: "less than half of the spines in adult hippocampus are primed to undergo plasticity under baseline conditions"
    note: "Kramar et al. 2012, PNAS 109:5121 (doi 10.1073/pnas.1120700109), abstract via PubMed 2026-09-25 (no full text in PMC through the tool): 'intrinsic variability among individual synapses imposes a repetitive presentation requirement for maximizing the percentage of potentiated connections'. Synapses of one trace are not identical, so capture of a trace is a fraction."
  - path: "PMC3992944"
    anchor: "specific mechanisms, such as increases in neuronal excitability and synaptic tagging and capture, determine the exact sites where memories are stored"
    note: "Rogerson et al. 2014, Nat Rev Neurosci 15:157 (doi 10.1038/nrn3667), abstract via PubMed 2026-09-25: neuronal allocation, synaptic tagging and capture, spine clustering and metaplasticity as one family of memory-allocation mechanisms."
  - path: "PMID:25102562"
    anchor: "neuronal memory allocation is based on relative neuronal excitability immediately before training"
    note: "Yiu et al. 2014, Neuron 83:722 (doi 10.1016/j.neuron.2014.07.017), abstract via PubMed 2026-09-25: raising excitability (as CREB does) biases which neurons join a memory trace and enhances memory. Basis for allocating related facts to shared units and for an excitability-carried importance mark."
  - path: "PMC4432479"
    anchor: "were not observed in an immediate memory test or for items strongly encoded before fear conditioning"
    note: "Dunsmoor et al. 2015, Nature 520:345 (doi 10.1038/nature14106), full text via PubMed Central 2026-09-25: the retroactive rescue reached related weak items only -- the behavioural face of branch-local, relatedness-specific capture."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# Capture is a competition for a limited, local resource, over synapses that are not all alike

**The constants this replaces (read in the code, 2026-09-25).**
- `SynapticTagCaptureLedger` (`webapp/da_tag_capture.py`) has ONE cell-wide PRP scalar `p`. Capture reads it and never
  consumes it, so every tagged synapse in the store is captured by any PRP event within its window, however many
  others are captured too. That is why the pair's fake design day (prereg Amendment 7) captured every fact told within
  the hour after any salient telling or SWR bout.
- Each managed block's tag is `h0 = |inc_k|`, and the composer writes a block as `g * zc[k]` with `zc` a unit phasor
  (`research/runners/one_brain_composer.py` `_write_block`), so every synapse of a block carries the SAME tag and the
  SAME late-phase trajectory `z_k`. A fact is captured or lost as one switch. The verify-go review measured the result
  as a step in the replay read R (research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md, branch
  research/pair-verify-go).

**What the tissue runs instead.** Limited PRPs that capture uses up, so new strong potentiation is maintained at the
expense of older weak potentiation (Fonseca 2004); capture that is strongest within a dendritic branch and falls with
distance and time (Govindarajan 2011); synapses that differ in readiness, so one episode potentiates only part of a
trace and a spaced repetition recruits the rest (Kramar 2012); and allocation of a memory to the currently most
excitable neurons (Yiu 2014; Rogerson 2014), which is how related memories come to share units.

## How the code is expected to bind to it (design only, nothing implemented)

Design doc steps 1 and 2: a seeded per-synapse readiness inside each managed block (graded capture), and a PRP supply
that capture consumes and that is shared only among blocks allocated to the same compartment by overlap of their
concept codes. The allocation rule is a declared host step until the store has dendritic structure. No
`constraints_config`: the biology constrains readiness only as a distribution, not a scalar.

⚠️ **Provenance honesty.** The Kandel anchor is in the local corpus. The PMID/PMC sources were read as abstracts through
PubMed on 2026-09-25; Dunsmoor et al. 2015 in full text.
