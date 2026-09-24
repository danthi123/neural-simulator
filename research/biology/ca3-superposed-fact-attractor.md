---
type: biology
id: ca3-superposed-fact-attractor
mechanism: Facts are stored one-shot by a local covariance-Hebbian rule into ONE shared, diluted CA3 recurrent matrix (plus shared perforant and CA3->EC readout matrices); a partial cue (agent + relation) is completed by the recurrent attractor under gamma-cycle inhibition, and capacity is set by synapses per cell and code sparseness, not by a per-fact block
status: proposed
last_verified: 2026-09-23
current_status: "PRE-REGISTERED, not yet measured. Default-off research runner research/runners/ca3_superposed_fact_attractor.py; 6-seed capacity grid staged (see research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md). Retires the localist premise of semantic-store-cortical-capacity (per-fact composites cannot show a capacity law)."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "activity patterns are stored as changes in connections"
    note: "Marr's CA3 proposal as Kandel states it: a memory is stored as changes in the connections BETWEEN active CA3 cells -- every memory in the same recurrent matrix, which is what makes interference (and a capacity) intrinsic."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "referred to as pattern completion."
    note: "retrieval from a partial cue: reactivating a subset of the stored assembly reactivates the whole ensemble through the recurrent connections -- the read this runner uses (agent + relation completes the patient)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "tive loss of LTP at the recurrent synapses between"
    note: "the lesion the runner mirrors: CA3-specific NMDA-receptor deletion removes LTP only at the recurrent synapses, and with it completion from partial cues -- hence the rec_zero / rec_shuffle arms, which disable only the recurrent edge."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "the decimal order of the number of converging synapses on a single cell"
    note: "the capacity law in words: with static connectivity an autoassociator's capacity is set by the order of the number of converging synapses per cell -- the G4 prediction (double c_rec, roughly double capacity)."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "be stored in the rat hippocampus would be tens of thousands"
    note: "the scale anchor for the owner question: a rat hippocampus holds tens of thousands of memories in its fast store; LLM-scale fact counts need the slow cortical store filled by consolidation."
  - path: doi:10.3389/fncel.2013.00098
    anchor: "p_max ~ k C_RC / (a ln(1/a)), k ~ 0.2-0.3; ~46 mossy-fibre synapses per CA3 cell for storage; heterosynaptic LTD lets new memories overwrite old"
    note: "Rolls 2013 (Front Cell Neurosci 7:98), full text read in the 2026-09-23 research round: the capacity formula, the sparse mossy detonator, and the forgetting requirement the bounded arm implements."
constraints_config:
  - key: CA3_SPARSENESS
    value: 0.01
    why: "Capacity rises as 1/(a ln(1/a)) (Tsodyks & Feigel'man 1988; Rolls 2013) and only ~a^2 of synapses change per memory (Fusi 2021). A dense CA3 code (the dense_nodg control arm, a=0.05) is the no-companion baseline, never the default."
  - key: MOSSY_FANIN
    value: 46
    why: "Mossy fibres are a sparse, strong input (~46 per CA3 cell, Rolls 2013) that selects the CA3 cells used for storage; recall runs over the weaker, denser perforant path instead."
implemented_by:
  - research/runners/ca3_superposed_fact_attractor.py
---

# One shared CA3 matrix stores every fact; capacity is a law, not a constant

**The claim the code must respect.** A hippocampal memory is stored "as changes in connections between active CA3
cells" (Kandel, after Marr 1971). Every memory lands in the SAME recurrent matrix, and a subset of the stored
assembly completes the rest. Interference between memories is therefore intrinsic, and the number of memories the
network can hold follows a law set by the number of recurrent synapses per cell, C, and the code sparseness, a:
p_max ~ k C / (a ln(1/a)) with k ~ 0.2-0.3 (Treves & Rolls 1991, restated in Rolls 2013). Buzsaki gives the same
statement in words ("the decimal order of the number of converging synapses on a single cell").

**What the localist store replaced with a constant.** `semantic-store-cortical-capacity` stores each fact as its own
composite, so the storage shared between facts is zero and recall cannot degrade with N. That is a design choice,
not a property of the brain; this entry is the superposed replacement the capacity instrument needs.

**Companion processes the code must include, not proxy with a bound.**
- Sparse coding through dentate-gyrus pattern separation (a fixed EC->DG expansion under gamma-cycle inhibition,
  then the sparse mossy detonator). This is what lowers a and decorrelates correlated facts.
- The covariance threshold is each cell's OWN running mean rate (a homeostatic sliding threshold), implemented as
  Welford's online covariance.
- For continual writing, bounded synapses need small or stochastic steps plus heterosynaptic LTD, so the store
  forgets the oldest memories gracefully (a palimpsest) instead of collapsing (Amit & Fusi 1994; Fusi 2021). The
  D6 store's W_MAX slam (every write saturates every synapse, q = 1) is the opposite regime.

**What this entry cannot catch.** The runner's gamma-cycle binary discretization (k-WTA per cycle) is a declared
abstraction of spiking inside a cycle and is not cross-checked against the full Izhikevich bridge here. The fixed
mossy and EC->DG topologies are developmental wiring drawn from the seed, not self-organized. The fast store alone
is expected to reach rat-hippocampus scale (1e4-1e5 facts on one GPU by extrapolation), not LLM scale: that needs
the slow cortical store filled by interleaved replay (`cls-interleaved-consolidation`).
