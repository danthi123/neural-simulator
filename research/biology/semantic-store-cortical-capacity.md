---
type: biology
id: semantic-store-cortical-capacity
mechanism: The neocortex is the LARGE-CAPACITY long-term repository of semantic facts -- many bound role-filler associations coexist as distributed traces; the hippocampal DG-CA3 index provides content-addressable retrieval over them (loading is the cortical store's capacity half; retrieval is the index's half)
status: established
last_verified: 2026-08-26
current_finding: research/findings/raw/_bulk_kb_load_smoke.json
current_status: "LOADING half of the knowledge-scale crux (#65): bulk-load structured agent-action-patient triples into the FHRR fact store as the composer's phasor bindings. Each fact is its OWN 3-role composite (distributed trace), so per-block recall integrity does not degrade with N. De-risk gate: patient/agent top-1 >= 0.95, moat 0 new confab (out-of-store abstains), practical bulk throughput (>=1000 f/s), and the vectorized closed-form encode reproduces the spiking composer's recall on a cross-check subsample. 1-seed numpy SMOKE (N=5000): patient@1=agent@1=1.0, moat 0/1500 confab, xcheck=1.0, bulk ~12k f/s, faithful spiking ~32 f/s, shuffle recall 0.0 vs chance 5e-4 -> GO; current_finding points at the smoke, the 6-seed N=100k pool sweep (research/findings/raw/_bulk_kb_load_6seed.json) is PENDING. The faithful per-op spiking resonate encode is the honest cost; production batches it on GPU. Pairs with dg-ca3-sparse-index (the sublinear RETRIEVAL half, already 6-seed GO)."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "serve as the long-term repository of the separate ele"
    note: "the defining claim -- the CORTICAL regions, not the hippocampus, are the long-term repository of the separate elements of a memory. The cortex is the large-capacity semantic store the bulk KB loads INTO; the hippocampal index only routes retrieval to it."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "pattern separation results from the divergence"
    note: "the retrieval companion (dg-ca3-sparse-index): DG pattern separation over the divergent EC->granule expansion is what makes content-addressable lookup over the large cortical store sublinear -- loading (this entry) + indexed retrieval (that entry) are the two halves of knowledge scale."
implemented_by:
  - research/runners/_bulk_kb_load_derisk.py
findings:
  - research/findings/raw/_bulk_kb_load_6seed.json
---

# The cortex is the large-capacity semantic repository; loading is its capacity half

**The claim the code must respect.** Systems-consolidation neuroscience places the long-term store in the
neocortex, not the hippocampus: the cortical regions "serve as the long-term repository of the separate elements"
of a memory (Kandel). The hippocampal DG-CA3 circuit is an INDEX -- it binds an episode during encoding and routes
a cue to the cortical trace during retrieval (`research/biology/dg-ca3-sparse-index.md`), but it is not where the
bulk of knowledge lives. Knowledge SCALE therefore has two separable halves: (1) the cortical store must HOLD a
large number of facts without the individual traces corrupting each other (LOADING -- this entry), and (2) a cue
must reach the right trace without scanning all of them (RETRIEVAL -- the DG-index entry, already 6-seed GO).

**How the fact store realizes the loading half.** Each fact is encoded as the composer's FHRR phasor binding: a
BUNDLE of role-filler binds, `composite = sum_r exp(2*pi*i*(role_phase_r + filler_phase_r))`, one composite PER
fact (a distributed trace, not a slot). Because facts are separate composites rather than superposed into one
shared vector, per-fact decode integrity (unbind a role -> matched-filter cleanup) is independent of how many other
facts are stored -- the loading risk is throughput and that the loaded bindings stay recallable, NOT capacity
collapse. The bulk loader evaluates the closed-form FHRR algebra (phase-add bind + phasor-sum bundle) the composer's
per-op spiking resonate CONVERGES to, cross-checked to reproduce the spiking composer's recall answers.

## What is established, and where the shortcut stands

**De-risk (`_bulk_kb_load_derisk.py`, 6 seeds):** bulk-load N=100k synthetic agent-action-patient triples at
production D=512; patient/agent top-1 recall on a sample, out-of-store cues abstain with 0 new confabulation,
shuffled triples collapse recall to chance, and the vectorized bulk encode reproduces the spiking composer's
query_patient answers on the cross-check subsample.

**Declared shortcut, and the burn-down.** The bulk encode is the CLOSED-FORM FHRR algebra, computed vectorized on
the host, NOT the per-op spiking resonate. It is verified equal to the spiking composer's recall (the resonate
converges to this fixed point; measured max circular phase error ~0.09, recall-identical), so it is the same
mechanism at the practical operating point -- but the FAITHFUL biological encode is the spiking bind+bundle
resonate, whose burn-down is the GPU-batched resonate (a block-diagonal encode over many facts in one launch, the
same batching the DG-index query already uses for the scan). This entry certifies the store's CAPACITY to hold the
loaded knowledge; the faithful per-fact write cost is reported separately and is the named production optimization.

## What this entry cannot catch

No `constraints_config`: the load-bearing properties are inequalities (recall >= 0.95; D large enough that a stored
match's cleanup score dominates a random distractor) and a call-graph shape (the bulk encode must use the composer's
EXACT concept+role codes and the decode must be the matched-filter cleanup, not a lookup by the ground-truth id).
Both live as runner gates: the cross-check-vs-spiking agreement is a hard gate, the shuffle control must collapse to
chance, and the out-of-store cues must abstain with 0 confab.
