# Integrated-loop hypothesis: cheap simulation confirms the core claim; per-helper necessity needs the full spiking model

## Plain-language summary

The project goal is a working local proof-of-concept of compositional
memory (binding several role-filler pairs, holding them, and answering
a query about a novel combination), built the way real brains do it.

Every earlier attempt this session tested one mechanism in isolation
and found it insufficient. The project owner's key correction: in real
brains this capability does not live in any single mechanism — it
emerges from several systems running together as one loop. "Each part
insufficient alone" is the *expected* signature of an emergent
capability, not proof the approach is unfit.

We grounded the loop in the project's own reference catalog rather
than outside sources:

- **Hagoort's Memory–Unification–Control model** of language. The
  project already has the Memory part (a lexical store) and the
  Control part (a prefrontal working-memory region) but has never
  built the **Unification** part — the combinatorial binding step
  (Broca's area) that actually composes items. (Hagoort 2014.)
- The reference catalog's proposal (Lisman & Idiart, via Buzsáki) that
  prefrontal working memory and hippocampal episodic encoding are not
  two separate systems but **one shared theta–gamma timing rhythm**:
  each gamma sub-cycle holds one item; the theta period sets the
  buffer; shifting the item sequence across theta cycles writes it to
  episodic memory.
- The catalog's note that the brain's cortex–basal-ganglia–thalamus
  loop is five parallel loops, of which the project only ever built
  the *motor* one — composition needs the *prefrontal* loop.
- Hippocampal relational binding (Eichenbaum–Cohen), replay-driven
  consolidation, and theta-paced sequence compression.

## What the cheap simulation tested and found

A small, fast, pure-NumPy simulation of this loop's logic (not the
full spiking network), 5 random seeds, at 2, 4 and 8 simultaneously
bound pairs. We removed one system at a time — the standard
lesion/necessity method, which the catalog itself uses (the
slow-oscillation→spindle→ripple nesting evidence). Three rounds of
catalog-guided fidelity corrections were applied transparently; the
numeric success thresholds were fixed in advance and never changed.

**Core result (robust, the load-bearing finding):** the full
integrated loop performs the task perfectly at every load. Removing
any one of the **three shared systems** — the binding operation, the
shared theta–gamma clock, or the fast hippocampal store — collapses
*both* the working-memory query and the episodic-sequence recall
*together*, at every load. This is exactly the catalog's central
prediction: these systems are not separable, and the capability
exists only in the integrated whole. This directly supports the
owner's reframe and held up across all three iterations.

**Honest limitation:** two helper systems (the selective gate and the
sequence-compression step) become cleanly necessary only at realistic
load (4–8 items), not at the near-trivial 2-item case; and the
replay/consolidation helper does not become strictly necessary in
this simplified algebra at all. This is a limitation of the
*simplification* — a 2-item compose is trivially easy and a
nearest-match readout over a handful of items is robust — not
evidence against the hypothesis. Forcing these to register as
necessary would require contriving the toy, which the project's
anti-overfitting discipline forbids.

## Decision and next step

The cheap simulation has done its designed job: it confirmed the
load-bearing core (the capability is non-separable from the integrated
loop) and showed that per-helper necessity cannot be settled at this
simplified tier. The project's pre-registered plan always specified
the **full spiking-network model as the decisive test**; the cheap
simulation only de-risks the core logic and screens for fatal flaws
(none found — the core signature is strong).

Therefore: do not keep tuning the toy (the residual is a
simplification limit, not a fidelity bug; further tuning would be
result-chasing). Do not set this aside, and do not defer the decision
upward. Proceed to build the full spiking-network version of the
integrated loop — reusing the project's already-validated subsystems
unchanged — where each system's necessity can be tested faithfully,
carrying forward the confirmed core finding.

## What this does and does not claim

- It **does** show: in a faithful abstraction of the catalog's loop,
  compositional memory is an emergent property of the integration —
  remove any shared system and it is gone.
- It does **not** claim: fluent open-ended language, an LLM-class
  model, or that the full spiking build will succeed. Those remain
  open and are the point of the next phase.
- All previously validated results and the project's honest
  boundaries are unaffected. The trustworthy "answer only when
  grounded" memory component and the validated subsystems remain
  intact and unchanged.

## References (project catalog)

- Hagoort, P. (2014). Nodes and networks in the neural architecture
  for language. *Curr. Opin. Neurobiol.* 28:136–141. (catalog G.21)
- Lisman & Idiart theta–gamma multiplex; Buzsáki (2006) *Rhythms of
  the Brain*, Cycle 12. (catalog N.16 supplemental)
- Eichenbaum–Cohen relational memory. (catalog D.01/D.02)
- Alexander/DeLong parallel cortico-basal-ganglia-thalamic loops;
  Kandel 6e Ch 38. (catalog A.05/A.06)
- Theta-paced sequence compression; Buzsáki Cycle 11. (catalog D.24)

## Files

- Recorded evidence: `research/findings/raw/q5_integration_probe_recorded.txt`
- Design: `docs/plans/2026-05-18-Q5-integrated-biology-grounded-closed-loop-design.md`
