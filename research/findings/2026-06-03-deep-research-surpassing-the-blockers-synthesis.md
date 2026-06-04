# Deep research: how to surpass the composition + conversation blockers — synthesis + action plan — 2026-06-03

Three parallel deep-research agents (compositional scaling in VSA/brain; thalamocortical dynamical gating;
broader untried options) returned an unusually actionable, citation-backed haul. This synthesizes their
findings into a prioritized, cheap-first action plan for Fork 1 (advance the biology-faithful composition
frontier). Two of the three top recommendations attack walls this session *just characterized*.

## The two blockers, restated

1. **Composition-at-scale** — at 320 concepts, two-attribute (F=3 resonator) and recursive-clause decoding
   collapse to ~0. A probe this session proved it is **capacity/dimension** (random codes fail identically to
   grounded), not correlation: the resonator's factoring capacity is exceeded by the 200-concept codebook at
   D=2048.
2. **Fluent generation** — surrogate-grad BPTT spiking LMs overfit at scale (~33,000× worse perplexity than a
   small transformer). Documented dead-end; the field's fix is ANN→SNN conversion.

## Track 1 — Composition-at-scale fixes (Agent A). The wall has documented cheap fixes; D is NOT the lever.

The key correction to my own conclusion: the resonator literature says the fix is **changing the update rule
or the code substrate**, which buys 50×–5,000× capacity *at fixed D* — far cheaper than scaling D.

| Rank | Fix | Mechanism | Predicted gain | LOC | Source |
|---|---|---|---|---|---|
| 1 | **Softmax-attention resonator** | replace `sgn(X Xᴴ z)` cleanup with `X·softmax(β·Re[Xᴴ z]/D)` | FHRR F=3 @ M≈5000: ~0% → **85–95%**, ~5× faster convergence | ~20 | arXiv 2403.13218 |
| 2 | **Iterative noise + threshold** | add Gaussian σ≈0.007 to the similarity each iter; zero sub-T≈0.1 entries | ≥50×, up to 10⁴× capacity; escapes limit cycles | ~10 | arXiv 2412.00354 (IMF) |
| 3 | **Decorrelate codebook** (ZCA / residue codes) | whiten the codebook before binding (= dentate-gyrus pattern separation) | restores convergence under correlation | ~15 | Frontiers 2026; Kymn NeCo 2024 |
| 4 | **ACF asymmetric reconstruction codebook** | distinct bit-flip mask per same-codebook slot — kills the repeated-factor ("problem of 2") limit cycle | >10⁹ @ F=2 | ~10 | arXiv 2412.00354 (ACF) |
| 5 | **CSim cleanup between recursive unbind hops** | phase-coupled least-circular-distance cleanup per hop | failure 20%→5% after repeated unbind | ~30 | arXiv 2412.00488 |
| 6 | **Sparse block codes** (L∞ + threshold + sampling) | one-hot-per-block codes; cortical/columnar sparsity | **~5,000×** capacity (10³→5×10⁶) | substrate change | arXiv 2303.13957 |
| 7 | **Partitioned / hierarchical resonator** | bind each filler with a per-slot role vector first (separate codebooks per role) | removes repeated-factor degeneracy structurally | ~25 | Renner Nat.MI 2024; Frankland-Greene |

Also surfaced: resonator capacity is *highest at F=3/F=4* and *lowest + most limit-cycle-prone at F=2* (the
repeated-factor "problem of 2"); the brain composes via **partitioned role subspaces** (Frankland-Greene
lmSTC), **routing not superposition** (Neural Blackboard Architecture, van der Velde), and **assembly
pointers** (Papadimitriou-Vempala-Maass). Items 1–4 are each a few-dozen-line numpy test against our existing
320-code phasor algebra. **Test softmax (1) + whitening (3) first — they attack the two distinct root causes
and compose.**

## Track 2 — Thalamocortical dynamical gating (Agent B). The deep diagnosis of "compose-pathways went silent."

This is the most important *mechanistic* finding. Our compositional pathways (`verb_pool_X → motor_Y`, grown
by STDP from zero-init) "went silent" because **an additive grown weight that must be silent-when-unbound and
strong-when-bound is exactly what STDP cannot reach from zero** — a vanishing-signal cold start. The
biology-faithful alternative (Logiaco-Abbott-Escola 2021, *Cell Reports*) is fundamentally different:

- The thalamus does **not learn** the binding. It **transiently reconfigures the cortical recurrent dynamics**
  via a **low-rank, multiplicative gate** on the effective connectivity:
  **`J_eff = J_cc + J_ct · S · J_tc`** (rank-one per binding: `J_cc + u·vᵀ`), where `S` is a BG-controlled
  diagonal disinhibition gate. Theory: Mastrogiuseppe & Ostojic 2018 (a rank-one term *sets* a recurrent
  network's computation).
- **Binding becomes state selection (which gate is open), not synaptic storage (which weight grew).** This (a)
  removes the STDP cold-start, (b) gives true **variable binding** (the *same* role circuit binds to
  *different* fillers on different occasions — a constant weight cannot represent this), (c) avoids
  **catastrophic interference / forgetting** (orthogonal thalamic subpopulations; "the synapse doesn't change —
  the effective connectivity landscape does", TiCS 2024), and (d) is **one-shot and reversible** (ms
  disinhibition, no training pass).

**Our `g11_bg` cascade already has the skeleton:** `gpi_X → thal_X → cortex_X` (lines ~1173, ~1279),
`cortex_X → stn` hyperdirect. The **one missing primitive** is a **per-pathway multiplicative transmission
gate** driven by thalamic activity — our neuromodulator subsystem has additive `excitability_drive` and
scalar `synaptic_gain` but no per-pathway multiplicative gate (CLAUDE.md even notes the plasticity gates
don't gate *current* — that complementary primitive is what's needed). Spaun/Nengo already uses the spiking
BG→thalamus→cortex loop as a *gated router* for VSA binding (Stewart-Choo-Eliasmith 2010) — the precedent.

**Cheap-first test (H1, ~1 day):** a 2-role × 2-filler reduced model on the NumPy backend; 4 structurally
pre-wired, normally-CLOSED routes (`R1→F1,…`); 4 thalamic gate pools each disinhibited by a `gpi` pool.
Decisive: bind(R1,F1) → R1 drives F1; **re-bind(R1,F2) with zero retraining** → R1 now drives F2 (a grown-weight
control cannot, without overwriting). Score with the existing permuted-mapping anti-cheat: gating should give
TRUE mapping rank 1/N deterministically *for every binding on command*, vs the seed-dependent ≤3/24 grown
weights gave. If H1 passes, escalate to the full low-rank `J_cc + Σ s_k(t)·u_k vₖᵀ` gate for *sequencing*.

Sources: Logiaco-Abbott-Escola 2021 (Cell Reports / arXiv 2006.13332); Kao-Jensen 2024 (Cell Reports);
Mastrogiuseppe-Ostojic 2018 (Neuron); Rikhye-Gilra-Halassa 2018 (Nat Neuro); Schmitt-Halassa 2017 (Nature);
Mukherjee-Hage-Halassa 2024 (TiCS); Bouton 2025 (gain modulation without relearning); Stewart-Choo-Eliasmith
2010 (Spaun BG-thal-cortex VSA router).

## Track 3 — Untried conversational options (Agent C). Reframe the wall; combine assembly-generation + grammar.

| Rank | Option | Goal-fit | 3090 | Buys | Honest limit |
|---|---|---|---|---|---|
| 1 | **Assembly Calculus / NEMO language** (our architecture's TWIN) | ★★★★★ | ★★★★★ | grounded SVO generation + word order, **no backprop**, *and a cited biological reason for our wall* | telegraphic, ≤20–40-token sequences; not fluent |
| 2 | **Grammar/template NLG over VSA composition** | ★★★★★ | ★★★★★ | controllable, hallucination-free conversation; literature: beats end-to-end on grounding + generalization | templated feel; coverage = grammar effort |
| 3 | **Neuro-symbolic hybrid** (spiking VSA cognition + tiny labeled realizer; Spaun pattern) | ★★★★☆ | ★★★★☆ | preserves no-confab thesis; reuses the stack | must carve "biological cognition vs learned articulation" |
| 4 | **Spiking state-space model cell** (SpikingSSM: PPL 33.94@75M beats SpikeGPT 39.75@213M) | ★★★☆☆ | ★★★★☆ | the only direct-trained spiking family that narrows the ANN gap; parallel-train sidesteps BPTT overfit | still gradient-trained; SSM core not brain-like |
| 5 | **ANN→SNN conversion** (SpikeZIP-TF → LAS, near-lossless) as a *labeled secondary* track | ★★☆☆☆ | ★★★★★ | fluent spiking *inference*; bounds the gap | weights are backprop-trained |
| — | Forward-Forward for sequences; predictive-coding-as-LM | ★★☆☆☆ | — | **dead-ends** (don't scale / fall short of backprop) | — |

**Strategic synthesis (Agent C):** the Assembly-Calculus language model (Mitropolsky-Papadimitriou 2025,
`github.com/dmitropolsky/assemblies`) is mechanism-for-mechanism our simulator (assemblies = concept pools;
k-cap = FS-WTA; Hebbian = our STDP; brain areas = regions; no backprop) and it reaches the **same** honest
ceiling we found (correct word order, not fluent; 20–40 token limit) — **turning our generation wall into a
predicted, citable biological boundary.** Combine #1 (assembly-based grounded generation) + #2 (grammar over
VSA composition) → a genuinely biology-faithful conversational artifact whose boundary is documented, with the
no-confabulation guarantee *preserved*. Run #4 (spiking-SSM) in parallel as the capability yardstick; #5
(conversion) as the clearly-labeled fluency-ceiling reference.

## Prioritized action plan (Fork 1, cheap-first)

1. **NOW — softmax-attention resonator** vs the 320 F=3 wall (Track 1 #1). ~20 LOC, decisive. If it lifts
   two-attribute/clause from 0 → high at fixed D, the composition-at-scale wall is surpassed cheaply.
   *(In flight.)*
2. **NEXT — thalamocortical multiplicative gate, cheap-first H1** (Track 2). The 2-role×2-filler reduced
   model; if gating gives deterministic variable binding where grown weights were seed-fragile, it's the deep
   lever for compositional binding (and addresses forgetting + one-shot rebinding). ~1 day.
3. **THEN — Assembly-Calculus reframe + grammar-over-composition** (Track 3 #1+#2) as the conversational
   artifact whose boundary is the documented assembly limit. Clone `dmitropolsky/assemblies` as a reference
   oracle; extend our concept-grammar to systematic generation over VSA roles.
4. **Parallel capability yardstick** — a spiking-SSM cell (Track 3 #4) on the existing char setup, measuring
   the train/test gap vs surrogate-grad BPTT, to quantify how far direct-trained spiking can be pushed.

Every item is cheap-first and biology-faithful (1–3) or a clearly-labeled capability bound (4). The walls are
the field's walls (confidence signal), and the field's own fixes are now mapped to our exact code.

## Cheap-first results already in (this session) — the 320 two-attribute wall is RE-LOCALIZED

Testing Track-1 #1 immediately corrected the diagnosis:

- **Softmax-attention resonator: NEGATIVE on our codes.** On the *clean* F=3 product at 320-concept codebook
  scale (adj M=60, noun M=200, D=2048), the **standard** resonator already recovers **0.96**; **softmax made
  it worse (0.21)**. The attention/noise tricks help *overloaded* resonators; our well-conditioned FHRR codes
  are not capacity-limited on the clean product — consistent with the earlier finding that Kymn-style noise
  injection didn't replicate on our codes. So **the resonator is not the 320 blocker.**
- **The real 320 blocker is BUNDLE CROSSTALK, not resonator capacity.** A decode trace shows the depth
  detection is *correct* (clause-detect 0.051<0.12 ✓, flat 0.055<0.30 ✓, residual 0.381<0.50 → routes to the
  3-factor resonator ✓) but the resonator returns garbage — because it operates on `p = unbind(bundle,
  PATIENT)` = the clean two-attribute product **plus** the agent+action role-bindings. At a 200-noun codebook
  the crosstalk drowns the signal (the product is only ~1/√3 of `p`); at the ≤5-codebook 40-concept scale the
  resonator tolerated it, at 200 it cannot.
- **Candidate fix (cheap, principled): crosstalk subtraction ("explain away" the known components).** The
  agent *already decodes* the agent and action to match the query, so it can subtract those role-bindings
  from the bundle before unbinding the patient, leaving the clean product the resonator handles at 0.96.
  *(In flight.)* This is more direct than the literature's capacity tricks because our problem was never
  capacity — it was crosstalk from composing a multi-role fact.

Lesson reinforced: cheap-first testing the literature's #1 fix against the *actual* wall (not the assumed one)
re-localized the blocker in minutes and pointed at a simpler, structure-specific fix.

**Track 1 outcome — the 320 composition wall is SURPASSED.** Crosstalk subtraction took the full 320-concept
agent from ~48% to **100%** (60 mixed nested facts × 2 seeds = 120/120; two-attribute and clause from 0/x to
16/16, 17/17, 18/18, 11/11), at the default D=2048, 38 tests green. Shipped in `nested_composition_agent.py`.

**Track 2 cheap-first H1 — RESOLVES at toy scale (gate-keeper passed).** `_thalamocortical_gating_H1_probe.py`
(4 roles × 4 fillers, re-binding protocol, 3 seeds): a multiplicative gate gives latest-binding accuracy
**1.000**; grown static weights only **0.695** (the first binding's weight persists — they cannot re-bind on
command). Honest caveat: this is a near-tautological principle-check (the gate reflects the command by
construction) — it is the *gate-keeper*, not the proof. The genuinely decisive test is integrating a
**per-pathway multiplicative transmission gate into the spiking `g11_bg` cascade** (the cascade already has the
`gpi→thal→cortex` skeleton; the neuromodulator subsystem has only additive drive + scalar gain) and testing it
on the actual compose problem (the v16 `verb→motor` binding that "went silent" with grown weights). That is a
multi-hour bridge build — the next big, owner-steerable step.

**Track 2 — the gate primitive is now SHIPPED + validated in spikes.** Beyond the toy H1, the per-pathway
multiplicative **transmission gate** is implemented in the spiking bridge (`RegionPathway.transmission_gate`
+ `bridge.set_transmission_gate`; `cp_transmission_gain` scales effective synaptic CURRENT in the step,
mirroring the plasticity-gate machinery; the complement CLAUDE.md flagged as unimplemented). Validated in
genuine spiking dynamics (`tests/test_transmission_gate.py`, 4 tests): a closed gate leaves the target SILENT
(0.000, no current despite a non-zero weight); opening it drives the target (0.30); and **re-binding** (close
route A→B, open A→C) reroutes the same source to a different target with **zero weight change** (sum|W|
unchanged) — the thalamocortical hypothesis in spikes, where grown weights could not re-bind. Regression-clean
(53 core CPU tests). **Next:** apply it to the actual v16 `verb→motor` compose problem — pre-wire the routes
fixed + gated-closed, open the (verb, motor) gate to bind, and test whether gated routing binds go→north etc.
where STDP-grown weights "went silent" (the 5/20 seed-fragile result). Then Option C (low-rank `J_cc + Σ
s_k u_k vₖᵀ`) for sequencing.

The Track-3 (assembly-generation + grammar-over-composition) lever remains the conversational-artifact bet.
