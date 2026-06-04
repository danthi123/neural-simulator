# Learned-code nesting agent — substrate-unification (option 1) first build milestone — 2026-06-03

**One line:** With every scientific axis de-risked (incl. grounded codes), the first build milestone of the
phasor substrate unification is shipped: a reusable, tested module that **learns** word→phasor-code
associations via spike-timing plasticity, wired into the full nesting agent — so flat facts, resonator-
decoded attributed entities, embedded clauses, Q&A, and abstention all run on **learned** (not constructed)
codes. Demonstrated at 40 concepts with an honest accuracy cost from grounded-code correlation.

This is engineering on a de-risked foundation, not new science. The owner greenlit proceeding ("proceed
autonomously") after the honest framing that **option 1 is spiking-faithful, not "fully biological"** — it
rests on the phasor-binding *hypothesis*, and full membrane/dendritic/grounded-embodied realism is beyond it.

## What shipped

**`research/runners/phasor_associative_memory.py` — `PhasorAssociativeMemory`.** Learns the map from a
grounded sparse word cue (`sim.text_embeddings.vocab_to_drive_pattern`) to a phasor concept code via online
weight-bounded spike-timing plasticity (real synapses, asymmetric STDP kernel, hard saturation — the
mechanism validated in `2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md`). Readout = the
resonate-and-fire phasor-neuron phase. API: `learn`, `recall` (with abstention), `recall_confidence`,
`code`, `bind` / `bundle` / `unbind_cleanup`. 6 tests.

**`research/runners/learned_nesting_demo.py` + an additive `external_codes` hook on `NestedCompositionAgent`.**
The agent can now be built on externally-learned codes (no behavior change when unused; the 21 nesting tests
stay green). The demo trains the memory, extracts the learned codes, and builds the full agent on them.

## Results

**Demo (5 concepts, learned codes):** flat (`dog chase cat`→cat), one-attribute (`bird see (red ball)`→red
ball, resonator-decoded), embedded clause (`dog eat (cat chase river)`→cat chase river, recursive unbinding),
and abstention (`cat hold`→None) all correct — the full compositional capability on learned codes.

**Scale (40 concepts: 20 nouns + 10 verbs + 10 adjectives, all STDP-learned, 24 mixed nested facts):**

| seed | facts correct / 24 | memory recall |
|---|---|---|
| 42 | 20/24 | 1.00 |
| 43 | 17/24 | 0.97 |
| 44 | 22/24 | 0.97 |

≈ 80% on mixed nested facts (flat / one-attribute / two-attribute / embedded clause), memory recall
0.97–1.00.

## Honest accuracy cost — grounded-code correlation

The constructed-code agent reached ~96% on mixed facts at 120 concepts; the learned-code agent is ~80% at
40. The difference is **grounded-code correlation**: learned codes carry the real ~10% word-code overlap
(`vocab_to_drive_pattern`), which makes the **resonator** (factoring `adj ⊗ noun` for attributed entities)
and the depth detection harder than the near-orthogonal constructed codes. The flat and clause paths
(cleanup / recursive unbinding) are more robust to it than the resonator path. A mis-recalled word (seed 43,
recall 0.97) also cascades into the facts that use it. This is a real, documented cost of using biologically-
grounded learned codes rather than idealized ones — not a failure, but the honest price of faithfulness.
Levers if it matters later: higher D, codebook decorrelation, or resonator thresholds tuned for correlated
codes (not chased here — the honest baseline stands).

## Where this sits

Option 1 (substrate unification) now has a working, tested learned-code foundation + the full nesting
capability on it. The remaining work is the **full membrane-level online spike-driven loop** (LIF/resonate-
and-fire membrane ODE + an online spike-pair STDP loop, replacing the closed-form/steady-state readout) and
then production integration + scale to 320. All engineering on de-risked science.

**Standing honest caveat (carry forward):** "spiking-faithful," not "fully biological." The phasor-binding
framework is a biologically-grounded hypothesis (motivated by theta-gamma phase coding), not established
brain mechanism. Never overclaim.

## Verdict

**Option 1 first milestone SHIPPED.** Learned-code memory + full nesting on learned codes, 28 tests, scale
demonstrated to 40 concepts (~80%, honest grounded-correlation cost). The biologically-grounded learned
representation supports the composition the non-invertible production binding cannot.
