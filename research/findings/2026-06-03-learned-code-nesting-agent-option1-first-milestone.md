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
40. **Per-kind breakdown (3 seeds, 72 facts) localizes the cost precisely:**

| patient kind | learned-code accuracy |
|---|---|
| flat | 16/19 = 0.84 |
| one-attribute (F=2 resonator) | **12/12 = 1.00** |
| two-attribute (F=3 resonator) | **10/18 = 0.56** |
| embedded clause | 21/23 = 0.91 |

The cost is **almost entirely the two-attribute case** — the F=3 resonator factoring `adj₁ ⊗ adj₂ ⊗ noun`,
where the two adjectives share a codebook (permutation symmetry, restart-residual selection) and **grounded
~10% correlation between the adjective codes** degrades the restart selection. One-attribute (single
resonator factor) is *perfect* on learned codes; flat and clause are strong. So it is a **narrow,
well-characterized boundary** (the hardest construction on correlated codes), not broad degradation. A
mis-recalled word (seed 43, recall 0.97) also cascades into its facts. This is the honest price of
biologically-grounded learned codes vs idealized ones.

**The lever is D, not restarts (measured, isolated two-attribute case):**

| | restarts=16 | restarts=48 |
|---|---|---|
| D=2048 | 0.83 | 0.83 |
| D=4096 | **0.96** | 0.96 |

Doubling the resonator restarts does **nothing** (0.83→0.83) — the issue is not restart selection (finding
the factorization) but the **dimension/SNR floor**: the grounded-correlated adjective codes sit too close at
D=2048. Doubling D (2048→4096) lifts the isolated two-attribute case to **0.96**, exactly as the substrate
D-capacity finding predicts. So the two-attribute gap is closed by dimension, at 2× compute. The agent default
stays D=2048 (the common cases — flat / one-attribute / clause — are already strong there; D=4096 is the knob
for two-attribute-heavy use). A narrow, characterized boundary with a known, measured fix.

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
