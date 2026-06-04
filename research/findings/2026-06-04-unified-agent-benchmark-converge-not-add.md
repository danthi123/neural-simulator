# Unified-agent benchmark — the converge-not-add measurement — 2026-06-04

**One line:** The fragmented composition / memory / abstention pieces are converged into ONE agent and
benchmarked at 320 concepts on a FROZEN conversational test set. Constructed codes: a robust 6-category core at
**100% across 5 seeds** with ONE honest ceiling (clause-in-clause). Grounded STDP codes: raw sensory readout
loses attribute composition (0%) — but **pattern completion (the CA3 autoassociator) RESOLVES it** (full
composition = constructed), anti-cheat-validated, while higher encoder fidelity alone does NOT. The
biology-translatable insight: composition operates on stable concept attractors, not raw sensory input.

## Why this, not another mechanism

The owner-approved strategic read (this arc): the bottleneck is no longer a missing mechanism — it is
FRAGMENTATION (many validated pieces in separate demos) + the absence of an honest end-to-end measurement. Two
framing facts: fluent generation from-scratch biology-faithful is a documented wall (so the deliverable is the
composition/memory/trust half), and phasor FHRR is de-risked as the unified substrate. So the move is CONVERGE:
build ONE coherent agent on ONE substrate and measure it honestly. `NestedCompositionAgent` already unifies
store/compose + who/what Q&A + abstention + dialogue and accepts learned `external_codes`, so the new module is
a BENCHMARK HARNESS, not a new mechanism.

## What shipped

`research/runners/unified_agent_benchmark.py` — drives the one agent through a **frozen** conversational test set
at 320 concepts (200 nouns + 60 verbs + 60 adjectives), multi-seed, with a per-category pass-rate AND an explicit
boundary report. Three modes:
- **constructed** — the agent's own near-orthogonal phasor codes (the validated path).
- **grounded** — codes recalled from grounded word cues by `PhasorAssociativeMemory` (online-bounded STDP), the
  raw noisy *readout* (the harshest test; conflates perception noise with composition).
- **grounded-cleanup** — pattern completion (CA3 autoassociator): snap the noisy readout to the nearest CLEAN
  concept ATTRACTOR and compose on THAT (the biological architecture: compose on the stable concept rep, not raw
  sensory noise). The honest residual cost is identification accuracy, reported as `id_acc`.

Frozen test set: 27 facts (8 flat / 6 one-attribute / 5 two-attribute / 5 depth-1 clause / 3 depth-2 clause) +
6 who-queries + 6 abstention probes. Every (agent, action) key globally unique; every abstention probe is an
in-vocabulary pair never stored (the hard no-confabulation case). `tests/test_unified_agent_benchmark.py`
(3 tests). All numpy/CPU by design (the algebra realization of the spiking phasor substrate; the spiking version
is the GPU path, backlog item #1).

## Constructed-codes result — 5 seeds (42–46), D=2048

| category | pass | rate | note |
|---|---|---|---|
| flat | 40/40 | 100% | |
| one-attribute | 30/30 | 100% | resonator F=2 |
| two-attribute | 25/25 | 100% | resonator F=3, 60-adjective codebook |
| clause-depth1 | 25/25 | 100% | one embedded clause |
| **clause-depth2** | **0/15** | **0%** | **the honest ceiling** |
| who-query | 30/30 | 100% | |
| abstain | 30/30 | 100% | no confabulation |

OVERALL: **180/195 = 92.3%**, identical every seed (zero variance). The six robust categories are a genuine
multi-seed 100% at 320 concepts.

**The honest ceiling (clause-in-clause).** A fact whose embedded clause's OWN patient is another clause
("dog see (cat chase (bird eat leaf))") fails 0/15 — structured, not random: the agent decodes the outer two
levels and the innermost arguments collapse, gaining a *spurious* attribute
(`got "cat chase soft noun_124 eat fast noun_103"  want "cat chase bird eat leaf"`). At `D=2048` the third-level
bundle SNR drops below the agent's per-level flat-vs-attributed decision margin, so the attribute resonator
over-fires on a flat inner noun. A clean 6-category core next to a hard 0% ceiling is the point: the benchmark
probes a real limit, so the 100%s are meaningful.

## Grounded-codes mode — raw readout vs pattern completion

First, a measurement correction: the cleanup's **threshold-free** nearest-attractor identification is far higher
than the *thresholded* recall (the abstention threshold crushes low-confidence hits — the same artifact the prior
320-scale finding flagged). Threshold-free `id_acc`: **0.91 @ n_input=512 → 0.97 @ 1024 → 1.00 @ 2048**. So the
perception MECHANISM (argmax over concept attractors) is essentially perfect by n_input=2048; the question is
purely whether composition can use it.

Per-category, D=2048 (constructed 5-seed; grounded variants at the noted n_input):

| category | constructed | grounded raw @4096 | grounded raw @8192 | **grounded-cleanup @4096** |
|---|---|---|---|---|
| flat | 100% | 100% | 100% | 100% |
| one-attribute | 100% | **0%** | **0%** | **100%** |
| two-attribute | 100% | **0%** | **0%** | **100%** |
| clause-depth1 | 100% | 60% | 60% | 100% |
| clause-depth2 | 0% | 100%* | 100%* | 0% |
| who-query | 100% | 100% | 100% | 100% |
| abstain | 100% | 100% | 100% | 100% |
| **overall** | **92.3% (5s)** | **66.7% (5s)** | **66.7% (2s)** | **92.3% (2s)** |
| `id_acc` | – | – | – | **1.00** |

Two decisive results:

1. **Encoder scale alone is NOT the fix.** Raw readout @8192 (higher cue fidelity) still has attribute
   composition **0%** — the resonator cannot factor the noisy readout codebook regardless of cue resolution.
2. **Pattern completion IS the fix.** Snapping the noisy readout to the nearest clean concept attractor before
   composing recovers **full composition (92.3% = constructed)**, because at n_input ≥ 2048 the nearest-attractor
   identification is perfect.

\*The raw-grounded depth-2 "100%" inverts the constructed 0% for a revealing reason — same component, opposite
regime. The attribute resonator is the fragile piece: with clean codes its residual is high → it RECOVERS real
attributes (1-attr/2-attr 100%) but OVER-FIRES on flat inner clause args (depth-2 0%); with noisy raw codes the
residual is low → it CANNOT recover attributes (1-attr/2-attr 0%) but also cannot over-fire (depth-2 "passes" by
a dead detector). Pattern completion restores the constructed profile exactly (depth-2 back to 0%) — confirming
it gives the resonator genuinely clean codes.

## Anti-cheat — composition must track identification (it does)

If pattern completion were just reverting to clean codes (cheating), composition would stay ~100% regardless of
how well perception identifies the concept. It does not — `grounded-cleanup` composition **degrades with `id_acc`**:

| n_input | id_acc | one-attribute | two-attribute | flat | overall |
|---|---|---|---|---|---|
| 512 | 0.91 | 75% | **40%** | 88% | 78% |
| 1024 | 0.97 | 100% | 100% | 100% | 92% |
| 2048 | 1.00 | 100% | 100% | 100% | 92% |
| 4096 | 1.00 | 100% | 100% | 100% | 92.3% |

At id_acc 0.91 a mis-identified token collides with another concept's attractor and breaks every fact using it;
multi-token facts (two-attribute, 5 tokens) degrade most (40%). Composition reaches full only when id_acc = 1.00.
So the grounded-ness is real — the bottleneck is **perception (identification)**, a clean capacity curve in
`n_input`, not the composition substrate.

## Verdict — pattern completion resolves grounded composition (cheat-backlog #4)

**SHIPPED + honest measurement + the grounded barrier RESOLVED.** The converged unified agent at 320 concepts:
- **Robust on BOTH idealized and grounded codes:** flat fact memory, who-queries, abstention (the
  no-confabulation moat) — survives end-to-end on grounded codes in every mode.
- **Attribute composition on grounded codes is RECOVERED by pattern completion.** Raw sensory readout is too
  correlated/noisy for the resonator (0%, even at 8192 cue resolution); the autoassociative cleanup to the
  consolidated concept attractor restores full composition (= constructed). Anti-cheat-validated: composition
  tracks identification accuracy, so it is genuine grounded composition, not a clean-code revert. This
  **RESOLVES cheat-backlog item #4 (ungrounded codes)**: the fully-grounded agent works, the honest architecture
  being *grounded cue → pattern-complete to concept attractor → compose*.
- **Biology-translatable insight:** composition operates on stable concept representations, not raw sensory
  input — the cortex/hippocampus division of labour (sensory input → CA3 autoassociative pattern completion,
  Marr 1971 → cortical composition). The cleanup is necessary (raw fails) and sufficient (clean attractor →
  full composition); the only residual cost is identification accuracy, which scales cleanly with encoder
  dimension.
- **Boundaries:** clause-in-clause (constructed; the per-level auto-detection over-triggers) — the one ceiling
  that is a property of the *composition*, not the *codes*, and is identical in constructed and cleanup modes.

This is the converge-not-add deliverable: one agent, one substrate, one frozen multi-seed benchmark, an honest
"here is what it does and exactly where it ceilings", and the grounded-composition barrier the benchmark
surfaced is now resolved with a biology-faithful mechanism.

## Next (owner-steerable)

- **Constructed/cleanup clause-depth2** is now the single remaining composition ceiling (identical in both
  modes — it is a composition limit, not a code-fidelity one). The failure is the inner-level attribute
  over-trigger; a more conservative inside-clause flat-vs-attributed margin OR higher `D` may lift it. Cheap;
  the benchmark is the gate; guard 1-attr/2-attr against regression.
- **Pure-biology backlog item #1 (algebra → spikes):** the GPU-worthy decisive build — the spiking realization
  of this same agent (resonate-and-fire + an autoassociative cleanup network for the pattern-completion step).
  The benchmark is the spec it must reproduce.
- This benchmark is the routine multi-seed gate for any composition/substrate change.
