# Unified-agent benchmark — the converge-not-add measurement — 2026-06-04

**One line:** The fragmented composition / memory / abstention pieces are converged into ONE agent and
benchmarked at the real 320-concept scale on a FROZEN conversational test set. Constructed near-orthogonal
codes: a robust 6-category core at **100% across 5 seeds** with ONE honest ceiling (clause-in-clause).
Grounded STDP-learned codes: the trust + retrieval core (flat memory, who-queries, abstention) **survives at
100%**, attribute composition **collapses to 0%** — the two regimes have OPPOSITE failure profiles, which is
the honest end-to-end characterization the project lacked.

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
boundary report. Two modes:
- **constructed** — the agent's own near-orthogonal phasor codes (the validated path).
- **grounded** — codes recalled from grounded word cues by `PhasorAssociativeMemory` (online-bounded STDP); the
  honest extension (cheat-removal backlog item #4: ungrounded codes). Uses the genuine noisy *readout*, not the
  clean stored target.

Frozen test set: 27 facts (8 flat / 6 one-attribute / 5 two-attribute / 5 depth-1 clause / 3 depth-2 clause) +
6 who-queries + 6 abstention probes. Every (agent, action) key globally unique; every abstention probe is an
in-vocabulary pair that was never stored (the hard no-confabulation case). `tests/test_unified_agent_benchmark.py`
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

OVERALL: **180/195 = 92.3%**, identical every seed (zero seed variance). The six robust categories are a genuine
multi-seed 100% at 320 concepts.

**The honest ceiling (clause-in-clause).** A fact whose embedded clause's OWN patient is another clause
("dog see (cat chase (bird eat leaf))") fails 0/15. The misses are STRUCTURED, not random — the agent decodes
the outer two levels correctly and the innermost arguments collapse, gaining a *spurious* attribute:
`what does dog see? -> got "cat chase soft noun_124 eat fast noun_103"  want "cat chase bird eat leaf"`.
At `D=2048` the third-level bundle SNR drops below the agent's per-level flat-vs-attributed decision margin, so
the attribute resonator over-fires on what should be a flat inner noun. That a clean, robust 6-category core sits
next to a hard 0% ceiling is the point: the benchmark probes a real limit, so the 100%s are meaningful, not a
too-easy test.

## Grounded-codes mode — the honest extension (backlog item #4)

First, grounded recall fidelity at 320 concepts is a **capacity curve** in the cue dimension `n_input`:

| n_input | recall accuracy (argmax = token) | mean confidence |
|---|---|---|
| 512 | 1.9% | 0.10 |
| 1024 | 16.9% | 0.12 |
| 2048 | 55.3% | 0.15 |
| 4096 | 86.3% | 0.17 |

So grounded recall is viable only at large cue dimension (the toy default 512 fails; ~86% at 4096). Then the
genuine test — does ~86%-recall *noisy* grounded code support the *composition*? — at `n_input=4096`, 2 seeds:

| category | grounded rate | vs constructed | |
|---|---|---|---|
| flat | 100% | = | cleanup is noise-robust |
| who-query | 100% | = | |
| abstain | 100% | = | **no confabulation survives grounding** |
| one-attribute | **0%** | −100pp | **resonator needs clean codes** |
| two-attribute | **0%** | −100pp | |
| clause-depth1 | 60% | −40pp | structure decodes; attributed inner args fail |
| clause-depth2 | 100% | **+100pp (inverted)** | passes for the "wrong" reason — see below |

OVERALL grounded: **52/78 = 66.7%**.

**The two regimes have OPPOSITE failure profiles, and one mechanism explains both.** The attribute resonator is
the fragile component. With clean (constructed) codes its reconstruction residual is high, so it RECOVERS real
attributes (1-attr/2-attr 100%) but also OVER-FIRES on flat inner clause arguments (depth-2 0%). With noisy
(grounded) codes the residual is low, so it CANNOT recover real attributes (1-attr/2-attr 0%) but also cannot
over-fire — leaving the depth-2 facts' flat inner arguments clean (depth-2 100%). The grounded depth-2 "pass" is
therefore an artifact of a dead attribute detector, not robust nesting; stated honestly so it is not mis-read as
a win.

`recall_conf` 0.17–0.18 even at 86% argmax recall: the readout codes are aligned-but-noisy. Cleanup (flat,
who, abstain) tolerates that; the resonator's iterative unbind↔cleanup amplifies it into failure.

## Verdict — the honest end-to-end characterization

**SHIPPED + honest measurement.** The converged unified agent at 320 concepts:
- **Robust on BOTH idealized and biology-grounded codes:** flat fact memory, who-queries, and abstention (the
  no-confabulation moat). This is the trust + retrieval core, and it holds end-to-end on grounded STDP-learned
  codes — the distinctive property survives the move off idealized codes.
- **Robust on constructed, lost on grounded:** attribute composition (one/two-attribute). The resonator needs
  clean codes; grounded recall noise at 320 destroys F=2/F=3 factoring. This **quantifies cheat-backlog item #4**:
  moving from idealized to grounded codes, the agent keeps trust+retrieval but loses attribute composition.
- **Boundaries:** clause-in-clause on constructed (the per-level auto-detection over-triggers); attribute recovery
  on grounded.

This is the converge-not-add deliverable: one agent, one substrate, one frozen multi-seed benchmark, an honest
"here is what it does and exactly where it ceilings" — replacing scattered per-demo wins.

## Next (cheap follow-ups, owner-steerable)

- **Constructed depth-2 fix:** the failure is the inner-level attribute over-trigger. A more conservative
  inside-clause flat-vs-attributed margin (or higher `D`) may lift it — test whether the ceiling is
  dimension-budget or fundamental to the auto-detection. Tune in `nested_composition_agent`; guard 1-attr/2-attr.
- **Grounded attribute composition:** the resonator's grounded-noise fragility is the real barrier to a
  fully-grounded agent. Either a denoising cleanup before the resonator, or a higher-fidelity grounded encoder
  (`n_input` ≥ 8192, or a learned-prototype readout). This is the genuine open problem the benchmark localizes.
- This benchmark is now the routine multi-seed gate for any composition/substrate change.
