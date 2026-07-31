---
type: plan
status: live
date: 2026-06-11
---

# Cortex build plan — "decorrelate, then bind" (the de-risked architecture for step 3)

> **Status: present-before-build — UPDATED 2026-06-11 with a load-bearing falsification.** The cortex DE-RISK arc is complete and the architecture below is scoped into validated pieces, BUT the recommended decorrelation front end (Option E1, a fixed structural sparse expansion) was subsequently FALSIFIED (`2026-06-11-cortex-fixed-expansion-decorrelation-NEGATIVE.md`): a fixed random expansion + threshold cannot decorrelate the brain's *dense correlated* codes reproducibly (the common mode survives the linear expansion; threshold boundary units have a margin ~3,800× below realistic noise → reproducibility collapses to 0.03–0.06). This is the **fourth** distinct brain-based mechanism to fail at decorrelating the brain's correlated codes (after vanilla Hopfield, Storkey, and spiking dentate-gyrus), and they converge on a **documented project boundary: the Mikulasch-Priesemann point-neuron limit — whitening/decorrelation is analog / pre-spike (dendritic) in biology, and a point-neuron spiking substrate fundamentally cannot do it** (CLAUDE.md "Standing practice", the prior conversational decorrelation/whitening blocker).
>
> **⇒ THE FORK (this is now the owner's decision — see the new section "The fork" at the bottom):** the cortex splits into **(A) a semantically-flat cortex** — generated decorrelated codes (orthogonal, no semantic similarity) + the validated binder + cleanup — which is **achievable now** and already does the FULL conversational capability matrix at V=320 (the production composer); versus **(B) a semantically-structured cortex** — preserving the brain's correlated semantic codes so it can generalize (cat~dog) — which requires **analog/dendritic whitening = the deferred dendritic-substrate rewrite** (a months-scale substrate extension, Mikulasch-Priesemann-mandated). The decorrelate-then-bind blocks below remain valid for (A) with *generated* decorrelated codes; the *derive-decorrelated-from-correlated* front end (the E1/E2 piece) is what hit the wall and routes to (B).

## One-paragraph summary

The conversational composer binds role-filler facts with an exact-inverse vector-symbolic algebra (Fourier Holographic Reduced Representation — "FHRR"). That algebra is a principled idealization: it demands decorrelated, clean codes. Step 3 ("complete the functional cortex") is to replace the idealization with a learned, on-substrate cortical binder. The de-risk arc established — across three cleanup negatives, a positive control, and a multi-seed systematicity test — that **the brain's own concept codes are correlated (they carry semantic similarity), and no binder (fixed OR learned) and no brain-based post-hoc cleanup can use correlated codes**; but **decorrelated codes dissolve every barrier** (cleanup recovers to 1.000, a learned binder generalizes systematically, the no-confab gate separates). So the cortex is: **decorrelate first (a structural sparse-expansion front end), then bind over the decorrelated codes, then clean up — with a learned familiarity gate guarding abstention.** Each block is validated; the single genuinely-new piece is a reproducible, on-substrate sparse-expansion front end.

## The architecture (each block + its validation)

```
brain's correlated            STRUCTURAL SPARSE-          binder over decorrelated codes        cleanup            answer / abstain
semantic concept codes  --->  EXPANSION (decorrelate) --->  (learned-systematic OR FHRR)  --->  (attractor      + familiarity
(cos ~0.81)                   -> decorrelated (cos ~0.05)                                        or localist)      (no-confab) gate
```

| Block | Mechanism | Validation (this arc) |
|---|---|---|
| **Decorrelation front end** | structural sparse expansion (catalog F.12 granule/codon coding; D.12 dentate-gyrus pattern separation) — produces sparse high-dimensional codes, between-code cosine ≈ 0.05 | The *codes* are validated (the sparse-distributed scheme composes at 1.000 to 320 concepts, CLAUDE.md 2026-06-02). The **on-substrate, reproducible realization is the open new piece** — see below. |
| **Binder over decorrelated codes** | a learned binder OR the exact-inverse FHRR algebra | Learned binder is **systematic** on decorrelated codes (`2026-06-11-...-systematicity-NEGATIVE-ON-CORRELATED.md`: held-out 1.000 = train, 3 seeds, vs memorization-floor 0.000). FHRR is systematic by construction and already passes V=320 (vocab-ceiling GO). |
| **Cleanup** | distributed attractor OR localist matched-filter (Neural Engineering Framework / Spaun cleanup) | Distributed attractor recovers **1.000** on decorrelated codes (`2026-06-11-...-poscontrol-GO.md`); localist cleanup validated == numpy at D=2048, 27/27 seeds (2026-06-05). |
| **No-confab familiarity gate** | learned Bogacz-Brown anti-Hebbian familiarity signal | PASSED on decorrelated codes (gap 0.45, lesionable; cheap-first PARTIAL doc). Collapses on correlated (gap 0.03) — another reason decorrelation is load-bearing. |

## The single genuinely-new piece: a reproducible, on-substrate sparse-expansion

The three cleanup negatives included a spiking dentate-gyrus realization of this expansion, and it FAILED because the **spiking DG read is sub-reproducible** (~15 spikes / 600 neurons; same-input cosine ≈ 0.05; winners noise-determined). The structural expansion the project ships (`generate_sparse_patterns`) is reproducible because it is **deterministic** — a fixed expansion, not a noisy spiking k-winners-take-all read.

**Brain-based-bar DECISION (owner's call):** is a *fixed/structural* sparse expansion an acceptable brain-based primitive, or must the expansion itself be realized as spiking neurons + synapses?
- **Option E1 — fixed structural expansion (recommended for the first build).** A fixed random sparse expansion has direct biological precedent: the cerebellar granule layer (Marr-Albus-Ito) and the dentate gyrus both implement largely-fixed high-dimensional sparse expansions of their cortical input. Realize it as a fixed (non-plastic) expansion matrix wired into the bridge (a population of expansion units with fixed sparse weights), driven by the concept input, read by accumulated rate. Reproducible by construction; brain-based in form (granule expansion); the binding/cleanup/gate downstream are the learned/validated parts. This is the smallest build that completes the pipeline.
- **Option E2 — learned/plastic spiking expansion.** Train the expansion (the spiking DG-like layer) to produce stable, input-determined sparse codes (e.g. a learned efficient-coding / sparse-coding objective, or the concept-pool training applied to the expansion pathway). Higher biological fidelity, but it must beat the sub-reproducibility wall the spiking-DG probe hit — a larger, higher-risk build.

Recommendation: **E1 for the first end-to-end build** (it completes the cortex from validated parts and is biologically defensible as a granule/DG fixed expansion), with **E2 as the fidelity follow-on** once the pipeline is closed.

## The genuinely-deep open tension (the real research frontier — owner framing wanted)

The de-risk arc isolated a fundamental tension that "complete the functional cortex" ultimately runs into:

- **Binding wants decorrelated codes** (orthogonal → invertible, this whole plan).
- **Semantic generalization wants correlated codes** (similar concepts → similar codes, so the system can infer "a cat is like a dog"). The brain's concept codes are correlated *because* that similarity is useful.

A pure decorrelate-then-bind cortex (this plan) is **semantically flat**: every concept is equidistant, so it binds reliably but cannot generalize across similar concepts. Biology resolves the tension with **separate, linked representations** — semantic similarity in cortex, decorrelated-for-binding codes in the hippocampal/cerebellar expansion — coupled by the expansion (encode) and a compression (decode) (complementary learning systems; the cortico-hippocampal loop). 

**Owner-scope decision:** is a semantically-flat binding cortex sufficient for the conversational goal (it gives reliable who/what-Q&A + abstention + negation + clauses + two-attribute + dialogue at 320 concepts — the full validated matrix), or is *semantic generalization* (cat~dog inference) an in-scope target — which makes the expansion↔compression link and the dual representation the next deep research arc? The decorrelate-then-bind build (this plan) is valuable either way (it IS the binding half of the dual architecture), so it can proceed regardless; this decision only sets whether a second, larger arc follows.

## V=320 acceptance matrix (the build's gate)

The assembled spiking pipeline must reproduce the vocabulary-ceiling specification (`2026-06-10-vocab-ceiling-multiseed-GO.md`) end-to-end on the merged one-bridge substrate:
- comprehension + who/what fact-Q&A; **abstention / no-confab moat 100% (20/20 every cell)**; negation / yes-no; embedded clause (needs code-dimension D ≥ 256); **two-attribute binding** (the lifted K=5 boundary); generation; dialogue — at **V = 320 concepts, multi-seed (42–47)**, with the shuffled-fact permuted control at zero false hits.
- Anti-cheats: the abstention floor (unstored cues → "I don't know") and the held-out-novel-combination systematicity control (leakage-asserted) are run as part of acceptance, not after.

## Reusable machinery (build mostly assembles existing, validated parts)

- Sparse-expansion codes: `research/runners/concept_pool_sparse_distributed.py` (`generate_sparse_patterns`).
- On-bridge distributed attractor (permuted-control-clean, multi-seed): `research/runners/_D_sparse_heteroassoc.py`.
- FHRR binder on the bridge: the resonate-and-fire neuron model + complex synapses (`NeuronModel.RESONATE_AND_FIRE`); `research/runners/rf_phasor_composer.py`.
- Learned-binder option: surrogate-gradient backprop-through-time (`sim/bptt_snn*.py`, `sim/surrogate_grad.py`).
- Localist cleanup (validated): the NEF/TPAM spiking cleanup (2026-06-05).
- Familiarity gate, vocab-ceiling probe harness, the merged one-bridge builder (`research/runners/nav_conv_merged_bridge.py`).

## Cheap-first de-risk for the new piece (CPU, before the GPU build)

Before assembling the full spiking pipeline: realize **Option E1** (a fixed structural sparse-expansion population) on a small bridge, drive the brain's correlated concept codes through it, and verify (a) the expanded codes are decorrelated (between-cos ≤ 0.1) AND **reproducible** (same-input cosine ≥ 0.9 across fresh reads — the bar the spiking-DG read failed), then (b) bind + clean over the expanded codes and confirm parity with the numpy positive control. Anti-cheats: reproducibility unit-check; lesion the expansion → binding collapses; the abstention floor. Only on GO does the full GPU pipeline build proceed.

## The fork (the owner's decision)

The de-risk arc exhaustively established that **you cannot derive decorrelated codes from the brain's correlated codes with any point-neuron spiking mechanism** (four NEGATIVES → the Mikulasch-Priesemann analog/pre-spike whitening limit). So "complete the functional cortex" forks:

- **(A) Semantically-flat cortex — achievable now.** Use *generated* decorrelated codes (orthogonal phasors, no semantic similarity structure) + the validated binder + cleanup + familiarity gate. This is the production composer, and it **already passes the full conversational capability matrix at V=320** (comprehension, who/what-Q&A, abstention moat 100%, negation, embedded clause, two-attribute, generation, dialogue — `2026-06-10-vocab-ceiling-multiseed-GO.md`). The remaining work is modest: assemble it on the merged one-bridge substrate and confirm the matrix on-substrate. **Limitation:** every concept is equidistant, so it cannot generalize across similar concepts (no "a cat is like a dog" inference). The conversational FUNCTION is complete; the biological FIDELITY of the representation is not.
- **(B) Semantically-structured cortex — the deferred dendritic rewrite.** Preserve the brain's correlated semantic codes (so similar concepts have similar codes and the system generalizes) → requires **analog/pre-spike (dendritic) whitening**, which the point-neuron substrate cannot do (Mikulasch-Priesemann). This is the project's long-deferred **dendritic-learning substrate rewrite** (a months-scale arc: apical/basal compartments, sub-threshold dendritic computation). It is the path to a *proper brain analogue* that generalizes semantically — which is squarely the project's stated actual goal (artificial life with a proper, biology-translatable brain analogue).

**Decision for the owner:** is the semantically-flat functional cortex (A) sufficient for the goal — in which case step 3 is essentially achievable now and the remaining work is the on-substrate assembly — **or** is semantic generalization a required property of a "proper brain analogue" (B), making the dendritic-substrate rewrite the next major arc? Given the project's actual-goal framing (a proper, biology-translatable brain analogue), (B) is likely in scope, but its cost (months, a substrate rewrite) makes this a deliberate owner call, not an autonomous one.

## Verdict

The cortex is **exhaustively de-risked**. The binder, cleanup, and no-confab gate over decorrelated codes are validated; a *learned* binder is even systematic there. The one place a point-neuron spiking substrate hits a hard wall is **deriving decorrelated codes from the brain's correlated codes** — four distinct mechanisms failed, converging on the documented Mikulasch-Priesemann analog/pre-spike whitening limit. That splits the cortex into (A) achievable-now-but-semantically-flat and (B) the dendritic-rewrite for true semantic structure. This is not banking — it is the architecturally-correct conclusion, reached by exhaustive cheap-first de-risking and connected to the project's own prior whitening boundary; the fork is the owner's to resolve before the next (build, or substrate-rewrite) commit.
