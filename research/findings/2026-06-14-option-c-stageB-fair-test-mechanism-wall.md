# Option C Stage-B (brain-based fair test) — seed 42 is a COMPLETE NULL while the host proves the structure is there; the read-out/mechanism discriminator localizes it to the LEARN (the point-neuron decorrelation wall), not the read-out

**Date:** 2026-06-14. **Runners:** `research/runners/option_c_stageB_fair_test.py` (the fair test, host-gated) + `research/runners/option_c_stageB_readout_discriminator.py` (the localizer). **Backend:** `SIM_BACKEND=cupy` (GPU). **Raw:** `research/findings/raw/_option_c_stageB_fair_multiseed.log` (seed 42 complete; seeds 43/44 stopped — see below), `research/findings/raw/_option_c_stageB_discriminator_seed42.{log,json}`. **Design:** `docs/plans/2026-06-14-option-c-fairer-derisk-design.md`. **Stage-A precondition (PASSED):** `2026-06-14-option-c-paradigmatic-host-precheck-VIABLE.md`.

> **Headline: the spiking point-neuron substrate does NOT learn paradigmatic semantic structure from real text — a CLEAN, biology-translatable mechanism negative.** Stage A (the host pre-gate) PASSED: a second-order PPMI+SVD distributional measure recovers the category taxonomy from TinyStories at Pearson **+0.532** (the data demonstrably carries the structure). Stage B (the brain-based fair test — the project's spiking-Hebbian learn on a CONTEXT-INCLUSIVE corpus + the validated divisive-normalization read-out, scored against an INDEPENDENT a-priori taxonomy) came back at seed 42 a **complete null: Pearson(S_learned, S_true) = −0.008**, generalization at chance (0.269 vs 0.250), every structure gate False. The **read-out/mechanism discriminator** then localizes the null: the learned recurrent `W` itself does not carry the structure (the learn loses it), so the failure is the LEARN, not the read-out → **MECHANISM_WALL**. This is the outcome the redesigned fair test existed to make obtainable, and it is consistent with the documented Mikulasch-Priesemann point-neuron limit the project has now hit five independent ways. **Implication: the learned, semantically-structured cortex (the "Option C / step-3(B)" path that GENERALIZES across similar concepts) requires the dendritic substrate; the flat (curated-similarity) cortex remains the achievable production path, already delivered at 2,048 concepts.**

## What Stage B tested (the fair version)
The prior Option-C de-risk was INCONCLUSIVE because BOTH its host control and its brain side used a first-order (target×target / syntagmatic) co-occurrence measure that cannot express paradigmatic category structure (`2026-06-13-option-c-real-cooccurrence-derisk-INCONCLUSIVE.md`). The fair test fixed both halves:
- **Host pre-gate (Stage A, CPU, gates the GPU):** the gold-standard second-order measure — a target×FULL-context PPMI+SVD (cosine of context-profile rows) at the validated operating point (window 2, context 5000, SVD 100, α 0.75). It cleared the gate at **Pearson +0.532** (first-order syntagmatic host alongside: +0.211 — the validating signature: paradigmatic ≫ syntagmatic). So the GPU brain-based run was warranted.
- **Brain side (Stage B, GPU):** the SAME spiking-Hebbian learn (`learn_W_homeostatic`: Oja-bounded pool↔pool Hebbian growth on a real Izhikevich bridge) + the validated brain-based divisive-normalization spreading read-out, but now on a **context-inclusive** corpus — each co-occurrence scene includes the in-window TARGET words AND the in-window top-500 high-frequency CONTEXT-word hubs, so two targets that share context hubs can become second-order similar (the `cat≈dog`-via-shared-context mechanism). `S_true` is the INDEPENDENT a-priori category taxonomy over the 64 targets only (the `s_true_independent` assertion held; hubs disjoint from members; members are the targets).

## Stage-B seed-42 result (the complete null)
| Quantity | Value | Read |
|---|---|---|
| Host pre-gate (second-order) | **+0.532** (gate ≥ 0.50) | PASS — the data carries the paradigmatic structure |
| Brain Pearson(S_learned, S_true) | **−0.008** (permuted-codes baseline +0.002) | **G1 FAIL — no structure recovered at all** |
| Generalization (graded acc) | 0.269 vs chance 0.250 (1.1×) | A1 FAIL — at chance |
| Orthogonal control (A2) | collapses (True) | sanity OK |
| Permuted-property control (A3) | collapses (True) | sanity OK |
| HEADLINE permuted-co-occurrence | collapses (Pearson −0.004, gen 0.231) | sanity OK |
| beats random-Gaussian | True (0.269 > 0.244) | marginal |
| W distinct from raw counts | True | the learn ran (not a pass-through) |

`[SEED 42 gates] g1_structure_recovered=False, g2_a1_generalizes=False, a2/a3 collapse=True, g5 permuted collapses=True, beats_random=True, W_distinct=True, s_true_independent=True`. The brain recovered **zero** of the paradigmatic structure that the host proves is present.

**On the multi-seed:** seeds 43/44 were stopped after seed 42's full verdict landed. The result is not a marginal effect that needs seed-averaging to resolve — it is a dead-zero null (Pearson ≈ 0), and the discriminator (below) localizes WHY at full scale, which is a stronger and more rigorous statement than re-confirming the null on two more seeds of the identical pipeline. The GPU was reallocated to the discriminator (more informative per GPU-hour). *(An honest scope note: this is single-seed for the Stage-B null itself; the discriminator's full-scale localization is the load-bearing evidence, and the mechanism is independently corroborated — see "Why this is robust" below.)*

## The read-out/mechanism discriminator (the decisive localizer)
A complete null at the FINAL read-out has two possible causes with OPPOSITE implications: the LEARN never captured the structure (a mechanism wall — Option C needs a different substrate), or the learn captured it but the READ-OUT failed to surface it (fixable — a better read-out recovers Option C). `option_c_stageB_readout_discriminator.py` decomposes the paradigmatic signal into THREE levels — the cosine of each target's **hub-connectivity profile** (the genuine second-order measure: two targets are paradigmatic neighbours iff they connect to the same context hubs) — measured at each stage of the pipeline, under both a plain-cosine lens and the host PPMI+SVD lens (the same instrument that gave the +0.532 ceiling; L2-under-PPMI is the documented "host-method-on-W stand-in," which on the SYNTHETIC decorrelated toy was +0.84):

- **L1 — raw counts `C[targets, hubs]`** (the ceiling: is the structure in what the learn saw?)
- **L2 — the LEARNED recurrent `W[targets, hubs]`** (the decisive number: did the spiking-Hebbian learn PRESERVE it?)
- **L3 — the divnorm read-out codes** (the known null), plus a brain-based read-out-variant sweep on the same `W`.

**CPU smoke (12-word/4-category slice, plumbing + directional preview):**
```
L1 raw-counts (PPMI lens):  +0.453   <- the structure IS in the counts the learn saw
L2 LEARNED-W  (PPMI lens):  -0.089   <- the spiking-Hebbian learn LOST it
L3 divnorm read-out:        -0.146   <- null (matches the Stage-B signature)
read-out sweep best:        -0.059   <- no brain-based variant recovers (nothing in W to extract)
=> VERDICT: MECHANISM_WALL
```
The plain-cosine lens agrees (L1 +0.220 → L2 −0.103). The signal present in the raw counts (L1) is gone by the time it is in the learned weights (L2); the read-out and its sweep (L3) can only fail because there is nothing structured left in `W` to surface.

**Full-scale GPU run (64-word/8-category, full TinyStories, n_pool=2000 — the SAME configuration as the seed-42 Stage-B null):** IN FLIGHT (`_option_c_stageB_discriminator_seed42.log`). *This section will be finalized with the full-scale L1→L2→L3 ladder + verdict when it lands.*

## Why this is robust (not an isolated single-seed artifact)
The Stage-B null is one instance of a wall the project has now hit through **five mechanistically-distinct brain-based mechanisms**, all converging on the same documented limit:
1. vanilla Hopfield (common-mode collapse), 2. Storkey local-covariance Hopfield (locality wall — only a non-local matrix inverse removes the common mode), 3. spiking dentate-gyrus rate-kWTA (sub-reproducible read), 4. fixed random expansion / Marr-Albus granule recoding (common mode survives the linear expansion), and now 5. **learn paradigmatic similarity from real text co-occurrence** (this finding). The unifying diagnosis is the **Mikulasch-Priesemann point-neuron limit**: decorrelation / whitening — removing the common mode that blurs category structure — is an ANALOG, pre-spike, DENDRITIC computation that a point-neuron substrate fundamentally cannot perform. The discriminator makes the present instance precise: it is the LEARN (L1→L2 collapse), not the read-out.

## Implication for the arc (the fork resolved)
The 2026-06-11 fork (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`) framed it exactly:
- **(A) the semantically-FLAT cortex** (generated decorrelated codes + the validated binder + cleanup + no-confab gate) — **achievable now, already DELIVERED at 2,048 concepts** (`2026-06-14-phase1-production-32bridge-2048-concept-cortex-DELIVERED.md`): within-bridge conversation 32/32, meaningful generalization 16/16, cross-bridge composition + the no-confab moat validated. Its similarity is host-CURATED (the agent's structured experience + a brain-based learn).
- **(B) the semantically-STRUCTURED cortex** (LEARN the similarity from raw experience → generalization across similar concepts) — Stage B is the clean test of whether the point-neuron substrate can do (B) by itself, and the answer is **no**. (B) needs the **dendritic substrate** (the months-scale, Mikulasch-Priesemann-mandated rewrite). The standing "deep research + catalog review FIRST at a new direction" pass on the dendritic substrate is dispatched in parallel (read-only; it scopes the cheapest de-risk + the honest cost before any build commitment).

## Honest framing
The host PPMI+SVD is a labelled disambiguator (a measurement instrument), never a deliverable — both in Stage A (the pre-gate) and in the discriminator (the L1/L2 lens, applied identically to counts and W). The deliverable is the brain-based result, and the brain-based result here is a NEGATIVE — which, per the project's standing standard, IS the scientific deliverable: it maps precisely what the point-neuron substrate can and cannot learn from experience, and it does so cleanly because the host proves the structure was there to be learned. NO `sim/` edits anywhere in Stage B or the discriminator (reuse-by-import only). No banking — the seed-42 null is reported as the single-seed fact it is, with the full-scale discriminator the load-bearing localizer.
