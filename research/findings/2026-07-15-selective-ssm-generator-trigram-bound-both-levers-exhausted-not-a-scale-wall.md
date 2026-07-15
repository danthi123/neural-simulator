# The selective-SSM language generator is TRIGRAM-BOUND at V=200 — BOTH levers (data + reservoir-size) are EXHAUSTED and it saturates ~0.26–0.33 ABOVE an interpolated trigram. A genuine architectural boundary of the fixed-reservoir approach, NOT a scale wall. Multi-seed, fair-baseline-verified.

**Date:** 2026-07-15 · **Status:** decisive HONEST NEGATIVE (the deliverable) on the fluency-crossover question. numpy CPU + GPU; NO `sim/` edit. The fair interpolated-trigram baseline (`_ngram_fair_baseline_probe.py`) caught the bigram-crossover artifact for the THIRD time this session.

## The question + the rigorous baseline

Compute was unlocked (owner lifted the gaming throttle); the mission-central question was whether the coupled selective-SSM generator's OVERALL next-token fluency crosses an n-gram at scale (the fluency crossover, ROADMAP §12). The lesson from earlier this session (bigram-starvation caught twice) forced the FAIR baseline: not the add-1 bigram (a starved strawman at any non-trivial V) but a **tuned add-k bigram AND a deleted-interpolation TRIGRAM** on the identical split (`_ngram_fair_baseline_probe.py`).

## The data (V=200, TinyStories, tuned add-k bigram + interpolated trigram baselines)

**The bigram crossover is a weak-baseline artifact** — the tuned bigram saturates ~2.70 early, so the selective SSM closes on it (`sel_over_tuned` −0.388@3k → −0.184@12k → −0.093@24k, 3-seed) — but the bigram is not the fair bar.

**vs the interpolated TRIGRAM (the fair bar) — sel is below and the gap does NOT close:**

| nt | trigram | sel_ce (np=200) | sel_over_trigram (np=200) |
|---|---|---|---|
| 3000 | 2.97 | 3.23 | −0.26 |
| 6000 | 2.75 | 3.07 | −0.30 (3-seed mean) |
| 12000 | 2.55 | 2.90 | −0.32 (3-seed mean) |
| 24000 | 2.48 | 2.77 | −0.30 (3-seed mean) |
| 48000 | **2.42** | **2.75** | **−0.33** |

**BOTH levers are exhausted:**
- **DATA lever — EXHAUSTED:** the selective SSM (np=200) SATURATES ~2.75 (nt=24000→48000 improves only 2.772→2.747 = −0.025) while the trigram saturates ~2.42 → a PERMANENT ~0.33 gap that if anything WIDENS as both saturate. More data does NOT cross the trigram.
- **RESERVOIR-SIZE lever — EXHAUSTED at np=500:** np=200 −0.302 → np=500 −0.262 (+0.04) → np=1000 −0.260 (+0.002 ≈ nothing) at nt=6000. Bigger reservoirs beyond np=500 do not help (multi-seed: np=500 closes the gap a consistent ~0.06 vs np=200, then np=1000 adds ~0).

⇒ the fixed-reservoir + selective-gate language cortex at V=200 saturates **~0.26–0.33 ABOVE a simple interpolated trigram**, and neither more data nor a bigger reservoir closes it. This is an **architectural capacity boundary of the fixed-reservoir approach**, not the "needs the ~23.7M-word regime" scale wall (the trigram itself saturates by nt=48000, and the model is still worse). It re-confirms `2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED` (loses to a proper n-gram) rigorously, from the reservoir-size + data-saturation angle.

## What is genuine (survives)

`sel_lift` vs the memoryless **bag** grows +0.73→+1.17 across data — the deep-tail selective mechanism (holding + input-selective conjunction) is real and strengthens with V, exactly as the validated Rung-3/4 results show. The boundary is on OVERALL fluency (the many shallow positions dominate the average), not the deep-tail mechanism.

## Why (the honest mechanism read)

A trigram uses EXACT 2-token context; the fixed random reservoir's context representation is lossy (a random projection the local read-out + selective gate cannot losslessly recover), so on the shallow-context majority of tokens the model is below an exact low-order n-gram. The R3 lever (a LEARNED input representation) was itself tuned-bigram-bound at tractable scale (`2026-07-15-emergent-input-representation-...`); learning W_rec is diagonal-credit-limited (`CONTROLS-REFUTED`); the reservoir-size + read-out-depth levers are exhausted here. So the WHOLE fixed-reservoir language-cortex family (read-out depth, input representation, reservoir size, selective gate) is n-gram-bound at V=200 / tractable scale.

## The next mechanism (per the boundary-surpassing workflow — a FRESH gate, not a re-tread)

The fixed-reservoir ladder is exhausted (all rungs trigram-bound), so per the workflow this launches a FRESH deep-research gate for a genuinely-NEW mechanism CLASS — a spiking/biological sequence cortex that represents multi-token context WELL ENOUGH to beat a low-order n-gram at tractable data, beyond {fixed-reservoir + local read-out + input-representation + reservoir-size} (all characterized). Candidates to research (external field + our record): input-dependent MULTIPLICATIVE gating at greater depth / stacked selective units; a biological attention-like content-addressable read with LEARNED keys (the `2026-07-11-content-addressable` frontier); the deep-credit/learned-recurrence frontier (parked, FA-family-exhausted — needs a new class). **Mission framing (ROADMAP §12 / the emergence bar):** the generator's ROLE is bounded-frame fluent wording BEHIND the no-confab gate (the EMERGE frames, which it renders on spikes), NOT beating a trigram in OPEN perplexity — so trigram-bound OPEN perplexity characterizes a proxy lens, it does not block the deployed conversational capability. The open-generation ladder (5 rungs + rung-6 discourse) does not depend on this crossover.

## Artifacts
- Runner: `_reslm_scale_trained_selssm_vectorized_derisk.py` (now reports `sel_over_tuned`; the tuned bigram is baked in). Fair baseline: `_ngram_fair_baseline_probe.py` (tuned add-k bigram + interpolated trigram). Configs: `raw/_fluency_np{200,500,1000}_nt*_s*.json`.
- Method: the fair-baseline discipline caught the bigram-crossover artifact 3× this session (input-repr gate, scale trajectory, this sweep) — STANDING RULE: any "beats the n-gram" language claim uses a tuned add-k bigram AND an interpolated trigram on the same split, never add-1.
