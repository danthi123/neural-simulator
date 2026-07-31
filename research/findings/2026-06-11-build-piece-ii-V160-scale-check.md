---
type: finding
status: corrected
date: 2026-06-11
mechanism: learned-embedding
---

# Build piece (ii) — V=160 scale-check: generalization SCALES; the brain-based read-out's GLOBAL Pearson degrades at scale (read-out propagation diagnostic in flight)

**Date:** 2026-06-11. **Runner:** `research/runners/learned_graded_embedding_divnorm_readout_probe.py` (the validated brain-based divnorm read-out, at 3.3× the toy vocab). **Backend:** `SIM_BACKEND=cupy` (GPU). **Raw:** `research/findings/raw/_lge_v160_scalecheck_seed42.json` + `_lge_v160_scalecheck.log`. **Scale:** V=160 (20 clusters × 7 = 160 concepts; n_pool=7000, pattern=100; 7300-neuron bridge, 30.7M synapses). **Seed:** 42. **Context:** the cheapest first build step (build plan piece ii) — does the de-risked recipe hold past toy vocab (48) before the V=320 run?

> **Verdict: QUALIFIED — the build-critical capability SCALES, one global diagnostic degrades.** At 3.3× the toy vocabulary, **generalization (the conversational matrix's consumed quantity) PASSES at 0.970** (bar 0.7; orthogonal control 0.250 = chance, permuted 0.307 — both collapse correctly) and **the 2nd-order cat~dog cosine margin PASSES at +0.402** (bar +0.10). BUT the **global structure-recovery Pearson(sim, S_true) for the brain-based read-out dropped to +0.333** (below the 0.5 bar; toy scale was +0.60–0.73). The structure is still recoverable from the learned weights (host-on-W stand-in +0.699, ceiling +0.955) — so the brain-based read-out captures the *local neighbour* structure (what generalization needs) but not the *global* correlation as well at this scale. A read-out propagation diagnostic (diffusion-step + sigma sweep on the same learned weights) is in flight to localize the lever.

## Why this ran
The dual/CLS learned-embedding de-risk is complete + fully brain-based, validated at toy scale (48 concepts) and — for cycle-independence — at full scale (homeostasis GO, confirmatory `b6n98g33h` official `CONSENSUS VERDICT: GO`, scaling t=600). Build plan piece (ii) is "scale to the production concept set + re-confirm the gates." The cheapest first step is a single-seed scale-check at an intermediate vocab (V=160) — the smallest run that confirms the recipe scales monotonically past toy before the V=320 run.

## Results (seed 42, V=160, brain-based divnorm read-out, cycles=2)

| Gate | V=160 | toy (48) ref | bar | status |
|---|---|---|---|---|
| **G2 generalization** (held-out-neighbour) | **0.970** | 1.000 | ≥0.7 | **PASS** (ortho 0.250 + perm 0.307 collapse) |
| **G1 2nd-order cat~dog margin** | **+0.402** | +0.42 | ≥+0.10 | **PASS** |
| G1 global Pearson(sim, S_true) | +0.333 | +0.60–0.73 | ≥0.5 | **soft** (degrades at scale) |
| anti-cheat Pearson(W, raw_counts) | +0.556 | +0.69 | ≪0.99 | PASS (genuine learning, distinct from counts) |
| host-on-W stand-in (the structure IS there) | +0.699 | +0.70 | — | structure recoverable |
| host ceiling (PPMI+SVD on raw counts) | +0.955 | +0.93 | — | target |
| raw-diffusion baseline (no divnorm) | +0.278 (margin +0.036) | — | — | divnorm lifts margin +0.036 → +0.402 |

**Elapsed:** 372s.

## Diagnosis (in flight) + honest framing
- **The build's PURPOSE scales.** The capability the structured cortex adds — generalization across similar concepts (cat~dog inference) — holds at 0.970 with the controls collapsing, and the 2nd-order margin holds at +0.402. These are the quantities the conversational matrix consumes. **The capability scales to V=160.**
- **The global Pearson is a stricter, holistic structure-recovery diagnostic** where the brain-based read-out underperforms the host method *at scale* (the host extracts +0.699 from the same weights; the brain-based read-out gets +0.333 globally while getting the local structure right). The most likely cause is **read-out under-propagation**: the diffusion read-out used steps=2 (tuned at toy scale); the 160-concept graph is larger, so 2 steps may not reach across it (the raw-diffusion margin is only +0.036 before divnorm sharpens it). The diagnostic sweeps diffusion-steps {2,3,4,6,8} × sigma {0.0003,0.001,0.003} × divnorm {ch, marginal} × order {interleave, post} on the same learned weights (fast post-learn) to find whether a scale-appropriate read-out recovers Pearson ≥0.5 while keeping generalization.
- **A second candidate lever (held in reserve):** this run used the de-saturation rescale learn (the divnorm probe's default), NOT the production homeostatic recurrent. At full scale the homeostatic recurrent (scaling t=600) reached Pearson(sim,S_true) +0.807 — much higher than this run's +0.333 — so the homeostatic learn may itself recover the global Pearson at scale. If the read-out sweep doesn't close it, the homeostatic-recurrent-at-V=160 is the next test.

## UPDATE 1 — read-out propagation FALSIFIED; the LEARN is the lever
The read-out diagnostic (`b87xnqi69`, 40 configs: diffusion-steps {2,3,4,6,8} × sigma {0.0003,0.001,0.003} × divnorm {ch, marginal} × order {interleave, post}, on the same learned weights; raw `_lge_v160_readout_diag.json`) **falsifies the under-propagation hypothesis: more diffusion steps makes it WORSE** — steps=3 → Pearson +0.207 (graded flips to 0); steps=2 stays optimal at +0.333. **No read-out config recovers Pearson ≥0.5.** The brain-based spreading-activation+divnorm read-out FAMILY caps at ~+0.333 global Pearson at V=160, while the host PPMI+SVD extracts +0.699 from the *same* W. So the read-out is exonerated as the *tunable* lever — and the gap localizes to the **learn**: this run used the divnorm probe's *de-saturation rescale* learn, NOT the production **homeostatic recurrent**. At toy scale the homeostatic recurrent gave the brain-based read-out Pearson +0.807 vs the de-saturation's +0.60–0.73 on the same data — i.e. the homeostatic W is materially more read-out-friendly. **Indicated next test (in flight, `be375kkzu`): the homeostatic recurrent at V=160** (scaling set-point bracket for n_pool=7000, cycles {2,5}) — does the production learn give the brain-based read-out global Pearson ≥0.5 at V=160, or is the global-Pearson degradation a genuine scale boundary (with generalization still passing)?

## UPDATE 2 — the production HOMEOSTATIC recurrent is a CLEAN GO at V=160 (near ceiling); the BOUNDARY was the de-saturation stand-in
The homeostatic-learn test (`be375kkzu`, raw `_lge_v160_homeo_seed42.json`) confirms the lever: swapping the de-saturation rescale for the production **homeostatic recurrent** recovers the global Pearson from +0.333 to **near the host ceiling**, all gates clean at V=160:

| Learn (V=160, brain-based divnorm read-out) | Pearson(sim,S_true) | 2nd-order | gen | graded | cycle-indep (c2→c5) |
|---|---|---|---|---|---|
| de-saturation rescale (the BOUNDARY run) | +0.333 ✗ | +0.402 | 0.970 | 1 | — |
| **Oja recurrent t=40 (BEST)** | **+0.977 → +0.975** ✓ | +0.930 | **1.000** | 1 | slope +0.0028/cyc |
| **synaptic scaling t=1200** | **+0.960** ✓ | +0.909 | **1.000** | 1 | slope +0.0061/cyc |
| scaling t=2400 (set-point too high @ n_pool=7000) | +0.308 | +0.172 | 0.988 | 0 | — (over-normalized) |
| host ceiling (PPMI+SVD) | +0.955 | +0.645 | 1.000 | — | — |

**⇒ At V=160 the production recipe (homeostatic recurrent + divisive-normalization read-out) passes ALL gates near the host ceiling — Pearson +0.96–0.977, generalization 1.000, graded=1, cycle-independent.** Actually *better* than the toy-scale de-saturation result (+0.60–0.73). The V=160 "BOUNDARY" was entirely an artifact of the scale-check accidentally using the divnorm probe's de-saturation-rescale default instead of the production homeostatic recurrent. **Set-point note for V=320:** the scaling set-point must track n_pool (t=1200 good, t=2400 too high at n_pool=7000); Oja's L2-norm set-point (t=40) is robust. **⇒ The recipe SCALES to 3.3× toy vocab, clean. Next: the V=320 scale-check (6.7× toy) with this confirmed production recipe.**

## Honest framing
- This is a **real, honest scale result**, reported as found: the build's recipe (production homeostatic recurrent + divisive-normalization read-out) passes all gates near the host ceiling at V=160 (the initial soft global-Pearson was the de-saturation stand-in, not the production learn — caught and corrected by the cheap-first diagnostic chain: scale-check → read-out diagnostic [falsified propagation] → homeostatic-learn test [GO]). Exactly what the cheap-first scale-check is for.
- **Per the owner's "validate a signal by its function" standard:** the FUNCTION the structured cortex must add is generalization across similar concepts — which PASSES at V=160 (0.970, controls collapse). The global Pearson is a stricter faithfulness diagnostic; whether it also recovers (via the homeostatic learn) is the in-flight question, but the functional capability is already validated at scale.
- Single-seed; the results are mechanistic. The diagnosis localizes the lever before any declaration.

**No banking** — the build's first scale-check reported exactly as found (capability scales; a read-out diagnostic degrades at scale); the lever-localizing run is already running.
