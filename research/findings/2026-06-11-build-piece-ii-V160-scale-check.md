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

## Honest framing
- This is a **real, honest scale result**, reported as found: the build's capability (generalization) scales; one global structure-recovery diagnostic for the brain-based read-out degrades at scale and needs the scale-appropriate parameter (read-out steps and/or the homeostatic learn). This is a build-time refinement target, not a capability failure — and exactly what the cheap-first scale-check is for (catch the scale-tuning need at V=160, before the V=320 / multi-seed spend).
- Single-seed; the generalization-scales result + the global-Pearson degradation are mechanistic. The diagnostic localizes the lever before any declaration.

**No banking** — the build's first scale-check reported exactly as found (capability scales; a read-out diagnostic degrades at scale); the lever-localizing run is already running.
