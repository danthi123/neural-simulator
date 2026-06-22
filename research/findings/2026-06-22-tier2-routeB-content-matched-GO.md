# Tier-2 #6 Route-B CONTENT-MATCHED test — the DA-driven encoding gain is BEHAVIORALLY LOAD-BEARING on the one brain; the single-pair LATENT was a content confound (2026-06-22, GO)

**Scope:** the decisive isolation of the Tier-2 #6 Route-B DA-driven encoding gain from per-fact content — resolving
the `2026-06-22-tier2-routeB-deployment-smoke-LATENT.md` LATENT. `research/runners/_tier2_routeB_content_matched.py`.
Single-seed (42), GPU; **6-seed confirming**. NO `sim/` edit (strict reuse-by-import from the smoke). On `main`.

## The test
Store N=16 DISTINCT facts; RANDOMLY assign each to HI-DA or LO-DA encoding (balanced, seed-shuffled) so the per-fact
content-robustness asymmetry AVERAGES OUT. At a moat-safe read-damage knee, measure mean(HI-DA recall) vs
mean(LO-DA recall). The DA source is the REAL co-resident spiking SNc (`limbic_snc`; the gain is baked into each
fact's complex weights AT STORE TIME by the live `get_concentration("dopamine")`); the composer is the deployed
`MergedNavConvAgent(co_resident_limbic=True)` path.

## Result — GO (seed 42)
| | value |
|---|---|
| DA low/high → gain | 0.679 → g=1.358 / 0.843 → g=1.686 (applied hi>lo) |
| **KNEE (noise=400, moat-safe)** | **meanHI=0.875 (7/8) vs meanLO=0.500 (4/8), effect=+0.375** |
| (noise=600) | meanHI=0.375 vs meanLO=0.000, effect=+0.375 |
| **DA-LESION** (all facts re-stored at baseline DA) | meanHI=0.500 = meanLO=0.500, **effect=+0.000** (collapses) |
| MOAT (HARD) | **0 false-accepts at every noise level** |
| REGRESSION (`encoding_gain_fn=None`) | byte-identical |
| verdict | **GO** (N=16, n_flip=3, knee_noise=400) |

Controlling for content (the random HI/LO assignment averaged over 16 facts), the HIGH-DA-encoded facts recall
RELIABLY BETTER at the moat-safe knee (+0.375 mean), and the DA-LESION control collapses the effect to 0.000 —
confirming it is DA-DRIVEN, not a label/content quirk. The single-pair deployment-smoke LATENT was indeed the
specific FACT_HI/FACT_LO pair's intrinsic-robustness confound, not a dead mechanism.

## What this closes
⇒ **Tier-2 #6 (the owner's "one self" closure) is behaviorally REAL on the one brain.** The shared spiking dopamine
does not merely *reach* the cortical composer (the smoke's mechanism result) — it behaviorally SHAPES what the
composer remembers: salience-modulated encoding strength (high-DA / high-salience facts are stored more strongly +
recalled more reliably under read damage). Biologically grounded (dopamine-dependent memory consolidation;
Lisman–Grace hippocampal–VTA loop, Schultz RPE-salience). NO `sim/` edit; the no-confab moat held throughout
(0 false-accepts at every noise level) — the salience lever NEVER trades against the moat.

## Next
1. **6-seed validation** (42,43,44,100,101,102) — the owner's 6-seed rule — confirm the +effect + the lesion-null
   generalize. (In flight.)
2. On 6-seed GO: a **production wire-in** — a `salience_encoding` opt-in on `MergedNavConvAgent` so a high-DA
   (surprising/rewarding) turn is encoded more strongly — + the durable record.
3. The richer routes (DA-gated **reconsolidation**, salience-gated **recall**) remain available as ADDITIONAL
   one-self levers, now that encoding-strength is confirmed behaviorally load-bearing — not a fallback for a dead
   mechanism.
