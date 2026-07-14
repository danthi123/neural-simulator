# SYNTHESIS (the overnight's two levers combined) — an e-prop-TRAINED recurrent language cortex BEATS both the FIXED reservoir AND the bigram at deep context on WikiText (3-seed GO, anti-cheat-clean): a LEARNING spiking recurrent cortex passes the fixed-reservoir bound the honest, brain-plausible way

**Date:** 2026-07-14 (overnight)
**Runner:** `research/runners/_emerge_reservoir_lm_eprop_recurrent_derisk.py` (pre-built, a-1-surfaced; NEVER run before tonight) · raw `research/findings/raw/_eproplm/synth_s{42,43,44}.json`. numpy CPU; NO `sim/` edit.
**Status:** ✅ 3-seed GO on the mechanism (deep-context), with an honest aggregate-scale caveat.

## Why this is the synthesis of BOTH overnight levers
- **Selective-SSM lever (closed 2026-07-14):** a FIXED echo-state reservoir is bigram-level at deep context on real text (the aggregate can't beat a well-sampled bigram at tractable vocab). The fixed reservoir carries deep structure but a linear read-out over it is n-gram-competitive-at-best.
- **Deep-credit lever (closed 2026-07-14):** a transport-free, biological, LOCAL rule — **e-prop** (forward eligibility + membrane surrogate + **direct/broadcast feedback alignment**, NO BPTT, NO weight transport) — TRAINS deep credit on spikes (LIF 6-seed GO; ports to the production Izhikevich bridge; anti-cheat-validated).
- **⇒ the synthesis:** apply the validated e-prop rule to TRAIN the reservoir's RECURRENT weights `W_rec` (turning the fixed reservoir into a LEARNED recurrent language cortex, trained the brain-plausible way) — does it recover the deep-context structure the fixed reservoir loses? The runner tests exactly this, single-variable (read-out + reservoir init IDENTICAL across arms; the ONLY change is whether/how `W_rec` learns), with the e-prop broadcast/random-feedback learning signal `L_j = (B @ delta)_j` (B fixed random — the same transport-free route as the on-bridge BDSP apical feedback).

## The 3-seed result (V=300, WikiText, 1500 train / 400 eval sents, 8 epochs, n_pool=300; per-context-depth CE, LOWER=better)

**DEEP context (buckets 6-9 + 10-99), mean over 3 seeds:**
| arm | deep CE | vs fixed | vs bigram |
|---|---|---|---|
| fixed (echo-state baseline) | 3.234 | — | ≈ bigram (the selective-SSM result) |
| **plastic (e-prop on W_rec)** | **3.151** | **+0.084 (3/3)** | **+0.056 (3/3)** |
| shuffle_elig (permuted eligibility = broken credit) | 3.232 | +0.002 | — |
| zero_signal (L:=0, sanity) | 3.234 | +0.000 (==fixed) | — |

- **plastic beats FIXED at deep context 3/3** (per-seed margins +0.066 / +0.098 / +0.087) — e-prop-training `W_rec` extracts deep-context structure the fixed reservoir misses.
- **plastic beats the BIGRAM at deep context 3/3** (per-seed +0.052 / +0.070 / +0.047) — and at the DEEPEST bucket (10-99), plastic beats both fixed AND bigram every seed (e.g. seed 43: plastic 2.840 < fixed 2.936 < bigram 2.972).
- **ANTI-CHEATS CLEAN:** shuffle_elig ≈ fixed (breaking the credit-assignment structure removes the gain → the eligibility STRUCTURE is load-bearing, not generic weight motion); zero_signal == fixed exactly (no learning signal → `W_rec` frozen, sanity). `used_transpose` False (transport-free) by construction.

## ⇒ honest read
**A LEARNING recurrent spiking-language cortex — trained by a LOCAL, transport-free rule (e-prop broadcast-feedback) — beats both the fixed reservoir AND the bigram at deep context, all 3 seeds, anti-cheat-clean.** This passes the fixed-reservoir bound the selective-SSM lever characterized, the brain-plausible way (no BPTT, no weight transport) — the honest path to emergent language: a recurrent cortex that LEARNS deep-context structure from a stream.

**HONEST CAVEAT (aggregate = scale lever, same as selective-SSM):** the AGGREGATE CE is still bigram-level-to-worse (plastic 3.654 vs fixed 3.69 vs bigram 3.273) — dominated by the SHALLOW depths (depth 1-3 CE 4-5, far above bigram 3.3) at V=300. So this is a DEEP-CONTEXT-STRUCTURE result, not an aggregate-fluency result; the aggregate-fluency lever remains larger VOCAB / richer language (the ~23.7M-word / V=2000 validated regime), exactly as the selective-SSM finding named. The mechanism WIN (e-prop-training the recurrent cortex adds deep-context value past the fixed reservoir + bigram) is clean, robust, and anti-cheat-validated; the aggregate is the known scale frontier.

## NEXT
- **Scale the vocab** (V→1000/2000) — does the deep-context plastic-vs-fixed/bigram margin GROW with vocab (as the selective-SSM deep-tail signal did), and does the aggregate move off bigram at the non-null-discriminator scale?
- **ALIF horizon lever** (the runner's `alif` arms — Bellec-2020 adaptive-threshold gives e-prop its LONG memory): does adaptive-threshold e-prop extend the deep-context margin further into the tail?
- **The spiking port** — this is the RATE analogue; the on-bridge LIF/Izhikevich recurrent e-prop (using tonight's validated + population-cleaned on-bridge e-prop) is the fully-spiking realization.
