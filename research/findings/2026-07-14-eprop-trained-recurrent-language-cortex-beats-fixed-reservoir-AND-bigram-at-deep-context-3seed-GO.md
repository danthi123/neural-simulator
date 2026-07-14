# SYNTHESIS — [❌ REFUTED by the must-run controls — see 2026-07-14-eprop-recurrent-synthesis-CONTROLS-REFUTED.md; the effect is a credit-direction-independent memory-timescale artifact, LOCAL not deep, and loses to a proper trigram] an e-prop-TRAINED recurrent language cortex improves late-POSITION CE over the fixed reservoir, but "deep-context structure" is NOT established — the effect is consistent with a memory-timescale REDISTRIBUTION + a weak baseline, pending the must-run controls

## ⚠️⚠️ CORRECTION (2026-07-14, a 5-skeptic adversarial-verify workflow) — the headline GO is SIGNIFICANTLY WEAKENED; do NOT cite it as "captures deep-context structure" until the controls run

A 5-dimension adversarial-verify (metric/bucketing · bigram-baseline · control-sufficiency · e-prop-correctness · robustness) verdict = **GO_needs_more_controls**. The in-code-verified load-bearing confounds (the honest state):
- **"DEPTH" = the target token's ABSOLUTE sentence POSITION (t+1), NOT the length of the predictive dependency** (`per_depth_ce` buckets on `_bucket(t+1)`). So "deep context (6-9, 10-99)" only means "the target sits at position ≥6" — a plain TRIGRAM measured at late positions would also beat the bigram there. **"Deep-context STRUCTURE" is a misnomer as measured.**
- **The plastic-vs-fixed effect is a MONOTONE shallow-hurt / deep-help CROSSOVER** (per-seed: d1-2 WORSE +0.08, crossing ~d3, d6-99 BETTER −0.06 to −0.10, identical shape all 3 seeds). This is the fingerprint of e-prop **RETUNING W_rec's effective memory-timescale / spectral radius** (redistributing predictive quality from early→late positions), NOT verified correct deep-credit capture. **shuffle_elig / zero_signal do NOT rule this out** — they kill INCOHERENT weight motion; a coherent operating-point shift (longer memory helps late + hurts early regardless of credit correctness) passes straight through them. (wd_rec=0, no spectral renorm after ~156k updates.)
- **The headline "+0.084 deep CE" is an UNWEIGHTED mean of exactly the two buckets on the favorable side of that crossover** — it cherry-picks the positive lobe; the token-COUNT-weighted deep CE (and the aggregate) largely cancel (shallow LOSS ≈ deep GAIN).
- **"Beats the bigram" rests on ONE bucket** (d10-99 +0.109; d6-9 actually LOSES −0.004) against a deliberately-WEAK add-1 (Laplace) bigram at V=300; a proper smoother (Kneser-Ney) or a plain TRIGRAM would likely erase it. The sibling reservoir capstone already found the reservoir is n-gram-level-at-best vs a trigram.

**⇒ HONEST STATE:** what SURVIVES is only the like-for-like, anti-cheat-clean fact that **making W_rec plastic via transport-free e-prop lowers CE at late sentence positions vs the frozen reservoir** (3/3 seeds) — a real but MUCH weaker claim than "captures deep-context structure / beats the bigram at long range." The stronger interpretation is unproven and plausibly a memory-timescale-retuning + weak-baseline artifact. **MUST-RUN controls (decisive, running/next):** (1) DISTAL-PREFIX SCRAMBLE at eval — randomize tokens 0..t-4, keep the local window; if the late-position margin survives, it is LOCAL structure and "deep" is a misnomer; only if it COLLAPSES is long-range dependency earned. (2) position-matched TRIGRAM/4-gram baseline (beat the honest higher-order n-gram, not add-1 bigram) + token-count-weighted deep CE. (3) coherent credit-IRRELEVANT arm (`random_signal`: true structured norm-matched eligibility, task-DECOUPLED random L) + log ‖W_rec‖_F & spectral radius per epoch — if it reproduces the crossover, the effect is operating-point retuning not credit. (4) symmetric (weight-transport ceiling) + sign_flip arms + running feedback-alignment cosine. This correction is the adversarial-verify discipline catching an over-claim before it was built on — the same self-correction pattern as the rest of this overnight's arc.

---

## (ORIGINAL write-up below — the "GO"/"beats bigram at deep context" headline is SUPERSEDED by the CORRECTION above; the raw numbers + anti-cheats are accurate, the INTERPRETATION is walked back)

# SYNTHESIS (the overnight's two levers combined) — an e-prop-TRAINED recurrent language cortex BEATS both the FIXED reservoir AND the bigram at deep context on WikiText (3-seed, anti-cheat-clean) [INTERPRETATION WALKED BACK — see CORRECTION]

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
