---
type: finding
status: boundary
lane: A (affect / emotion keystone)
date: 2026-08-13
mechanism: affect-arousal-channel
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_arousal_channel_derisk.py
artifacts:
  - research/findings/raw/_affect_arousal_channel_6seed.json
  - research/findings/raw/_affect_arousal_channel.json
builds_on:
  - research/findings/2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md
  - research/findings/2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md
---

# A SEPARATE emergent AROUSAL/intensity channel — the SUM channel of the same innate-primary conditioning whose DIFFERENCE is valence — DOES predict held-out Warriner AROUSAL (6-seed mean r=+0.265, every seed ≥+0.12) and adds a real (modest, ~13%) graded-valence-STRENGTH increment, but is a BOUNDARY: full sign-orthogonality holds in 4/6 seeds only, and the arousal contribution to strength is small. <!--derived-->

<!--derived-->
The headline numbers are aggregate means over the 6-seed JSON `means` block (means over seeds live in no per-seed file); per-seed values are in `research/findings/raw/_affect_arousal_channel_6seed.json`.

<!--derived-->

**Runner:** `research/runners/_affect_arousal_channel_derisk.py`
**Raw:** `research/findings/raw/_affect_arousal_channel_6seed.json` (+ provenance sidecar), smoke `..._channel.json`.
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import of [A]/[E]'s composed opponent + the DR-2 stream cortex, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102. **Warriner AROUSAL is EVAL-ONLY ground truth, NEVER an input** (asserted byte-identical in code — corrupting the arousal ground-truth leaves every arousal read unchanged).

## What this tested (the named surpass from [A], and why it is NOT another valence tweak)

<!--derived-->

[A] (`2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md`) proved — with an ORACLE ceiling — that graded valence STRENGTH is an INFORMATION boundary of the sparse ~10-primary valence-SIGN conditioning channel: no third-factor magnitude (contingency, graded US intensity, even oracle |Warriner|-US) recovers per-concept intensity. Its named surpass #2 was a **SEPARATE graded AROUSAL/intensity channel** — biologically a DISTINCT dimension from valence sign, carried by separate systems (Russell 1980 circumplex; Kandel 6e Ch.40: the noradrenergic locus ceruleus modulates overall AROUSAL/alertness/attention, an ascending modulatory system separate from the dopaminergic reward pathways; phasic LC bursts precede responses to SALIENT stimuli regardless of reward sign). We realize it from the **SAME** self-organized innate-primary conditioning stream, as the orthogonal partner of the existing opponent:

- **VALENCE SIGN** = the reinforcer **DIFFERENCE** `s_c = (n_pos − n_neg)/(n_pos + n_neg)` (the [E] opponent, r≈0.5).
- **AROUSAL** = the reinforcer **ENGAGEMENT MAGNITUDE**, sign-agnostic. Primary emergent read = the interoceptive-salience contingency `A_c = ipos + ineg`, `ipos = n_pos/total_ctx`, `ineg = n_neg/total_ctx` (the FRACTION of a concept's contextual company that is bodily reinforcers, frequency-robust), with a **per-polarity GAIN normalization** (divide each engagement by its population mean = equal opponent-input gain) so the SUM does not leak whichever polarity's reinforcers dominate the corpus.

SUM vs DIFFERENCE on the identical experience ⇒ the arousal read is genuinely a **different operation** than the valence-sign opponent, and by construction tracks INTENSITY not SIGN. Warriner AROUSAL scores the held-out prediction (EVAL only). Three alternative emergent sources are reported (unbalanced contingency, raw reinforcer drive `n_pos+n_neg`, RAW-PPMI code-magnitude L2, context-distribution entropy).

## Result (6-seed) — G1/G3/G5 PASS, G2/G4 FAIL ⇒ BOUNDARY (a strong, near-GO one)

_Per-seed rounded from the cited 6seed JSON `per_seed[]`; means are its `means`. Warriner arousal is EVAL-only._
<!--derived-->

| seed | AROUSAL r (held) | corr(A, sign) | corr(A, \|val\|) | permute perm-p | strength sign-only | strength +arousal | lift |
|---|---|---|---|---|---|---|---|
| 42 | +0.256 | −0.218 | +0.228 | 0.030 | +0.296 | +0.330 | +0.034 |
| 43 | +0.315 | −0.375 | +0.179 | 0.005 | +0.022 | +0.152 | +0.130 |
| 44 | +0.120 | −0.243 | +0.127 | 0.124 | +0.111 | +0.169 | +0.057 |
| 100 | +0.248 | −0.262 | +0.046 | 0.010 | +0.316 | +0.256 | −0.060 |
| 101 | +0.183 | −0.165 | +0.239 | 0.050 | +0.177 | +0.215 | +0.038 |
| 102 | +0.467 | −0.362 | +0.105 | 0.005 | +0.303 | +0.289 | −0.014 |
| **mean** | **+0.265** | **−0.271** | **+0.154** | — | **+0.204** | **+0.235** | **+0.031** |

**GO gate (pre-registered):** G1 arousal-predicts (mean r≥0.25 AND every seed≥0.10) **PASS** (+0.265, min +0.120). G2 arousal⊥sign (|corr|<0.30 ALL seeds) **FAIL** (4/6; seeds 43, 102 at −0.375/−0.362). G3 no-source lesion collapses (|r|<0.15) **PASS** (+0.000, all seeds). G4 permute beaten (perm-p<0.05 ALL seeds) **FAIL** (5/6; seed 44 at 0.124 — the weakest-signal seed). G5 strength lift (combined≥sign+0.03) **PASS** (+0.031; combined +0.235 > sign +0.204). **GO iff G1..G5 ⇒ BOUNDARY** (G2, G4 fail).

## What HOLDS, what STOPS, and the honest diagnosis of each residual

<!--derived-->

**HOLDS — the separate emergent arousal channel is real:**
- **It predicts held-out arousal.** Mean held-out r=+0.265 (all-reinforced +0.224), every seed ≥+0.120. The reinforcer-ENGAGEMENT SUM carries genuine arousal information the valence-SIGN DIFFERENCE channel discards.
- **It is EMERGENT from experienced reinforcer engagement, not a lookup.** No-source lesion (zero the reinforcer co-occurrence) collapses the read to **+0.000 in all 6 seeds** (100% of the effect is attributable to the source, 0% in the control); permute beaten in 5/6 (perm-p<0.05). **Warriner-arousal-free asserted in code**: every arousal-source function takes no arousal argument; corrupting the arousal ground-truth leaves the reads byte-identical.
- **It tracks INTENSITY.** corr(A, |valence|) is POSITIVE in all 6 seeds (mean +0.154) — the read rises with valence extremity, the definitional arousal signature.
- **It adds real graded-STRENGTH info ([A]'s residual).** Combining the emergent valence-SIGN read + the emergent arousal read recovers graded valence-STRENGTH at r=+0.235 vs sign-only +0.204 (lift +0.031; positive in 4/6 seeds, up to +0.130). The arousal channel supplies intensity the sign channel cannot.

**STOPS — and each residual is DIAGNOSED, not asserted:**
- **G2 (full sign-orthogonality) holds in 4/6, not 6/6.** Seeds 43, 102 retain corr(A, sign)≈−0.37. **Diagnosis:** partly the LABELS' OWN valence-arousal asymmetry — the Warriner arousal ground-truth itself anti-correlates with valence sign at corr(a_true, sign)=**−0.176** (negative words are genuinely more arousing in this child-story lexicon), so a PERFECT arousal predictor would also read ≈−0.18; the residual over that (−0.27 vs −0.18) is a polarity-frequency confound the global per-polarity gain only halves (the sign-balancing lever cut seed-102's leak from −0.619 raw → −0.362, and flipped G5 from fail to pass). At n≈64 held concepts (SE≈0.12) the two failing seeds are within sampling range of the labels' −0.18. The bar was pre-registered at 0.30 and 2 seeds miss it — so it is an honest BOUNDARY — but the miss is largely the affect labels' intrinsic circumplex asymmetry plus finite-sample noise, not a mechanism failure.
- **G4 (permutation significance in ALL seeds) holds in 5/6.** The one miss is seed 44 (perm-p=0.124), which is exactly the weakest-arousal-signal seed (r=+0.120) — its arousal read is too weak to clear the permutation null on ~64 concepts. Consistent with a weak-but-real signal, not a broken instrument.
- **G5 passes but the arousal contribution is SMALL.** The +0.031 lift is real but the combined strength (+0.235) is 86.9% attributable to the sign-magnitude feature (which, under the [M] magnitude-supervised ridge read-out, already reaches +0.204, near [M]'s +0.29 full-code ceiling); arousal owns only ~13% of the combined effect. **The arousal channel adds a genuine but modest increment toward the graded-strength ceiling; it does not, alone, close [A]'s ≈0.19 residual.**

**Reframe (diagnosed):** a separate emergent arousal dimension EXISTS and is recoverable from the reinforcer-engagement SUM of the very same innate-primary conditioning — the valence⊥arousal circumplex separation is realized emergently (not a Warriner-arousal lookup). Its arousal-prediction (r≈0.27) and its incremental strength contribution (+0.03) are real but modest; full sign-orthogonality is limited by the affect labels' own valence-arousal asymmetry, and the interoceptive-salience proxy (co-occurrence with ~10 innate reinforcers over child stories) is an information-thin arousal signal.

## The next mechanism (the boundary's surpass — NOT the Warriner-arousal lookup)

<!--derived-->

The channel is real but the CONTINGENCY proxy is thin, so the surpass is a RICHER emergent arousal source, not a scaling of this one:
1. **A spiking LC-like arousal population** whose rate = the total reinforcer drive (the fully-spiking form of this rate-level read) — the direct next rung, and the brain-based version of the interoceptive SUM.
2. **An autonomic / bodily-state proxy** — a physiological-salience signal tied to the reinforcers' MAGNITUDE of bodily activation (heart-rate/arousal-surge analogue), genuinely sign-independent by construction (removes the residual polarity-frequency leak the corpus SUM retains).
3. **Broader reinforcer COVERAGE** — [E]'s ablation showed count-of-primaries lifts salience; a larger, arousal-diverse innate-reinforcer set would give the engagement SUM more intensity resolution.

This is an affect-DIMENSION availability boundary (the emergent arousal PROXY is thin), NOT hidden-layer credit assignment — the deep-credit-on-spikes negatives do not bear on it, and that refuted rule is NOT re-proposed.

## Honest residuals (brutally)

<!--derived-->

1. **It is a BOUNDARY, not a GO.** G2 (2 seeds) and G4 (1 weak seed) fail the pre-registered bars. I do NOT relabel this as GO — but it is a NEAR-GO characterization of a genuinely new, emergent affect dimension.
2. **The arousal channel's strength contribution is modest** (+0.031 lift, ~13% of the combined read). It does not close [A]'s graded-strength residual on its own; the sign-magnitude ridge read-out still dominates.
3. **The interoceptive-salience proxy is thin** — co-occurrence with ~10 innate reinforcers over child stories predicts arousal at only r≈0.27 (a richer bodily/autonomic signal is the surpass).
4. **~10 innate primary SIGNS remain host-supplied** (the faithful floor; world+body boundary). The arousal MAGNITUDE is emergent from co-occurrence; the ±1 primary signs are not.
5. **Rate-level numpy read** (the codes are the spiking-validated stream cortex; a fully-spiking LC arousal population is the named next rung). **Standalone de-risk bridge** — `build_one_brain` fold-in pending.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **Warriner-arousal-free (asserted, not commented):** every arousal-source function takes no arousal argument; corrupting the Warriner arousal ground-truth leaves each read byte-identical; the no-source lesion collapse (+0.000, all seeds) gives the assertion teeth.
- **AROUSAL ⊥ VALENCE measured both ways:** corr(A, sign) reported (the ⊥ requirement) AND corr(A, |valence|) reported (the intensity it SHOULD track); the labels' own corr(a_true, sign)=−0.176 recorded so the residual sign-leak is interpreted honestly against it.
- **No-source lesion → collapse; permute → collapse** (permutation test, 200 draws, per seed). **6 seeds** 42/43/44/100/101/102 (smoke first: the tiny 8k-story smoke read +0.228, the 60k is authoritative).
- **Strength lift as a controlled A/B:** sign-only vs sign+arousal, same magnitude-supervised ridge read-out, fit on TRAIN / evaluated on HELD; `attributable_to` prints that arousal owns ~13% of the combined effect (the honest split, not the total).

## Sources

- Russell, J.A. (1980), J. Pers. Soc. Psychol. 39(6):1161 — the circumplex model of affect: valence and arousal are two SEPARATE, orthogonal dimensions. Barrett, L.F. & Bliss-Moreau, E. (2009) — affect as valence + arousal.
- Kandel, *Principles of Neural Science* 6e, Ch.40 (The Brain Stem) — the noradrenergic locus ceruleus modulates overall AROUSAL/alertness/attention (an ascending modulatory system separate from the dopaminergic reward pathways); phasic LC bursts precede responses to salient stimuli, tonic mode tracks behavioral flexibility (Aston-Jones/Cohen). Grounds arousal as an intensity/salience channel distinct from valence sign.
- Bayer, H.M. & Glimcher, P.W. (2005), Neuron 47(1):129 — graded reward-magnitude DA (the intensity [A] ruled OUT of the valence-SIGN channel; here sought in a separate arousal dimension).
- Namburi, P., Tye, K.M. et al. (2015, Nature) — opposing BLA valence-coding populations (the DIFFERENCE channel this arousal SUM is orthogonal to).
- [A] `2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md` (its named surpass #2); [E] `2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md` (the self-organized valence-sign opponent this reuses).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_arousal_channel_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_arousal_channel_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_affect_arousal_channel_6seed.json
```
