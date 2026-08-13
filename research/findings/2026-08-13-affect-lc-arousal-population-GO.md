---
type: finding
status: go
lane: A (affect / emotion keystone)
date: 2026-08-13
mechanism: affect-lc-arousal-population
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_lc_arousal_population_derisk.py
artifacts:
  - research/findings/raw/_affect_lc_arousal_population_6seed.json
  - research/findings/raw/_affect_lc_arousal_population.json
builds_on:
  - research/findings/2026-08-13-affect-arousal-channel-BOUNDARY.md
  - research/findings/2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md
---

# A RICHER emergent arousal source — a spiking LC-like POPULATION that INTEGRATES two convergent salience afferents (reinforcer ENGAGEMENT + cortical DRIVE magnitude) of the self-organized code — STRENGTHENS the arousal channel: it predicts held-out Warriner AROUSAL at 6-seed mean r=+0.311, BEATING [R]'s single-afferent engagement-SUM (r=+0.265) in 6/6 paired seeds, and lifts graded valence-STRENGTH by +0.047 (> [R]'s +0.031). GO on the mission target — WITH three honest narrowings (below). <!--derived-->

<!--derived-->
Headline numbers are the 6-seed JSON `means` block; per-seed values are in `research/findings/raw/_affect_lc_arousal_population_6seed.json` `per_seed[]`. All correlations are held-out (EVAL-only Warriner arousal).

**Runner:** `research/runners/_affect_lc_arousal_population_derisk.py`
**Raw:** `research/findings/raw/_affect_lc_arousal_population_6seed.json` (+ provenance sidecar), smoke `..._population.json`.
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import of [R]'s corpus build + emergent afferents + the composed valence opponent, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102. **Warriner AROUSAL is EVAL-ONLY, NEVER an input** (asserted byte-identical in code; the no-source lesion collapse gives the assertion teeth).

## What this tested (the named surpass from [R]) — and the mechanism

<!--derived-->

[R] (`2026-08-13-affect-arousal-channel-BOUNDARY.md`) established that valence⊥arousal separates EMERGENTLY from the same innate-primary conditioning: valence SIGN = the reinforcer DIFFERENCE, arousal = the reinforcer ENGAGEMENT SUM `(n_pos+n_neg)/total_ctx`. That single-afferent contingency proxy predicts held-out arousal at r=+0.265 but is INFO-THIN. [R]'s named surpass #1: a RICHER emergent source — a spiking LC-like population whose graded rate INTEGRATES multiple salience afferents (Aston-Jones & Cohen 2005 adaptive-gain: the LC is a many-input salience integrator; Kandel 6e Ch.40: low tonic rate when drowsy, graded tonic when alert, PHASIC bursts to salient stimuli regardless of reward sign).

We built exactly that, FULLY EMERGENT FROM CORPUS (no new host input): a heterogeneous-threshold LIF population (N_lc=64) driven by the CONVERGENCE of two afferents, both UNAMBIGUOUSLY arousal-POSITIVE by biological role (so their fixed +gain is not peeked from the arousal labels):
- **a1 = interoceptive ENGAGEMENT** = [R]'s balanced `(n_pos+n_neg)/total_ctx` (reinforcer-salience contingency; r~0.265 alone).
- **a2 = cortical DRIVE magnitude** = L2 norm of the RAW PPMI code (how hard the concept drives cortex; r~0.227 alone, partly INDEPENDENT of a1).

Each afferent is population-z-scored (equal-gain), summed with FIXED +weights, min-max mapped to a tonic..phasic input CURRENT, fed to the LIF population; the graded POPULATION spike-rate is the arousal read (phasic-to-salient + tonic baseline). The operating point (i_tonic just above mean rheobase, phasic span ~tonic..2.5× rheobase) is FROZEN from LC dynamic-range biology, NOT tuned to the arousal labels.

## Result (6-seed) — GO on the mission target (r>0.27 AND lift>0.031), robustly beats [R] 6/6 <!--derived-->
<!--derived-->
The comparison value +0.031 is [R]'s strength-lift (from `research/findings/raw/_affect_arousal_channel_6seed.json`); this build's lift is +0.047.

_Per-seed rounded from the cited 6seed JSON `per_seed[]`; means are its `means`. Warriner arousal is EVAL-only._
<!--derived-->

| seed | LC r (held) | [R] engage r | Δ (paired) | corr(LC,sign) | corr(LC,\|val\|) | perm-p | strength sign | +LC | lift |
|---|---|---|---|---|---|---|---|---|---|
| 42 | +0.300 | +0.256 | +0.044 | −0.297 | +0.196 | 0.020 | +0.296 | +0.315 | +0.019 |
| 43 | +0.331 | +0.315 | +0.015 | −0.418 | +0.226 | 0.010 | +0.022 | +0.181 | +0.159 |
| 44 | +0.223 | +0.120 | +0.103 | −0.425 | +0.194 | 0.040 | +0.111 | +0.229 | +0.118 |
| 100 | +0.294 | +0.248 | +0.046 | −0.324 | +0.063 | 0.015 | +0.316 | +0.258 | −0.058 |
| 101 | +0.195 | +0.183 | +0.012 | −0.293 | +0.050 | 0.050 | +0.177 | +0.216 | +0.040 |
| 102 | +0.523 | +0.467 | +0.056 | −0.523 | +0.162 | 0.005 | +0.303 | +0.306 | +0.003 |
| **mean** | **+0.311** | **+0.265** | **+0.046** | **−0.380** | **+0.168** | — | **+0.204** | **+0.251** | **+0.047** |

**Mission GO gate (pre-registered):** G1 LC-predicts (mean r≥0.27 AND every seed≥0.12) **PASS** (+0.311, min +0.195). G1b LC BEATS [R]'s engagement-SUM paired in ALL seeds **PASS** (6/6, Δ +0.046, every seed positive). G3 no-source lesion collapses (|r|<0.15) **PASS** (+0.000 all seeds). G5 strength lift ≥+0.031 AND combined>sign **PASS** (+0.047; combined +0.251 > sign +0.204). **GO iff G1·G1b·G3·G5 ⇒ GO.** Reported characterization (NOT part of the mission GO — these were [R]'s label-limited residuals): G2 arousal⊥sign |corr|<0.30 holds in **2/6** (regressed vs [R]'s 4/6); G4 permute beaten in **6/6**.

## What HOLDS (verify-go survived), and the THREE honest narrowings

<!--derived-->

**HOLDS — a genuinely richer, brain-based arousal source that strengthens the channel:**
- **It beats the engagement-SUM, seed-robustly.** LC r +0.311 vs [R]'s +0.265, positive Δ in 6/6 seeds (min Δ +0.012). Excluding the strongest seed (102), LC mean is still +0.268 vs [R]'s +0.224 — the paired improvement does NOT depend on one seed. Reproducible: seed 42 re-ran byte-identical (+0.300 twice).
- **The gain is genuine COMPLEMENTARY integration, not a relabel.** LC combo (+0.311) exceeds BOTH single afferents (engage +0.265, code-drive +0.227). The cortical-drive afferent carries arousal variance the engagement contingency discards; the LC pools them.
- **It is EMERGENT and not a mass/frequency/chance artifact.** No-source lesion (zero the afferent drive → constant input → constant population rate) collapses the read to **+0.000 in all 6 seeds**; permute beaten in **6/6** (perm-p<0.05). Warriner-arousal-free asserted in code (byte-identical under corrupted ground-truth; every afferent fn takes no arousal argument).
- **It tracks INTENSITY** (corr(LC,|val|)=+0.168, positive all seeds) and **lifts graded valence-STRENGTH MORE than [R]** (combined +0.251 vs sign-only +0.204, lift +0.047 > [R]'s +0.031).
- **It is BRAIN-BASED** — an actual heterogeneous-threshold LIF spiking population ([R]'s named next rung), realized with NO new host input.

**STOPS — three narrowings, each measured, none buried:**
1. **The LIF nonlinearity is rank-NEUTRAL; the win is the multi-afferent CONVERGENCE.** LC population r (+0.311) ≈ the linear z-sum of the same afferents (+0.308). The spiking population is the faithful REALIZATION (the brain-based next rung), and it is correlation-neutral vs the rate-level combination — the strengthening comes from integrating engagement + cortical-drive, not from the spiking transfer per se. Honest, not overclaimed.
2. **Orthogonality to valence sign REGRESSED.** corr(LC,sign) = −0.380 (vs [R]'s −0.271); only 2/6 seeds clear |corr|<0.30 (vs [R]'s 4/6). **Diagnosis:** the added cortical-drive afferent (distinctive words) itself anti-correlates with valence sign in this child-story lexicon — distinctive words skew aversive — compounding the labels' own corr(a_true,sign)=−0.176 asymmetry [R] measured. So the arousal-PREDICTION strengthened but the circumplex SEPARATION did NOT — it degraded. This build strengthens the arousal channel's PREDICTIVE STRENGTH, not its orthogonality; I do not claim improved separation.
3. **The strength lift is modest and seed-variable.** The combined strength (+0.251) is ~81% attributable to the sign-magnitude feature; the LC owns only ~19%. Per-seed lift ranges −0.058 (seed 100) to +0.159 (seed 43); 2/6 seeds are ≤+0.003. Same character as [R]'s modest +0.031 — a real but small increment toward the graded-strength ceiling, not a closure of [R]/[A]'s ~0.19 residual.

**Also reported (NOT headlined):** a 3rd afferent (distinctiveness = −context-entropy, contestable sign) HURT (lc-3afferent +0.284 < +0.311), so the 2-afferent read is the right primary. The supervised-afferent CEILING (train-fit weights, held-eval) reached +0.249 — near the fixed-weight read, so the fixed equal-gain is not badly suboptimal; the info the corpus afferents carry is ~r0.25-0.31, consistent with a real but bounded corpus arousal signal.

## The bodily-magnitude proxy — suggestive of the embodiment direction, but NOT a clean result (leak, reported)

<!--derived-->

[R]'s named surpass #2 was an AUTONOMIC/bodily-state proxy: an innate per-primary bodily-activation MAGNITUDE (distinct from the ±1 sign), a host WORLD/BODY floor. We assigned each of the ~20 primaries a magnitude by AUTONOMIC CATEGORY (sympathetic defensive/nociceptive/startle & intense affiliative = high; parasympathetic consummatory/comfort = low), FROZEN before results, then read arousal = engagement WEIGHTED by that magnitude. It is the STRONGEST single source (r=+0.317, > the LC's +0.311). **But it is NOT clean:** the leak audit — corr(assigned magnitude, the primaries' OWN Warriner arousal) — is **+0.63 mean (0.38–0.80 per seed)**, so the assigned magnitudes substantially correlate with real arousal norms. Its edge over the LC therefore does NOT cleanly demonstrate "the body carries the info". It also does not ADD over the corpus LC (LC+bodily supervised = +0.298 < LC alone +0.311). **Honest reading:** the bodily-magnitude direction is SUGGESTIVE of the embodiment hypothesis [R] anticipated, but a clean test needs autonomic magnitudes derived WITHOUT arousal-norm knowledge (e.g. measured physiological/interoceptive responses from the body interface) — which is legitimately host per brain-based-only (the body provides the interoceptive signal, the brain reads it). We do NOT headline it and do NOT relabel it as a win.

## Reframe (diagnosed) and the next mechanism

<!--derived-->

[R]'s corpus-information boundary is PARTIALLY surpassed: a richer FULLY-EMERGENT corpus source (multi-afferent salience convergence, in a spiking LC population) lifts held-out arousal prediction 0.265→0.311 (6/6 paired) and strength-lift 0.031→0.047, with NO new host input — the arousal channel is STRENGTHENED. But the residual is real and points where [R] predicted: the deeper arousal RESOLUTION (and a CLEAN circumplex orthogonality) likely still needs a genuine BODILY/INTEROCEPTIVE input, because (a) corpus afferents plateau at r~0.25-0.31 (the supervised ceiling confirms the ceiling, not just a bad readout), and (b) the only source that beats the corpus LC is the bodily-magnitude proxy, whose clean version requires arousal-norm-free autonomic magnitudes from the world/body interface. **Next mechanism:** the LC population fed by a REAL interoceptive/autonomic afferent from an embodied loop (the host-legitimate body signal), NOT a hand-assigned magnitude and NEVER the Warriner-arousal lookup. This is an affect-DIMENSION availability boundary being surpassed rung by rung, NOT hidden-layer credit assignment.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **Warriner-arousal-free (asserted, not commented):** every afferent fn takes no arousal argument; corrupting the Warriner arousal ground-truth leaves the LC read byte-identical; the no-source lesion collapse (+0.000, all seeds) gives the assertion teeth.
- **No-source lesion → collapse (deterministic LIF):** zeroing the afferent drive makes the population input constant → a zero-variance read → r=+0.000 all seeds. (Input noise was removed — it faked a nonzero lesion correlation by injecting spurious per-concept variance; the deterministic heterogeneous-threshold LIF is still a genuine spiking population.)
- **Permute → collapse** (permutation test, 200 draws per seed; beaten 6/6). **arousal⊥valence measured both ways** — corr(LC,sign) AND corr(LC,|val|) reported, interpreted against the labels' own corr(a_true,sign)=−0.176.
- **BEATS-[R] is a clean paired A/B:** [R]'s engagement-SUM and the LC read are computed on the IDENTICAL seed/split/corpus/held-set — the only difference is the read; both arms in a responsive range (r 0.12–0.52, not floored/ceilinged). **Fixed afferent +weights + operating point (NOT fit to arousal)**; only the reported CEILING fits on the disjoint TRAIN split. **6 seeds**, smoke first (8k read +0.272; the 60k is authoritative), determinism spot-checked (byte-identical rerun).
- **Bodily-magnitude leak reported transparently** — corr(assigned magnitude, primaries' own Warriner arousal)=+0.63, so the proxy is not claimed as a clean emergent/host-floor win.

## Sources

- Russell, J.A. (1980), J. Pers. Soc. Psychol. 39(6):1161 — the circumplex: valence and arousal are two separate, orthogonal dimensions. Barrett & Bliss-Moreau (2009) — affect as valence + arousal.
- Kandel, *Principles of Neural Science* 6e, Ch.40 — the noradrenergic locus coeruleus is the ascending AROUSAL population (low tonic rate when drowsy, graded tonic when alert; PHASIC bursts to salient stimuli regardless of reward sign).
- Aston-Jones, G. & Cohen, J.D. (2005), Annu. Rev. Neurosci. 28:403–450 — the LC integrates convergent afferents signaling salience/utility; phasic vs tonic modes; adaptive gain. Grounds the LC as a MANY-INPUT salience integrator (the multi-afferent convergence this build realizes).
- Namburi, P., Tye, K.M. et al. (2015, Nature) — opposing BLA valence-coding populations (the DIFFERENCE channel this arousal SUM is orthogonal to).
- [R] `2026-08-13-affect-arousal-channel-BOUNDARY.md` (its named surpass #1); [A] `2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md`.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_lc_arousal_population_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_lc_arousal_population_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_affect_lc_arousal_population_6seed.json
```
