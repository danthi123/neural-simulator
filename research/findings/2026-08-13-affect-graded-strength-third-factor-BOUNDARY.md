---
type: finding
status: boundary
lane: A (affect / emotion keystone)
date: 2026-08-13
mechanism: affect-graded-strength-third-factor
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_graded_strength_third_factor_derisk.py
artifacts:
  - research/findings/raw/_affect_graded_strength_third_factor_6seed.json
  - research/findings/raw/_affect_graded_strength_third_factor.json
builds_on:
  - research/findings/2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md
  - research/findings/2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md
  - research/findings/2026-08-13-affect-appraisal-origin-self-organizes-from-reinforcement-6seed-GO.md
---

# A graded DA-magnitude third factor does NOT close the affect graded-STRENGTH residual — and the ceiling analysis proves WHY: graded valence STRENGTH is an INFORMATION boundary of the sparse ~10-primary conditioning channel, not a third-factor scaling. The SIGN retirement HOLDS (6-seed).

<!--derived-->

**Runner:** `research/runners/_affect_graded_strength_third_factor_derisk.py`
**Raw:** `research/findings/raw/_affect_graded_strength_third_factor_6seed.json` (+ provenance sidecar), smoke `..._third_factor.json`.
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import of [E]'s composed runner, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102.

## What this tested (the named surpass from [E]+[M])

<!--derived-->

Two lanes on 2026-08-13 localized the affect residual precisely. [E]
(`2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md`): deriving the spiking opponent V+/V- weights FROM
the self-organized conditioning map (NO Warriner) RETIRES the seed for held-out valence SIGN (r=+0.508) but graded
STRENGTH underperforms (salience |differential|~|valence| r=+0.10 vs the magnitude-supervised ridge's 0.27). [M]
(`2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md`) PROVED the READ-OUT is not the bottleneck (ridge +
point-soma already hits 0.327). Both named the same next lever: a **graded reinforcement-STRENGTH third factor** —
the DA/US gate driven by reward MAGNITUDE (Bayer & Glimcher, 2005: midbrain DA fires GRADED with reward magnitude,
not just sign), so the third factor `s_c` becomes a graded associative strength instead of the SATURATING
Rescorla-Wagner asymptote `s_c = (n_pos-n_neg)/(n_pos+n_neg)` (which divides out the magnitude, encoding sign
robustly but strength weakly). This runner builds that third factor and measures whether graded STRENGTH lifts
toward the ridge WHILE the SIGN r holds and the weights stay Warriner-free.

## Result (6-seed) — the graded third factor does NOT lift strength; even an ORACLE US-magnitude does not

_Per-seed values rounded from the cited 6seed JSON `per_seed[]`; means are its `means`. Warriner is EVAL-only._
<!--derived-->

Six arms share the SAME reinforced set + the SAME robust count-based sign; they differ ONLY in the per-concept
third-factor magnitude (so any STRENGTH delta is purely the magnitude term). All read the SPIKING opponent
differential off `cp_firing_states`:

| arm (6-seed mean) | third-factor magnitude source | STRENGTH r (\|d\|~\|val\|) | SIGN r |
|---|---|---|---|
| **ridge-Warriner** (reference target) | per-CONCEPT Warriner (magnitude-SUPERVISED) | **+0.288** | +0.728 |
| **boundary** ([E]'s method) | \|RW purity\| = \|(n+ − n−)/(n+ + n−)\| (saturating) | +0.101 | +0.507 |
| **graded PPMI** (pre-registered gated) | \|Σ sign·PPMI(concept,primary)\| (contingency) | **+0.082** | +0.509 |
| ORACLE innate-US-magnitude (ceiling/cheat) | magnitude-weighted RW, primaries = \|Warriner\| | +0.081 | +0.509 |
| Warriner-free US-magnitude (peakedness) | magnitude-weighted RW, primary co-occ concentration | −0.005 | +0.445 |
| sign-only (unit magnitude) | 1 | −0.013 | ~0.51 |

**The runner's own verdict is BOUNDARY** (G2/G3a/G5 fail; G1/G3b/G4 pass). Per-seed STRENGTH r [graded, boundary,
oracle-US, free-US | ridge]:

| seed | graded PPMI | boundary | oracle-US | free-US | ridge | SIGN r (graded) |
|---|---|---|---|---|---|---|
| 42 | +0.168 | +0.252 | +0.062 | +0.243 | +0.277 | +0.495 |
| 43 | +0.168 | −0.136 | +0.144 | −0.238 | +0.375 | +0.650 |
| 44 | +0.255 | +0.199 | +0.113 | +0.103 | +0.211 | +0.560 |
| 100 | +0.204 | +0.323 | +0.372 | +0.272 | +0.224 | +0.654 |
| 101 | −0.040 | +0.060 | +0.055 | −0.149 | +0.393 | +0.368 |
| 102 | −0.267 | −0.095 | −0.259 | −0.261 | +0.246 | +0.326 |

## What HOLDS, what STOPS, and the airtight ceiling that reframes the residual

<!--derived-->

**HOLDS — the SIGN retirement is preserved (confirms [E]):**
- Graded held-out SPIKING SIGN r = **+0.509** (every seed ≥ +0.326 ≥ the 0.25 floor). The graded third factor does
  not trade away the sign retirement. G1 PASS.
- **No-conditioning lesion** (`s_c := 0`) collapses the read to **+0.000** (all seeds): the weights come from
  EXPERIENCE, not Warriner. G4 PASS. **Warriner-free asserted in code:** `graded_da_magnitude` takes no Warriner
  argument; corrupting `s_true` leaves the weights byte-identical.
- **G3b** (magnitude isolation): graded (+0.082) beats unit-magnitude sign-only (−0.013) by +0.095 — the PPMI
  magnitude IS doing something (it lifts strength above unit), it just does not beat the purity.

**STOPS — the graded third factor does NOT lift graded STRENGTH:**
- Graded PPMI STRENGTH r = **+0.082**, BELOW the boundary's +0.101 (lift −0.019; positive in only 2/6 seeds). The
  pre-registered contingency third factor is a NEGATIVE. G2 FAIL.
- **The permute-magnitude control is beaten in only 1/6 seeds** — because there is no reliable per-concept graded
  magnitude signal to beat the null with. G3a FAIL (consistent with the negative, not an instrument failure).
- Permute-code holds 5/6 (seed 102, the known weak-draw seed from [E], p=0.294). G5 FAIL on that one seed; the SIGN
  correlation bar still passes.

**THE AIRTIGHT CEILING — the residual is NOT a third-factor-scaling problem at all:**
- The **ORACLE** arm (a declared CHEAT: each innate primary weighted by its OWN |Warriner| intensity, the literal
  Bayer-Glimcher "reward magnitude is a property of the US" reading, via a purity-preserving magnitude-weighted RW)
  reaches STRENGTH r = **+0.081** — the SAME as the boundary, wildly variable (−0.259…+0.372). Giving the innate
  primaries their TRUE intensities does not lift strength at the authoritative 60k-story scale. (A 1-seed 8k-story
  smoke showed oracle +0.476 — a tiny-corpus artifact that the 6-seed overturns; recorded so the smoke is not
  mistaken for the result.)
- The **Warriner-free peakedness** proxy reads −0.005 (corpus statistics do not recover intensity).
- Only the **ridge** (per-CONCEPT magnitude supervision) reaches +0.288. The strength information IS present in the
  self-organized CODE geometry — the ridge extracts it — but the single-step ~10-primary conditioning WRITE cannot,
  regardless of how the third factor is scaled (unit, purity, PPMI-contingency, or oracle-US-intensity).

**Reframe (diagnosed, not asserted):** graded valence STRENGTH is an **information boundary of the sparse innate-
primary conditioning channel**, not a saturation of one scalar. The channel carries valence SIGN (r≈0.5) but not
per-concept graded STRENGTH (≈0.10) — and NO third-factor magnitude (contingency, graded US intensity, even oracle)
recovers it, because the missing quantity is per-concept intensity that ~10 reinforcers + child-story co-occurrence
do not contain. This is a STRONGER, more precise statement than [E]/[M]'s "the RW asymptote saturates": the
saturation is real but rescaling it is not the fix.

## The next mechanism (the boundary's surpass, evidence-based — NOT the refuted deep-credit rule)

<!--derived-->

The ceiling rules OUT the third-factor family, so the surpass is not another scaling of the same write. The missing
per-concept intensity must come from a channel the sparse valence-sign conditioning does not carry:
1. **Reinforcer COVERAGE** — [E]'s own ablation showed 12–16 innate SIGNS (not intensities) lift salience to ≈0.20
   (a coverage effect, non-monotonic). This finding's oracle result explains WHY intensity-per-primary does not
   help while count-of-primaries does: the channel is information-starved, not miscalibrated.
2. **A separate graded AROUSAL / intensity channel** — biologically, affect INTENSITY (arousal) is a dimension
   DISTINCT from valence sign, carried by separate systems (LC-noradrenergic / interoceptive arousal vs VTA-BLA
   valence). A single valence-sign opponent cannot encode magnitude; a second, parallel magnitude channel
   conditioned on bodily-arousal reinforcers is the faithful surpass.
3. **Higher-order (concept↔concept) conditioning** — the current write is single-step (concept↔primary); graded
   strength could accumulate by propagation/amplification through already-valenced concepts (the self-organized
   analogue of DR-2's retired label-propagation).

This is a SINGLE-LAYER associative-write / information-availability boundary, NOT hidden-layer credit assignment —
the deep-credit-on-spikes negatives do not bear on it (`2026-07-22-gap4-real-issue-NOT-dendrites.md`), and that
refuted rule is NOT re-proposed.

## Honest residuals (brutally)

<!--derived-->

1. **The pre-registered graded PPMI-contingency third factor is a NEGATIVE** (STRENGTH +0.082 ≤ boundary +0.101).
   The named surpass from [E]/[M], as a third-factor magnitude, does not close the residual.
2. **The graded-innate-US-magnitude axis is ruled out by its own ORACLE** (+0.081 at 6-seed). The smoke's optimistic
   +0.476 was a tiny-corpus fluke — recorded, not hidden.
3. **SIGN retires; STRENGTH does not.** The Warriner-free ceiling for graded strength through this channel is the
   purity's ≈0.10; the ridge's ≈0.29 requires per-concept magnitude supervision. The residual gap is ≈0.19.
4. **~10 innate primary SIGNS remain host-supplied** (the faithful floor; a 140→~10 compression, not a removal).
5. **Rate-level numpy Hebbian map** (the codes are the spiking-validated stream cortex; a fully-spiking write is a
   separate rung). **Standalone de-risk bridge** — `build_one_brain` fold-in pending.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **Warriner-free (asserted):** `graded_da_magnitude` + `selforg_opponent_weights` take no Warriner argument;
  corrupting `s_true` leaves the weights byte-identical; the no-conditioning collapse (+0.000, all seeds) gives the
  assertion teeth. The ORACLE arm is EXPLICITLY declared a cheat/ceiling and is NEVER part of the Warriner-free claim.
- **Instrument validated by three reference points:** no-conditioning reads exactly +0.000; the boundary arm
  reproduces [E]'s +0.10 strength EXACTLY; the ridge reproduces [M]'s +0.27–0.29 strength target. The negative is
  therefore a real weight-source result, not a broken read.
- **Held-out = own-reinforcement-withheld** (the DR-2 leave-out: the held concept's own `s_c` is not in the map).
- **Sign held FIXED across arms** so any STRENGTH delta is purely the magnitude term (the lever `lever(...)` prints
  MOVED each seed: the graded arm is genuinely not the boundary arm).
- **Permutation controls** (permute-code, permute-magnitude) on the linear read; **6 seeds** 42/43/44/100/101/102
  (smoke first; the 6-seed is authoritative — the smoke's oracle was NOT).

## Sources

- Bayer, H.M. & Glimcher, P.W. (2005), Neuron 47(1):129 — midbrain dopamine neurons encode a quantitative
  (graded-magnitude) reward-prediction error: the proposed graded reinforcement-STRENGTH third factor tested here.
- Rescorla, R.A. & Wagner, A.R. (1972) — the associative-strength ASYMPTOTE; its saturation is why the composed
  `s_c` encodes sign > strength (the residual this arc targets).
- Namburi, P., Tye, K.M. et al. (2015, Nature) — opposing valence-coding BLA populations (the innate V+/V- opponent
  channel the weights inject into).
- [E] `2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md` — the self-organized opponent this composes
  from; [M] `2026-08-13-magnitude-preserving-plateau-readout-BOUNDARY.md` — the read-out is not the bottleneck.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_graded_strength_third_factor_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_graded_strength_third_factor_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_affect_graded_strength_third_factor_6seed.json
```
