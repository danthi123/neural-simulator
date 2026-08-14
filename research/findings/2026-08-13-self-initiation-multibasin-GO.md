---
type: finding
status: live
date: 2026-08-13
mechanism: self-initiated-spontaneous-thought
runner: research/runners/_self_initiation_multibasin_derisk.py
artifacts:
  - research/findings/raw/_self_initiation_multibasin_derisk.json
---

# Self-initiation SELECTS among multiple balanced basins: 6-seed GO — under noise (no cue) the wander visits several disjoint stored concepts, and a curiosity recurrent-gain biases WHICH surfaces (identity-controlled, 66% attributable)

**2026-08-13 (autonomous, GPU/cupy, n_ca3=2000, n_mem=4).** The self-initiated-spontaneous-thought de-risk
([`2026-08-13-self-initiated-spontaneous-thought-GO.md`](2026-08-13-self-initiated-spontaneous-thought-GO.md)) is
6-seed GO but used n_mem=2, which reliably reactivates ONE dominant basin — so it de-risked the STEERING of a single
reactivating thought (identity-controlled novel-vs-familiar on the SAME thought), NOT SELECTION among several
equally-reactivatable concepts. That finding's named next rung #1 is exactly this lane. This de-risks it: store
several DISJOINT balanced concepts, drive the noise-seeded wander with NO cue, and show it VISITS multiple distinct
stored concepts over a session, each coherent, with a curiosity neuromodulatory gain biasing WHICH surfaces more.
**Functional self-initiation SELECTION correlate only — no claim of phenomenal experience.**

## The mechanism (the surpass = PATTERN SEPARATION for balance, on the two validated organs)

_Config values + readouts from the committed artifact `research/findings/raw/_self_initiation_multibasin_derisk.json`._

<!--derived-->

The gap#5 RANK-1 spontaneous-reactivation substrate (6-seed GO,
[`2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md`](2026-07-22-gap5-RANK1-spontaneous-reactivation-6seed-GO.md))
draws its assemblies with independent random `rng.choice` — they OVERLAP (~29 shared cells for two 240-cell
assemblies at n_ca3=2000), so the strongest overlapping structure wins → one dominant basin (the n_mem=2 limit the
prior finding reported; the same intrinsic-basin dominance our multireferent-WTA scoping
[`2026-06-19-multireferent-wta-biased-competition-scoping.md`](2026-06-19-multireferent-wta-biased-competition-scoping.md)
documented for independent attractors). **THE SURPASS: PATTERN-SEPARATED (DISJOINT) encoding** — partition a random
permutation of CA3 cells into n_mem NON-overlapping equal-size assemblies (dentate-gyrus pattern separation → orthogonal
engrams; McNaughton & Morris 1987; Leutgeb et al. 2007 *Science* "Pattern separation in DG"; Bakker et al. 2008).
Equal size + no shared cells + identical BTSP one-shot encode → INDEPENDENT, co-equal attractor basins, no host thumb
on the scale (verified: max pairwise assembly overlap = **0** every seed). This is the ONLY change from the RANK-1
substrate (`_prepare_balanced`, reuse-by-import of `_build`/`_set_gates`/`_extract_ca3ca3_vec`); NO `sim/` edit.

Under weak NON-SPECIFIC Poisson background (rate 0.015, 1500 pA, dur 10 — the SAME operating point as the RANK-1 GO;
NO cue, 0 external CONTENT drive) each discrete noise-seeded volley ignites WHICHEVER balanced basin its coincidental
within-window overlap favours; the bistable KIR down-state returns the net to silence before the next event. Over a
spontaneous session the wander therefore VISITS DIFFERENT stored concepts (Bouhadjar et al. 2023 *PLoS Comput Biol*
"Coherent noise enables probabilistic sequence replay"; the biased-competition WTA of Desimone & Duncan 1995). Each
concept's NOVELTY (the ENVIRONMENT) maps through the production CURIOSITY organ's SPIKING ASK-pool want to a
proportional NEUROMODULATORY RECURRENT GAIN on its engram (McNamara 2014; Ambrose/Pfeiffer/Foster 2016; Mattar & Daw
2018 need × gain) — a more-novel basin completes from a smaller volley → it wins MORE of the noise-seeded races.

## Result — 6/6 GO (seeds 42, 43, 44, 100, 101, 102), gain-scale 1.0, rest 4000, n_mem=4

_Per-seed values are from the cited committed artifact — verify against the raw JSON. GPU/cupy summation order is not
byte-deterministic, so masses jitter ~3% run-to-run; the GO and every anti-cheat hold with margin._

<!--derived-->

The store holds 4 DISJOINT balanced concepts; NOVELTY is assigned to a RANDOM permutation of concepts per seed (so
intrinsic basin strength is uncorrelated with novelty). Two run-time conditions are read plus a REVERSED anti-curiosity
control and two acids. Curiosity bias is read **identity-controlled**: each concept is seen at three gains {uniform 1.0,
its curiosity gain, its REVERSED gain}, and the SAME novel concepts' visit share HIGH-gain (curiosity-on) vs LOW-gain
(reversed) isolates the gain's causal effect (intrinsic basin strength cancels — same concepts).

| seed | balanced visits (of 4) / entropy | production visits / top1 | coherence member vs rand | novel-share HIGH vs LOW (reversed) | attributable | within-concept dose r | NO-NOISE dwell | STORE-LESION member |
|------|-----------------------------------|--------------------------|--------------------------|-------------------------------------|--------------|------------------------|----------------|---------------------|
| 42   | 3 / 0.78 | 3 / 0.61 | 0.36 vs 0.04 | 0.86 vs 0.49 | 43% | 0.52 | 0 ✓ | 0.00 ✓ |
| 43   | 3 / 0.78 | 3 / 0.77 | 0.38 vs 0.04 | 0.77 vs 0.25 | 67% | 0.73 | 0 ✓ | 0.00 ✓ |
| 44   | 3 / 0.79 | 3 / 0.51 | 0.38 vs 0.04 | 0.51 vs 0.36 | 29% | 0.34 | 0 ✓ | 0.00 ✓ |
| 100  | 3 / 0.80 | 3 / 0.61 | 0.41 vs 0.04 | 0.86 vs 0.18 | 79% | 0.73 | 0 ✓ | 0.00 ✓ |
| 101  | 3 / 0.74 | 3 / 0.77 | 0.35 vs 0.04 | 0.77 vs 0.03 | 96% | 0.73 | 0 ✓ | 0.00 ✓ |
| 102  | 3 / 0.82 | 3 / 0.46 | 0.33 vs 0.04 | 0.46 vs 0.11 | 76% | 0.48 | 0 ✓ | 0.00 ✓ |

**Aggregate (6-seed):** balanced-condition visits mean **3.0** distinct coherent basins (entropy ~0.78 = 3 co-equal
basins); production (curiosity-on) wander visits mean **3.0** (top1 mean 0.62); coherence member **0.37 vs random
0.04** (~9.6× chance); curiosity novel-share HIGH-gain **0.71 vs LOW-gain 0.24** → **66% of the novel-concept surfacing
attributable to the curiosity gain**; within-concept dose-response mean r **0.59**. The runner's OWN Verdict decided
**GO** (all preconditions met).

Every seed → GO on all anti-cheats (each VERIFIED, not asserted):

- **BALANCED / no host thumb.** DISJOINT basins (max pairwise overlap 0), identical per-concept encode, so under
  UNIFORM gain the wander visits ~co-equal basins (entropy 0.74–0.82). The 4 concepts are genuinely equally-selectable,
  not one dominant.
- **SELECTS + COHERENT.** The wander visits multiple distinct concepts, each overlapping its STORED assembly at member
  0.33–0.41 vs random 0.04 (~8–10× chance). **STORE-LESION (NO-ENCODE, same noise+gain) → member 0.00** every seed:
  the content is the learned store, not the noise.
- **CURIOSITY-BIASED (identity-controlled).** The SAME novel concepts surface materially MORE under HIGH gain than
  under the REVERSED (LOW) gain (66% attributable aggregate; 29–96% per seed), and the within-concept dose-response
  (demeaning share per concept to cancel intrinsic strength) is positive (r 0.34–0.73). The bias is the curiosity
  VALUE, not the basin identity.
- **INTERNALLY-GENERATED.** 0 external CONTENT drive (only non-specific Poisson to random CA3-exc cells). **NO-NOISE
  (gain on, noise off) → 0 surfacing** every seed. Plasticity byte-FROZEN during the session every seed.

## Honest scoping (what this does NOT show)

<!--derived-->

- **3 of the 4 basins ignite, not 4/4.** Every seed visits exactly 3 distinct coherent basins (in BOTH balanced and
  curiosity-on); the 4th disjoint basin is consistently weakly ignitable (sub-threshold at this operating point). So
  the wander SELECTS among **3 co-equal balanced basins**, not all 4. Named residual: get the weakest basin to ignite
  (stronger/synchronous encode — the Kopsick within-ensemble W_HIGH lever — or a per-basin adaptive coincidence
  threshold, or larger n_ca3 so 4 disjoint 240-cell basins do not over-subscribe the shared inhibition).
- **SELECTION-breadth vs STEERING-strength trade-off (characterized, not hidden).** The curiosity recurrent-gain
  concentrates the wander toward novel basins. At the strong operating point used here (gain-scale 1.0) the bias is
  large (66% attributable) while the production wander still visits 3 distinct (top1 0.62). A GENTLE gain (gain-scale
  0.6, measured on seed 42) broadens the production wander but the bias becomes weak (13% attributable, novel-share
  contrast ~0.09) — and a STRONGER gain over-concentrates (at rest 5000, gain 1.0, 2/6 seeds visited only 2 distinct).
  There is a genuine window; gain-scale 1.0 / rest 4000 is the reported operating point where BOTH hold (production
  visits ≥2, mean 3.0, AND strong identity-controlled bias). The **companion process** that would DECOUPLE them (per
  the "what runs alongside that we replaced with a constant?" lens) is short-term ADAPTATION/depression on the
  reactivated assembly — biology forces transitions so the wander moves on even under a strong salience bias; not
  built here (a named next rung, below).

## What is SUBSTRATE vs HOST (the honesty boundary is a deliverable)

- **SPIKING (load-bearing):** the reactivation + the SELECTION itself (CA3 dendritic-plateau attractor competition
  under non-specific noise decides WHICH basin ignites each event — there is **0 host content-draw / no
  `random.choice` over concepts** in the wander loop; only uniform Poisson to random CA3-exc cells), the silence
  between events, and the steering VALUE (the curiosity ASK-pool want is read off `cp_firing_states`).
- **HOST (declared, rides existing burn-downs):** (i) the per-concept NOVELTY levels are the ENVIRONMENT; (ii) the
  DISJOINT partition is a wiring choice (pattern separation), NOT a per-event content-draw; (iii) the PROJECTION of
  the spiking want onto the CA3 engram as a recurrent-gain factor is a host-parameterised neuromodulatory projection
  scaling (the one-brain-merge rung below).

## Named next rungs (no defer — the capability continues)

1. **Seed → utterance routing** (the mission's named final rung): route the surfaced thought vector into the
   composer/mouth so a spontaneous, curiosity-selected thought becomes a self-initiated question or remark — closes the
   loop to the DMN's conversational role.
2. **Get all n_mem basins to ignite** (4/4, then 6–8): stronger synchronous within-ensemble encode (Kopsick W_HIGH) or
   a per-basin adaptive coincidence threshold / larger n_ca3, so the store is fully balanced at higher n_mem.
3. **Adaptation-driven transitions** (the companion process): short-term depression/fatigue on the reactivated
   assembly, so a strong curiosity bias steers WHICH surfaces more WITHOUT narrowing how many are visited (decouples
   the selection-breadth vs steering-strength trade-off).
4. **One-brain merge:** release the `curiosity` neuromodulator directly onto the CA3 store on ONE bridge, so the gain
   is set BY the spiking modulator instead of a host scalar (the co-resident-merge rung the affect/surprise/episodic
   organs each carry).

**Status: runner-level de-risk GO (NOT wired to production / NOT integrated).** Functional self-initiation SELECTION
correlate only; no claim of phenomenal experience. Runner: `research/runners/_self_initiation_multibasin_derisk.py`.
NO `sim/` edit; reuse-by-import of the gap#5 RANK-1 reactivation substrate + the production curiosity organ + the
self-initiation steering machinery.
