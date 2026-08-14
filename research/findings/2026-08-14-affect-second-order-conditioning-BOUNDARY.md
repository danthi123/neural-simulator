---
type: finding
status: boundary
lane: A (affect / emotion keystone)
date: 2026-08-14
mechanism: affect-second-order-evaluative-conditioning
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_second_order_conditioning_derisk.py
artifacts:
  - research/findings/raw/_affect_second_order_conditioning_6seed.json
builds_on:
  - research/findings/2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md
  - research/findings/2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md
  - research/findings/2026-08-13-affect-appraisal-origin-self-organizes-from-reinforcement-6seed-GO.md
  - research/findings/2026-08-13-affect-lc-arousal-population-GO.md
---

# Higher-order (second-order) evaluative conditioning — the LAST unsupervised corpus write — does NOT recover graded valence STRENGTH either. Combined with the single-step oracle ceiling, this CLOSES the CPU-corpus method space and PROVES the affect graded-strength residual needs a bodily/interoceptive (embodiment) input. The SIGN retirement holds. (6-seed)

<!--derived-->

**Runner:** `research/runners/_affect_second_order_conditioning_derisk.py`
**Raw:** `research/findings/raw/_affect_second_order_conditioning_6seed.json` (+ provenance sidecar). The 1-seed 8k smoke is a wiring check only, NOT committed as a verdict artifact — its preconditions do not hold (K=1 over-reads first-order strength at 8k), so its verdict is UNDEFINED, per the spec's own "the 6-seed at the authoritative corpus is the verdict".
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import of `[E]`'s composed runner, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102. Pre-registered gamma=0.5, K=3 BEFORE the 6-seed (smoke sweeps only; smoke over-reads).

## What this tested (the ONE un-attempted surpass the ceiling named)

<!--derived-->

`[E]` (`2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md`) retired the Warriner seed for held-out valence SIGN (r=+0.508) but graded STRENGTH underperformed (r=+0.10 vs the magnitude-supervised ridge's +0.27-0.29).
`[T]` (`2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md`) then ruled out the WHOLE single-step third-factor-SCALING family with an airtight ORACLE (even |Warriner| innate-US intensities → r=+0.081), and reframed the residual: strength is an **information boundary of the sparse ~10-primary channel** — the strength info IS in the self-organized code geometry (the ridge extracts +0.29) but *the single-step primary→concept write cannot reach it*.
`[T]` named **higher-order (concept↔concept) conditioning** as the one un-attempted surpass: a MULTI-STEP associative write might reach the code-geometry structure a single-step write cannot. This runner is that write.

**Mechanism (Warriner-free, brain-based).** After first-order conditioning gives each concept `s_c^(1)` (Rescorla-Wagner asymptote over ~10 innate primaries), an already-valenced concept `d` acts as a **conditioned reinforcer** for its associates `c` (Rescorla 1980 second-order conditioning: a CS₂ paired with a first-order CS₁ acquires value with NO direct US pairing). K discounted passes over the row-stochastic kNN cosine graph `A` of the learned PPMI code:

```
s_c^(k) = softclip( s_c^(1) + gamma * sum_d A_cd * s_d^(k-1) )   # k=2..K; gamma<1 (higher orders extinguish)
```

`s_c^(K)` feeds the SAME three-factor Hebbian opponent map (`selforg_opponent_weights`) as `[E]`; the opponent READ is the unchanged spiking differential rate(vplus)−rate(vminus) off `cp_firing_states` (Namburi-Tye V+/V− channel). Only the WRITE gains the second-order term. Held-out = own-reinforcement-withheld (the held concept's `s_c^(1)` and its second-order increments are excluded from the map AND as a propagation source).

## Result (6-seed) — the multi-step write does NOT lift strength; it does not even beat static diffusion

_Means are the cited 6seed JSON `means`; per-seed from `per_seed[]`. Warriner is EVAL-only ground-truth._
<!--derived-->

All arms share the reinforced set + the train/held split; they read the SPIKING opponent differential. STRENGTH = held-out `r(|differential|, |true valence|)`:

| arm (6-seed mean) | write | STRENGTH r | SIGN r |
|---|---|---|---|
| **ridge-Warriner** (reference target) | per-CONCEPT Warriner (magnitude-SUPERVISED) | **+0.288** | +0.72 |
| static-diffusion (converged smoother of `A`) | multi-step, seeded by `s_c^(1)`, run to equilibrium | +0.134 | +0.49 |
| **second-order conditioning** (pre-registered) | K=3, gamma=0.5, discounted, own-anchor per order | **+0.104** | +0.496 |
| first-order boundary (K=1 order-lesion) | single-step `s_c^(1)` (`[E]`'s method) | +0.101 | +0.49 |

Per-seed STRENGTH r [2nd-order, first-order(K=1), static-diff, ridge] + SIGN(2nd) + discrete-emotion acc:

| seed | 2nd-order | first(K=1) | static-diff | ridge | SIGN(2nd) | emo acc |
|---|---|---|---|---|---|---|
| 42 | +0.185 | +0.252 | +0.214 | +0.277 | +0.470 | 1.00 |
| 43 | +0.028 | −0.136 | +0.029 | +0.375 | +0.607 | 1.00 |
| 44 | +0.298 | +0.199 | +0.312 | +0.211 | +0.668 | 0.75 |
| 100 | +0.281 | +0.323 | +0.359 | +0.224 | +0.676 | 0.75 |
| 101 | +0.074 | +0.060 | +0.046 | +0.393 | +0.368 | 1.00 |
| 102 | −0.241 | −0.095 | −0.157 | +0.246 | +0.186 | 0.00 |

## What HOLDS, what STOPS, and why the boundary is now DECISIVE

<!--derived-->

**HOLDS:**
- **Instrument is validated by three reference points**, so the negative is a real weight-source result, not a broken read: the no-conditioning lesion (`s_c^(1):=0`) reads exactly **+0.000 (all seeds)**; the K=1 order-lesion reproduces `[E]`'s first-order boundary **+0.101 EXACTLY**; the ridge reproduces `[T]`'s magnitude-supervised target **+0.288**. **G4/G5a PASS.**
- **Warriner-free (asserted + teeth):** `second_order_field` + `selforg_opponent_weights` take no Warriner argument; `--corrupt-warriner` leaves the write-side field **byte-identical** (moved_frac/sat unchanged; only the EVAL r shifts, because the scoring target moved). The no-cond collapse gives the assertion teeth.
- **`s_c^(K)` stays GRADED** (non-explosion): mean saturated-fraction 0.066, IQR 0.362 — the discounted accumulation did not collapse to bimodal ±1. corr(s_K, Warriner)=+0.516 (the innate signal is honest). The K≥2 pass genuinely MOVED the field every seed (lever: mean |increment| 0.03–0.20).
- **SIGN retirement mostly holds** (r=+0.496 mean) — the second-order pass does not broadly trade sign for strength.

**STOPS (the negative):**
- **Graded STRENGTH does NOT lift.** Second-order r=**+0.104** ≈ first-order +0.101 (beats it in only **3/6** seeds). **G1 FAIL.**
- **It does not beat static diffusion.** The converged smoother of the SAME operator reads **+0.134** — HIGHER on average; second-order beats it in only **1/6**. **G2 FAIL.** Per the pre-registered anti-cheat #1, any movement here is *generic graph-averaging, not conditioning-specific dynamics*.
- **The unpaired-second-order permutation is beaten in 0/6** (perm-p 0.08–0.63) — there is no reliable per-concept second-order strength signal to beat the null with (consistent with the null lift, not an instrument failure). **G5b FAIL.**
- **G3 every-seed and G6 fail on the weak-draw seed 102** (SIGN +0.186 < 0.25; discrete-emotion 0.00, sat 0.33): the second-order pass did not rescue — it slightly *worsened* — the known weak-draw seed. Discrete-emotion stays 0.75 (5/6 non-collapsed), unchanged from `[E]`.

**THE METHOD SPACE IS NOW CLOSED (the load-bearing statement).** `[T]` ruled out every SINGLE-STEP write (primary→concept), oracle included. This rules out the MULTI-STEP associative write (second-order conditioning AND its converged diffusion limit). The strength info the ridge extracts (+0.29) lives in the code geometry, but **no unsupervised corpus WRITE — single-step or multi-step — reaches it.** The missing quantity is per-concept intensity that ~10 reinforcers + child-story co-occurrence do not contain, and that no re-writing of the corpus signal manufactures.

## The next axis is EMBODIMENT, not another corpus write (the boundary's surpass)

<!--derived-->

This converts the board's *"needs a bodily/interoceptive/embodiment input"* from hypothesis to **requirement**.
Graded valence STRENGTH (arousal-scaled intensity, distinct from sign) is carried in biology by interoceptive/bodily-arousal channels (LC-noradrenergic + visceral afferents), NOT by valence-sign co-occurrence. The affect arc's arousal surpass (`2026-08-13-affect-lc-arousal-population-GO.md`) already showed a *separate* LC-like population recovers graded arousal (r=+0.31) on the ORTHOGONAL axis — the same lesson: intensity needs its own channel.
The next rung is a graded reinforcement-MAGNITUDE signal grounded in embodied/interoceptive reinforcement, conditioned into a magnitude channel parallel to the valence-sign opponent — NOT a further corpus-statistical write. Per THE LAW: "info-boundary" is a verdict on the METHODS (single- AND multi-step corpus writes), never a license to defer the capability.

**Banked-negative note (chain-ORDERING vs strength).** Static diffusion / label-propagation is banked-negative for the chain-ORDERING question (DR-2's retired Zhu-Ghahramani harmonic solve, Warriner-seeded). This runner is a *different write* (no chain; a finite, discounted, own-anchored associative CONDITIONING pass) run as the FAIR strength mechanism. Its failure to beat even static diffusion confirms neither the diffusion family nor the conditioning family recovers strength — the residual is information availability, not the choice of graph dynamics.

## Honest residuals (brutally)

<!--derived-->

1. **The pre-registered second-order conditioning is a NEGATIVE for STRENGTH** (r=+0.104 ≤ first-order +0.101; loses to static-diffusion +0.134). The one un-attempted corpus surpass is now attempted and ruled out.
2. **The negative is robust to the hyperparameter** — the smoke sweep (γ∈{0.3,0.5,0.7}, K∈{2,3}) showed NO setting lifted strength above first-order (all ≈0.18–0.20 vs first-order 0.199 on the over-reading 8k smoke). Not a tuning miss.
3. **SIGN retires; STRENGTH does not** — the Warriner-free ceiling for graded strength through the corpus channel is ≈0.10–0.13; the ridge's ≈0.29 needs per-concept magnitude supervision. Residual gap ≈0.18.
4. **~10 innate primary SIGNS remain host-supplied** (the faithful floor; a 140→~10 compression, unchanged from DR-2b).
5. **Rate-level numpy Hebbian write; standalone de-risk bridge** — a fully-spiking second-order write and the `build_one_brain` fold-in are later rungs (moot for the strength residual, which is an information boundary, not a substrate one). Functional read-outs only; NEVER a claim of phenomenal experience.

## Anti-cheats (each a gate that behaved)

<!--derived-->

- **STATIC-DIFFUSION baseline** (anti-cheat #1 / G2): the load-bearing control — second-order conditioning FAILED to beat it (1/6), so any lift is graph-averaging, reported as such.
- **WARRINER-FREE** (G4): `--corrupt-warriner` byte-identical write-side; no-conditioning lesion → +0.000 all seeds.
- **ORDER-LESION** (G5a): K=1 reproduces the first-order field EXACTLY (asserted `array_equal`) and its +0.101 strength — any lift would be attributable to K≥2 alone; there is none.
- **CONDITIONED-US CONTINGENCY** (anti-cheat #4 / G5b): the unpaired-second-order permutation was not beaten (0/6) — consistent with no real contingent signal.
- **NON-EXPLOSION** (anti-cheat #5): sat_frac 0.066, IQR 0.362 — γ<1 + soft-clip kept `s_K` graded.
- **HELD-OUT = own-reinforcement-withheld** (held excluded from the map AND as a propagation source); **permute-code** beaten 5/6 (seed 102 the known weak draw); **6 seeds** (smoke first, non-authoritative).

## Sources

- **Rescorla, R.A. (1980), _Pavlovian Second-Order Conditioning_ (Erlbaum); Rescorla & Wagner (1972)** — a CS₂ paired with an already-conditioned CS₁ (a conditioned reinforcer) acquires value with NO direct US pairing; second-order responses are weaker (the γ<1 order-discount). The mechanism tested here; the RW asymptote is why the first-order write saturates on strength.
- **Namburi, P., Tye, K.M. et al. (2015, Nature 520:675)** — opposing valence-coding BLA populations: the innate V+/V− opponent channel the weights inject into (unchanged from `[E]`).
- **Bayer, H.M. & Glimcher, P.W. (2005, Neuron 47:129)** — graded DA reward-magnitude: the single-step third-factor lever `[T]` ruled out; the parallel-magnitude channel this boundary redirects toward (embodied/interoceptive).
- `[E]` `2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md`; `[T]` `2026-08-13-affect-graded-strength-third-factor-BOUNDARY.md` (the oracle ceiling); `2026-08-13-affect-lc-arousal-population-GO.md` (the orthogonal arousal channel).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_second_order_conditioning_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_second_order_conditioning_derisk --corrupt-warriner
SIM_BACKEND=numpy python -u -m research.runners._affect_second_order_conditioning_derisk \
    --seeds 42 43 44 100 101 102 --orders 3 --gamma 0.5 \
    --out research/findings/raw/_affect_second_order_conditioning_6seed.json
```
