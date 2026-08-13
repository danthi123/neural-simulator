---
type: finding
status: boundary
lane: A (affect / emotion keystone)
date: 2026-08-13
mechanism: composed-selforganized-affect-opponent
---

# Composed affect: the spiking opponent weights DERIVE FROM the self-organized valence map — the Warriner-seeded ridge-fit RETIRES for held-out valence, but the graded-magnitude bars are an HONEST BOUNDARY (6-seed)

**Runner:** `research/runners/_affect_composed_selforganized_opponent_derisk.py`
**Raw:** `research/findings/raw/_affect_composed_selforganized_opponent_6seed.json` (+ provenance sidecar), smoke `..._opponent.json`.
**Discipline:** `SIM_BACKEND=numpy` CPU lane, reuse-by-import, **NO `sim/` edit**. 6 seeds 42/43/44/100/101/102.

## The composition this closes (and where it holds vs stops)

Two affect de-risks landed on `main` with a clean seam. The **emergence** lane
(`_affect_evaluative_conditioning_derisk.py`, DR-2b GO) grows a concept→valence map from an evaluative-conditioning
stream by a LOCAL three-factor (DA-gated Hebbian) rule over the self-organized PPMI code, anchored by ~10 innate
primary reinforcers (a genome-cheap ±1 sign) — a LEARNED valence map, no Warriner lexicon (held-out r=+0.55). The
**affect-deepen** lane (`_affect_appraisal_emotion_reappraisal_derisk.py`, GO) is a spiking opponent population
(`appr_vplus`/`appr_vminus`) that reads valence off the substrate + drives discrete emotions + reappraisal. Its
declared #1 residual: the opponent feedforward weights are **ridge-fit in numpy AND SEEDED from Warriner norms** —
"the seed supervision is NOT retired."

This runner **derives the affect opponent V+/V− projection FROM the emergence map** — a three-factor Hebbian
outer-product over the learned code, `w = Σ_{c∈train} code_read_c · s_c`, rectified into the Namburi-Tye split
(`W+ = g·max(w,0)`, `W− = g·max(−w,0)`), injected as the SAME `code_in`→opponent FF the affect-deepen bridge uses —
**replacing the Warriner-seeded ridge-fit entirely**. The opponent READ stays spiking (`cp_firing_states`), unchanged.
Then the affect-deepen quality bars are re-run on this composed, **Warriner-free** circuit.

## Result (6-seed) — a QUALIFIED closure: the SIGN retirement HOLDS; the graded-MAGNITUDE bars do not

_Per-seed values are rounded from the cited 6seed JSON; seed-aggregates are its `means`._
<!--derived-->

| seed | held-out spiking r | ridge-Warriner baseline | \|d\|~valence-strength | permute-code perm-p | rung-b emotion acc | corr(s_c,Warriner) |
|---|---|---|---|---|---|---|
| 42 | +0.496 | +0.659 | +0.196 | 0.030 | 0.75 | +0.503 |
| 43 | +0.601 | +0.863 | −0.023 | 0.005 | 1.00 | +0.437 |
| 44 | +0.662 | +0.748 | +0.258 | 0.010 | 0.75 | +0.567 |
| 100 | +0.675 | +0.722 | +0.291 | 0.005 | 1.00 | +0.557 |
| 101 | +0.363 | +0.692 | +0.108 | 0.005 | 1.00 | +0.456 |
| 102 | +0.251 | +0.684 | −0.222 | 0.194 | 0.00 | +0.497 |

**The runner's own verdict is BOUNDARY** (5 of 11 pre-registered checks fail). What HOLDS and what STOPS:

**HOLDS — the Warriner SEED is retired for held-out valence generalization (the central deliverable):**
- **C-A1 held-out spiking valence r** — held-out concepts (their OWN reinforcement WITHHELD from the map) appraise to
  a SPIKING opponent differential correlating **r=+0.508** with true signed valence (every seed ≥ +0.251); the
  retired ridge-to-Warriner baseline reads **+0.728** (residual gap **+0.220**). PASS (mean ≥ 0.45; every seed ≥ 0.25).
- **C-A3 no-conditioning lesion** — remove the conditioning stream (`s_c := 0`) and the composed weights collapse →
  read **+0.000** (all seeds). The weights come from EXPERIENCE, not Warriner. This control replaces the ridge's
  Warriner seed and is the load-bearing "self-organized" proof.
- **C-A5 unpaired-US** permutation control beaten in **6/6** (perm-p<0.05): the CS↔US contingency is load-bearing.
- **C-B2 / C-B4 reappraisal** — the vmPFC→amygdala gate down-regulates `appr_vminus` by **83%** (reap-lesioned
  **+3%** ≈ 0). PASS.
- **corr(acquired s_c, Warriner)=+0.503** — the innate signal is honest; valence genuinely propagates from ~10 signs.
- **Warriner-free (asserted in code):** `selforg_opponent_weights` takes no Warriner argument; corrupting `s_true`
  leaves the weights byte-identical; the C-A3 collapse gives the assertion teeth. Warriner is EVAL-only ground-truth.

**BOUNDARY — the graded-MAGNITUDE-dependent bars underperform the magnitude-supervised ridge:**
- **C-A2 salience-strength** — `|differential|` tracks valence STRENGTH at only **r=+0.102** (vs the ridge's +0.27);
  negative on seeds 43/102. (The input-lesion HALF of C-A2 is rock-solid: `|d|` collapses 0.025→0.000.)
- **C-A4 permute-code** beaten in **5/6** (seed 102's weak weight, perm-p=0.194).
- **C-B1 / C-B3 / C-B5 discrete emotion** — mean discrimination **0.75** but seed 102 collapses to 0.00 (all four
  conditions → one pool), so not all-distinct; the WTA-lesion-collapse and mismatch-separation bars fail on the
  weak-valence seeds. The strong 5/6 select their intended emotion at 0.75–1.00.

## Root cause (diagnosed, not asserted) + three levers TESTED before naming the boundary

The failures concentrate on (i) the graded MAGNITUDE of valence and (ii) one genome-primary draw (seed 102). The
Rescorla-Wagner asymptote `s_c = (n_pos−n_neg)/(n_pos+n_neg)` **saturates** (Rescorla & Wagner, 1972): it encodes
valence SIGN robustly but graded STRENGTH weakly, so the composed weight carries sign (r≈0.5) far better than
magnitude (`|d|`~strength ≈ 0.1). The magnitude-supervised ridge inherits graded Warriner norms, hence its +0.27.
Per the wall-reframe ("what did we replace with a constant?"), the missing companion is a **graded** reinforcement
strength; the RW ratio is the constant we substituted. Three cheap levers were run against the residual:

1. **Graded conditioning signal** (log-odds `log((n+ +1)/(n− +1))`, evidence-confidence-weighted, net/√N) — **no help**
   (`|d|`~strength stays ≈ 0.08–0.10; seed 102 stays negative). Ruled out.
2. **Richer self-organized code** (n_hub 64→192) — **no help** (held-r ≈ 0.55, permute-code stuck 5/6, seed 102 weak).
3. **Innate-primary count** (the `--ablation`, in the artifact) — the residual is **SOFT**: 12–16 innate signs recover
   permute-code **6/6** and lift salience to ≈ 0.20, but 20 signs drop again (non-monotonic) — a
   fidelity / genome-draw-variance signature, NOT a substrate wall, and NOT a clean "just add primaries" fix.

| innate signs | held-r mean | held-r min | salience-r mean | permute-code |
|---|---|---|---|---|
| 8 | +0.49 | +0.25 | +0.08 | 4/6 |
| 10 (default) | +0.54 | +0.35 | +0.10 | 5/6 |
| 12 | +0.55 | +0.44 | +0.17 | 6/6 |
| 16 | +0.56 | +0.50 | +0.20 | 6/6 |
| 20 | +0.51 | +0.30 | +0.06 | 5/6 |

**We do NOT cherry-pick 16 to force a GO** (the non-monotonicity proves it is a lucky set, not a robust mechanism),
and we do NOT relabel the graded-magnitude gap "acceptable." The gap IS the deliverable.

## The next mechanism (the boundary's surpass, not a stop)

The residual is **graded-strength fidelity of the self-organized valence signal**, not the retirement itself, and it
is a SINGLE-LAYER associative write (a concept-code → innate-opponent map) — NOT hidden-layer credit assignment, so
the deep-credit-on-spikes negatives do not bear on it (`research/findings/2026-07-22-gap4-real-issue-NOT-dendrites.md`).
The faithful surpass is a **graded reinforcement-STRENGTH third factor**: drive the DA/US gate with the MAGNITUDE of
the reinforcement (a dopamine-ramp amplitude that scales with reward intensity, Bayer & Glimcher, 2005) rather than a
saturating sign, so `s_c` becomes a graded associative strength instead of the Rescorla-Wagner ratio; the opponent
valence-coding channel stays the innate V+/V− substrate (Namburi/Tye et al., 2015). A larger innate reinforcer set
(~12–16 genome-cheap signs) is a partial, buildable stopgap. Either keeps the appraisal Warriner-free while
recovering the magnitude the ridge got from human ratings.

## Honest residuals (brutally)

<!--derived-->
<!-- per-seed numbers restated below are rounded from the cited 6seed JSON per_seed[]. -->

1. **The graded-magnitude bars are unmet at the default (10 signs).** The SIGN retirement + held-out generalization
   HOLD (bar 1); the salience-strength (bar 2) and full discrete-emotion discrimination (bar 3) do NOT.
2. **Seed 102** (one 10-primary draw) drives most failures: r=+0.251, `|d|`~strength=−0.222, rung-b acc 0.00. This is
   genuine robustness-to-genome-choice variance, softened (not removed) by more signs.
3. **~10 innate primary SIGNS remain host-supplied** — the biologically-faithful floor (valence IS innately anchored
   by primary reinforcers), a 140→~10 compression, not a removal.
4. **Rate-level numpy Hebbian map** (the codes are the spiking-validated stream cortex; the fully-spiking graded
   three-factor write is the named rung above). **Standalone de-risk bridge** — `build_one_brain` fold-in pending.
5. **Operating-point gain calibration** — the composed weight is normalized to a fixed L2 norm (`W_L2_REF=1.7`) so
   the affect-deepen rung-b operating point transfers. A single global scalar carries NO per-concept valence and is
   correlation-invariant (a gain calibration, not a Warriner seed) — declared, not hidden.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._affect_composed_selforganized_opponent_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._affect_composed_selforganized_opponent_derisk \
    --seeds 42 43 44 100 101 102 --ablation \
    --out research/findings/raw/_affect_composed_selforganized_opponent_6seed.json
```

## Sources

- Rescorla & Wagner (1972) — the associative-strength ASYMPTOTE; the saturation is exactly why `s_c` encodes sign > strength.
- Namburi, Tye et al. (2015, Nature) — opposing valence-coding BLA populations (the innate V+/V− opponent channel).
- Bayer & Glimcher (2005, Neuron) — dopamine neurons encode a quantitative (graded-magnitude) reward-prediction error: the proposed graded-strength third factor.
- DR-2b: `research/findings/2026-08-13-affect-appraisal-origin-self-organizes-from-reinforcement-6seed-GO.md` — the self-organized valence map this composes FROM.
- Affect-deepen: `research/findings/2026-08-13-spiking-appraisal-discrete-emotion-reappraisal-derisk.md` — the spiking opponent + bars, and the Warriner-seed residual this retires.
