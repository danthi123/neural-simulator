---
type: finding
status: go
date: 2026-08-08
mechanism: teacher-loop-contrastive-teaching-plus-learned-familiarity-source-monitor-gate
lane: E-language
runner: research/runners/_teacher_loop_contrastive_familiarity_moat_derisk.py
artifacts:
  - research/findings/raw/teacher_loop_contrastive_familiarity_moat_6seed.json
  - research/findings/raw/teacher_loop_contrastive_familiarity_moat_6seed.json.prov.json
---

# Teacher-loop: the learned-moat leak is CLOSED 6/6 — contrastive teaching + a learned familiarity/source-monitor gate

## The leak this closes

The first teacher-loop de-risk (`_teacher_loop_corrective_acquire_derisk.py`, main `7f48cbad`) taught the brain a new
fact — *dax eats grass* — by its OWN e-prop plasticity (verified: readout weights moved, no `composer.store()`),
then re-used it at a later turn. But the abstain/answer gate for the ACQUIRED fact was the readout CONFIDENCE
threshold, and it **LEAKED 6/6**: because the fact was taught with a SINGLE class of targets, the readout saturated
to a constant grass-bias, so UNTAUGHT cues (*dax chases ?*, *wug eats ?*) also read `grass` at confidence 1.0 (2/2
false-accepts, every seed). A genuinely-learned fact has no structural kb-membership block by construction, so it
needs a LEARNED specificity moat. This de-risk builds the two levers the design named — and both are load-bearing.

## The two levers (brain-based)

**Lever 1 — contrastive teaching.** Teach *dax→grass* INTERLEAVED with a background of other referent→patient
mappings the brain also holds (*dog→bone*, *cat→fish*), all through the SAME e-prop readout (the a1-GO transport-free
e-prop net — the brain's own plasticity, the sole learner). The readout can no longer minimise loss by emitting a
constant; it must DISCRIMINATE on the percept. Result: **3/3 taught facts read their own patient, no cross-talk,
every seed.**

**Lever 2 — a learned familiarity / source-monitor gate.** The Bogacz-Brown anti-Hebbian familiarity gate (catalog
D.04 perirhinal repetition suppression; reuse-by-import `RealAntiHebbianFamiliarity` from the phaseB learned-moat
arc) imprints the cues the teacher ACTUALLY taught, as a SINGLE conjunctive VSA binding of the referent-percept code
with the action code — so ANY mismatch (untaught referent OR untaught action) makes the whole cue novel. At query it
reads novelty `N(x)=||x||²−xᵀWx`: familiar → the readout answer is trusted; novel → ABSTAIN, whatever the readout
confidence.

**Why both.** Contrastive teaching alone makes the readout discriminative, but a discriminative classifier still
emits its best guess for any cue — the confidence-only gate (the old mechanism) STILL false-accepts untaught cues
(measured: gate-OFF false-accepts = 2/2 on every seed). The familiarity gate alone would gate a degenerate constant
readout. Together: contrastive earns a genuine multi-fact map; the gate earns the abstain.

## Result — 6/6 GO (earned `tools.verdict.Verdict`, every precondition measured and held)

Rounded from `research/findings/raw/teacher_loop_contrastive_familiarity_moat_6seed.json`:

<!--derived-->

| seed | dax before→after | untaught abstain? | FA gate-ON | FA conf-only-OFF | novelty margin | lesion margin | facts no-cross-talk | main mean | mispaired mean |
|-----:|:-----------------|:------------------|:----------:|:----------------:|:--------------:|:-------------:|:-------------------:|:---------:|:--------------:|
| 42 | None→grass | dax+chases & wug+eats both None | 0 | 2 | +0.780 | -0.000 | 3/3 | 0.98 | 0.02 |
| 43 | None→grass | both None | 0 | 2 | +0.847 | -0.000 | 3/3 | 1.00 | 0.01 |
| 44 | None→grass | both None | 0 | 2 | +0.802 | +0.000 | 3/3 | 1.00 | 0.01 |
| 45 | None→grass | both None | 0 | 2 | +0.843 | -0.000 | 3/3 | 0.99 | 0.00 |
| 46 | None→grass | both None | 0 | 2 | +0.789 | +0.000 | 3/3 | 0.93 | 0.11 |
| 47 | None→grass | both None | 0 | 2 | +0.788 | -0.000 | 3/3 | 1.00 | 0.00 |

**The T3 specificity that leaked 6/6 is closed 6/6.** After teaching, `dax eats` answers `grass`, `dog eats` answers
`bone`, `cat eats` answers `fish`, while `dax chases ?` (untaught action) and `wug eats ?` (untaught referent) both
ABSTAIN — 0 false-accepts, every seed.

## Teeth (all pass 6/6; comparators flip in their failing direction every seed)

- **T1** before/after: gate un-imprinted → abstain (None); after teaching → `grass`.
- **T2** the readout WEIGHTS moved (e-prop grew the leaky readout) AND the composer kb length is UNCHANGED (no store-write).
- **CT1** discrimination: dax held-out > 0.6; the readout emits ≥2 distinct classes with ≥2 taught referents reading their own patient (not a constant grass-bias).
- **CT2** contrast-flip: the single-class control learns only its 1 fact (`sc_n_correct=1`, backgrounds fall to grass or the attractor) while contrastive discriminates 3 — contrastive is load-bearing.
- **FG1** margin: novelty(untaught) ≫ novelty(taught), +0.78 to +0.85 (a-priori-separable; `NOV_GATE=0.5` is the perirhinal unit-norm midpoint, NOT tuned on the probes).
- **FG2** specificity (the headline): 0 false-accepts, gate ON.
- **FG3** load-bearing: (a) the conf-only gate (gate OFF) false-accepts 2/2 on EVERY seed; (b) lesioning the gate's learned projector collapses the novelty margin to ~0.00 — the abstain rides the LEARNED weights.
- **T4** lesion-learning: freeze e-prop (lr=0) during the identical teaching → readout does not move → dax held-out ≤ chance (LEARNED, not wired).
- **T5** lesion-pairing: a MISPAIRED teacher (a consistent WRONG referent→patient assignment) → mean true-target held-out ~0.00-0.11 vs main ~1.0 (the answer is the teacher's SPECIFIC pairing). A random-label shuffle is NOT a clean control here — separable percept clusters let a shuffled-label net form an arbitrary map that coincidentally aligns on some seeds; a consistent wrong pairing is definitely wrong on the true targets.
- **off-flag byte-identical** every seed (the shim adds nothing when disabled).

## Honest seams (declared, not sold)

- **The readout answer** is a host argmax/softmax over the K patient words — the same declared render as the first de-risk (the neural-motor read-out is the standing next target).
- **The conjunctive cue** uses fixed random VSA codes (a percept-projection `P` + per-action codes) — the composer-as-idealization host seam for what a learned cortex would encode. The LEARNED, load-bearing part is the anti-Hebbian projector `W` (imprinted from the taught cues, lesioned in FG3 → margin → 0).
- **Percept separability boundary (mapped, then surpassed).** At the first de-risk's `d_p=12` the small e-prop readout collapsed the headline fact on 2/6 seeds (correlated all-positive `[0,1]` percepts are hard to separate 3-ways). Richer sensory dimensionality (`d_p=32`, `hidden=40`, `epochs=80`) plus a settled-percept read (integrate ~15 glances, not one noisy sample) resolved it 6/6 — a percept/capacity boundary, distinct from the moat-leak this finding targets.
- **Joint teaching.** All three facts are taught together. SEQUENTIAL/continual acquisition (learn *dax* AFTER *dog/cat* without forgetting) is the declared NEXT step, and is the true "learn many facts over time without cross-talk" test.

## Reproduce

```
# 6-seed GO (needs 6/6 at 42..47):
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_contrastive_familiarity_moat_derisk \
  --seeds 42 43 44 45 46 47 \
  --out research/findings/raw/teacher_loop_contrastive_familiarity_moat_6seed.json
# single-seed SMOKE:
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_contrastive_familiarity_moat_derisk --seeds 42 \
  --out research/findings/raw/teacher_loop_contrastive_familiarity_moat_s42.json
```

Discipline: reuse-by-import (OnBridgeEpropNet + `_train_eprop` + `_softmax`; `RealAntiHebbianFamiliarity` + `hrr_bind`;
`RFPhasorComposer`). NO `sim/` edit; additive/default-off shim; `SIM_BACKEND=numpy`; `cfg.seed` via the seed= arg the
a1 net passes to `CoreSimConfig.seed`. Elapsed 764 s for the 6-seed run (CPU).
