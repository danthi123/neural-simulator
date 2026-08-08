---
type: finding
status: contributing
date: 2026-08-08
mechanism: deep-credit-on-spikes
lane: developmental-teacher-loop
artifacts:
  - research/findings/raw/teacher_loop_corrective_acquire_s42.json
  - research/findings/raw/teacher_loop_corrective_acquire_6seed.json
  - research/runners/_teacher_loop_corrective_acquire_derisk.py
---

# Developmental teacher-loop, first de-risk: the brain LEARNS ONE new fact by CORRECTION (its OWN synaptic weights move, no host store-write) and RE-USES it through the unchanged live-loop read — but the ACQUIRED read-path's learned confidence gate LEAKS (no structural moat), the boundary the dev engine must close (6-seed, single fact)

<!--derived-->

## What was built

A minimal first de-risk of the developmental teacher-loop
(`research/runners/_teacher_loop_corrective_acquire_derisk.py`). A host TEACHER — legitimate as the SOCIAL
ENVIRONMENT, same status as the world/sensory-render — teaches the brain ONE new fact it does NOT already know,
`dax eats grass`, by CONTINGENT correction. The brain ACQUIRES it via its OWN plasticity: the a1-GO transport-free
e-prop rule on the production Izhikevich bridge (`OnBridgeEpropNet`, reuse-by-import), whose FF readout weights
integrate the corrective error `softmax(logits) - onehot(grass)`. The acquired fact is then RE-USED at a later
turn through the UNCHANGED composer read `query_patient('dax','eats')`, routed via an additive default-off shim.

New referent `dax` = a noisy perceptual prototype (`ReferentEnv`, the a1 small perceptual category). The WORDS all
pre-exist as codes (`grass` is a patient code; `eats`/`chases` actions); only the FACT (this cue → this patient) is
new — absent from the composer kb and from any plastic map. Input = percept(referent) ⊕ action one-hot; K=6 patient
classes.

## The dividing line, made mechanical (brain-based-only)

The teacher PRESENTS a corrective target (the same cue the brain is responding to, paired with `grass` — a
Kuhl-style contingent recast entering as a corrective third factor on the co-active cue→answer eligibility) LIKE a
sensory input. The brain ACQUIRES it by moving its OWN synaptic weights. **NO `composer.store()` is called for the
taught fact**: `kb_len` is asserted UNCHANGED (2→2) while the e-prop readout pathway grows (readout-norm +18.4).
The error VANISHES at match (`softmax→onehot`), so it can never become the clamp-as-crutch the 2026-06-08
teacher-correction finding warns about (clamp-crutch regression: mean |δ| falls from initial to final).

## Result — 6/6 seeds (42..47), single taught fact

| Tooth | Criterion | 42 | 43 | 44 | 45 | 46 | 47 |
|---|---|---|---|---|---|---|---|
| T1 before/after | before abstain(None), after `grass` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T2 weights moved / no store | readout +Δ>1e-3 AND kb unchanged | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T3 specificity (learned moat) | untaught cues 0 false-accept | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| T4 lesion learning-pathway | learning-gate off → not acquired | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T5 lesion contingency | main > non-contingent + 0.15 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| T6 credit-route (shuffle-DFA) | depth lesion — INVALID at 1 hidden layer | – | – | – | – | – | – |

<!--derived-->

Held-out FRESH-draw accuracy (chance ≈ 0.17): **main 1.00 / 1.00 / 1.00 / 1.00 / 1.00 / 1.00**; non-contingent
0.03 / 0.00 / 0.00 / 0.00 / 0.17 / 0.00; shuffle-DFA 1.00 across (depth-1: the exact readout delta-rule carries the
task, so the DFA hidden-credit is not load-bearing — a1 established this; T6 belongs with the depth-2
semantic-inheritance task + frozen-reservoir control, the next de-risk). Attribution (`tools.lab`): 97.5% of the
main-vs-non-contingent effect is the teacher CONTINGENCY.

USE-in-loop: `query_patient('dax','eats')` returns `None` before teaching and `grass` after, rendering the later
turn `"dax eats grass"`. With the shim flag OFF the query is byte-identical to the raw composer over the whole cue
battery (6/6 seeds).

## The honest negative — the mapped boundary (T3)

**The acquisition is genuine and the concept is re-used, but the learned read-path's abstain/answer gate LEAKS.**
After teaching only `dax→grass`, the untaught cues `dax chases ?` (untrained action) and `wug eats ?` (2nd
untaught referent) BOTH read `grass` — at softmax confidence **~0.9997 and ~0.99998**. So **no confidence
threshold closes the leak**; a single-positive-class softmax read-path with NO structural moat saturates the taught
answer everywhere. This is the seam the design declared: for an ACQUIRED fact the gate is the e-prop readout
confidence, NOT the composer's structural kb-membership moat (a genuinely-learned fact has no kb block by
construction — that would be a store-write). The boundary is the load-bearing deliverable: it maps precisely what
the developmental engine must build next.

## What surpasses it (next), not a stop

The leak is NOT threshold-fixable → it needs CONTRASTIVE structure. Next de-risk: teach `dax→grass` AGAINST a
background of other referent→patient percept mappings the brain already knows (so the readout is not a constant
`grass` bias), and/or a learned source-monitor/familiarity gate (the `_phaseB_harden_320_learned_moat` arc), so the
learned confidence gate earns specificity the way the structural moat has it. Also declared next: brand-new lexeme
CODE allocation (the harder dendritic/allocation frontier), and driving the FULL stageA `run_multi_turn_loop`
(transformer mouth + GNW) rather than the direct `query_patient` re-use shown here.

## Reproduce

Single-seed SMOKE:

```
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_corrective_acquire_derisk --seeds 42 \
  --out research/findings/raw/teacher_loop_corrective_acquire_s42.json
```

6-SEED (the claims above; core teeth 6/6, T3 leak 6/6):

```
PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_corrective_acquire_derisk --seeds 42 43 44 45 46 47 \
  --out research/findings/raw/teacher_loop_corrective_acquire_6seed.json
```

Discipline: reuse-by-import (`OnBridgeEpropNet` + `_train_eprop` + `RFPhasorComposer`), NO `sim/` edit,
SIM_BACKEND=numpy, cfg.seed-controlled substrate, additive/default-off shim (byte-identical when off).
