---
type: finding
status: mixed
date: 2026-08-19
lane: perception
mechanism: invariance-from-temporal-continuity
runner: research/runners/_vision_hmax_hierarchy_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/vision_hmax_hierarchy_6seed.json
---

# Position-invariant CONFIGURAL recognition (board #44): the HMAX S->C hierarchy CLEARS the wall the flat learned pool could not, but the invariance is carried by the innate COMPOSED-POOLING TOPOLOGY, not by template-learning (random == learned) — the flat-case NO-GO's "topology not learning" root cause, confirmed one layer up

**One-line verdict.** The predecessor de-risk
([`2026-08-19-vision-pooling-invariance-topology-not-learning-NOGO.md`](2026-08-19-vision-pooling-invariance-topology-not-learning-NOGO.md))
proved a SINGLE learned cross-position pool is a 6-seed NO-GO and named the root cause: a learned pool "can only
place weight where it SAW activity", so no flat layer can exceed V1's RF-overlap tolerance. It named HMAX
(Riesenhuber & Poggio 1999) as the surpass: accept innate LOCAL pooling as biology, STACK simple->complex layers so
GLOBAL invariance EMERGES from COMPOSED local shift-tolerance — no single layer weights unseen positions. Built and
tested on histogram-MATCHED CONFIGURAL objects (so a flat orientation-histogram pool is forced to chance): the
hierarchy DOES clear the wall — held-out-position decode **0.5972** vs V1-direct-held **0.3698** and the flat pool
**0.2674** (chance 0.25), with a ~0 invariance gap (same-position **0.4479** vs held-position **0.5**), position
pooled out, and pixel-scramble at chance (5/6 seeds clear the FULL simultaneous gate; the 6th marginally leaks
position). BUT the load-bearing quantity is the composed-pooling ARCHITECTURE, not learning: a RANDOM S2 projection
matches the trace-learned one (**0.5851 vs 0.5972**; template-learning load-bearing 0/6 pooled), and the trace rule
ties its temporal-shuffle control (**0.5972 vs 0.592**) — so the invariance is innate-topological, the flat-case
NO-GO's exact root cause confirmed at the S2 stage. No `sim/` edit; additive runner; `SIM_BACKEND=numpy`; 6 seeds.

`EXTERNAL-SEARCH-RAN:` HMAX = Riesenhuber & Poggio 1999, *Nature Neuroscience* 2:1019-1025 ("Hierarchical models of
object recognition in cortex" — alternating S template-tuning / C MAX-pooling layers; the MAX operation gives
shift-tolerance while preserving feature selectivity; invariance is COMPOSED up the hierarchy; S2 units tune to C1
feature CONJUNCTIONS). Learned S2 patches = Serre/Wolf/Poggio 2007. Trace rule = Foldiak 1991. Verified the mechanism
(alternating S/C, MAX-pool) against the source; the external search is recorded (lane `perception`). This finding
also updates the gap
[`research/biology/invariance-from-temporal-continuity.md`](../biology/invariance-from-temporal-continuity.md) flags —
its trace-rule GO was for CATEGORY membership and states "POSITION ... invariance ... were NOT tested"; POSITION is
now tested, and the honest result is that for position the trace-learning is NOT load-bearing (see below).

## Design — one shared front end, six arms, one GO gate, 6 seeds

**Task: histogram-MATCHED CONFIGURAL objects.** Each object = 3 oriented strokes at 3 fixed relative slots; identity
= the ARRANGEMENT (a permutation of orientations {0,60,120 deg} across the slots). Every class shares the SAME
orientation multiset, so the global orientation HISTOGRAM is identical across classes -> a flat "pool everything per
orientation" pool (the move the prior levers reached for) is FORCED to chance, and CONFIGURATION is the only signal.
4 classes (chance 0.25); the whole object translates as a rigid unit across 8 retinal positions; **interleaved
held-out** — train {0,2,4,6}, test {1,3,5,7} NEVER seen in training (anti-cheat 1).

**Front end (all arms; a FLAGGED innate developmental scaffold, per route-1's "accept innate local pooling as
biology"):** pixels -> deployed Gabor/V1 simple (`sim.visual_cortex.build_v1_simple_weights`, reused by import, NOT
edited) -> hypercolumn orientation-competition (z, lateral inhibition across orientation columns) + a firing-threshold
gate (keeps background silent so a MAX pool sees the object, not the noise floor) -> **C1** innate LOCAL retinotopic
MAX-pool per orientation (local shift-tolerance). Decode = nearest-cosine-centroid.

- **A · V1-DIRECT** — decode off the flattened C1 (position-specific; the floor the prior NO-GOs also read).
- **H · FLAT-POOL** — global orientation histogram (= REMOVE the S2 stage, pool C1 straight to global). This is
  BOTH the configuration-blind control AND the S2-stage lesion.
- **B · HMAX-IMPRINT** — S2 templates one-shot-Hebbian imprinted from C1 patches of train images (Serre/Poggio 2007).
- **T · HMAX-TRACE** (the PRIMARY arm) — S2 templates learned by trace-modulated competitive Hebbian on a
  moving-object continuity stream (Foldiak 1991); the fully-emergent biological arm. Control: temporal-SHUFFLE.
- **R · HMAX-RANDOM** — LESION of template-learning: identical hierarchy, RANDOM S2 templates (a random projection).
- **P1 · HMAX-p1** — ABLATION: S2 extent = 1 C1 cell (a single-stroke detector, no conjunction).

S2 is convolutional (one template applied at EVERY C1 position by innate retinotopic weight-sharing) -> C2 = global
MAX-pool per template -> position invariance. **The point that answers the prior NO-GO:** the LEARNED quantity (an S2
template) is NOT a pool over positions — it is a small local conjunction detector applied at every position by the
innate convolution, so it NEVER has to weight unseen positions; the retinotopic conv + global MAX span all positions
identically, held-out ones included.

## Result — 6 seeds (42/43/44/100/101/102), chance 0.25

Artifact: `research/findings/raw/lanes/perception/vision_hmax_hierarchy_6seed.json`. Held-out-position object decode
(mean over 6 seeds):

| arm | mean held decode |
|---|---:|
| A · V1-direct held (position-specific floor / RF-overlap ceiling) | **0.3698** |
| H · FLAT-POOL held (= no S2 stage; configuration-blind)           | **0.2674** |
| **T · HMAX-TRACE held (PRIMARY, biological)**                     | **0.5972** |
| R · HMAX-RANDOM held (template-learning lesion)                   | 0.5851 |
| B · HMAX-IMPRINT held (Serre/Poggio patches)                      | 0.3802 |
| P1 · HMAX-p1 held (single-cell ablation)                          | 0.2986 |
| — pixel-scramble (HMAX-trace)                                     | 0.2535 |

Headroom (stored differences): HMAX-trace over V1-direct-held **0.2274**; over flat-pool **0.3298**; over the random
projection **0.0121** (≈ 0 -> learning inert). Invariance: same-position CV **0.4479** vs held-position CV **0.5** ->
gap mean **0.0191** (held ≈ same-position, genuine invariance). Dissociation off the C2 held code: object
**0.507** vs position **0.3541** (chance-position 0.25); label-shuffle null **0.2327** (≈ chance). Verdict fractions:
**capability_go 0.8333** (5/6); **architecture_load_bearing 1.0** (6/6); **flat_pool_configuration_blind 1.0** (6/6);
**template_learning_load_bearing 0.1667** (1/6 seed by noise; pooled diff 0.0121); **trace_load_bearing 0.0** (0/6).

## The anti-cheats — what each returned

1. **Held-out positions (THE bar).** HMAX-trace held **0.5972** exceeds V1-direct held **0.3698** and the flat pool
   **0.2674**, with a ~0 same-vs-held gap. The hierarchy exceeds the RF-overlap ceiling the prior flat levers could
   not. GO on 5/6 seeds (seed 42: everything passes except a marginal position leak, **0.4375** vs the bar
   chance-position 0.25 + margin 0.15). <!--derived: 0.25+0.15 = the pooled-out threshold-->
2. **Learned part load-bearing.** Two lesions, opposite verdicts, and BOTH are the result: (a) lesion the S2
   CONJUNCTION stage -> flat pool -> **0.2674** (chance): the composed conjunction+pooling ARCHITECTURE is
   load-bearing (6/6). (b) lesion the template-LEARNING -> random projection -> **0.5851** ≈ trace **0.5972**:
   template-learning is NOT load-bearing (0/6 pooled), and the trace ties its temporal-shuffle control **0.592**. A
   random projection preserves the (separable) configural identity once the innate conjunction+pool architecture
   exists.
3. **Position pooled out.** Off the C2 code: object decodable **0.507** (>> chance 0.25) while position decode
   **0.3541** (≈ chance-position 0.25) — position discarded, identity retained (5/6; seed 42 leaks weakly at 0.4375).
4. **6 seeds + nulls.** pixel-scramble **0.2535** and label-shuffle **0.2327** both at chance; determinism verified
   (byte compare of a re-run). P1 ablation **0.2986** (≈ chance) shows the conjunction EXTENT (p>1) is required — a
   single-cell S2 collapses to the histogram.

## Root cause — quantified, and it UNIFIES with the flat-case NO-GO

The `attributable_to` decomposition (runner-computed, pooled means; stored differences): of HMAX's held invariance,
the S2 CONJUNCTION stage contributes **0.3298** (over the flat pool) and the composed hierarchy **0.2274** (over
V1-direct), while LEARNED templates contribute only **0.0121** (over a random projection). So:

- **The capability is real and the hierarchy is the mechanism:** composing innate LOCAL shift-tolerance (C1) with an
  innate GLOBAL pool (C2), through an intermediate CONJUNCTION stage (S2) that makes configural identity survive the
  global pool, gives position-invariant configural recognition where the single flat pool (the NO-GO) gave chance.
  No layer ever weights an unseen position — the retinotopic convolution handles all positions identically.
- **But the invariance is carried by the innate composed TOPOLOGY, not by learning.** This is the flat-case NO-GO's
  root cause ("the invariance-carrying quantity is the pooling TOPOLOGY, which is retinotopic/pre-wired, NOT learned")
  confirmed one layer up: once the innate conjunction+pool architecture exists, a random S2 projection preserves the
  separable configural identity, so there is nothing for template-learning to add. The prior finding predicted
  learning would only matter "when identity is a CONJUNCTION a random projection would NOT preserve" — here identity
  IS a configural conjunction, yet the 4 classes remain separable by a random projection (Johnson-Lindenstrauss:
  well-separated classes survive a random max-pooled projection), so template-learning stays inert.

This is consistent with the biology: complex-cell pooling RFs are developmentally pre-structured (retinotopic
OR/MAX-pooling — Hubel-Wiesel; the Riesenhuber-Poggio C layers are wired, not learned), and temporal-continuity
learning (Foldiak/SFA) is load-bearing for OTHER invariances (CATEGORY membership — the prior emerge50 trace-rule GO),
not for POSITION, which is innate-retinotopic.

## Honest residuals (NOT a wall — a characterized operating point)

- **capability_go is 5/6, not 6/6.** The 6th seed (42) clears every criterion except a marginal position leak
  (**0.4375** vs the pooled-out bar). The effect is real but MODERATE (held ~0.6 vs V1 ~0.37); operating-point sweeps
  trade one seed's marginal miss (position leak / V1-margin / gap) for another, the signature of a moderate margin,
  so the op-point was NOT tuned past this to force 6/6 (that is the "tuning optimises whatever the metric rewards"
  trap).
- **The result is a RATE de-risk of a spiking HMAX.** S = synaptic integration (template dot-product), C = a
  MAX-like complex-cell nonlinearity (the Riesenhuber-Poggio cortical MAX op), imprint = one-shot Hebbian tuning,
  trace = a slow eligibility variable + competitive Hebbian. A rate model is GENEROUS: if rate fails, spiking will
  not save it. The retinotopic weight-sharing and pooling windows are FLAGGED innate scaffolds (route-1's concession);
  the honest next step for full brain-based credit is the spiking S->C stack (not required to establish the capability
  or the topology-not-learning verdict, which are backend-independent).
- **What is NOT closed:** this is a de-risk, not production-integrated (the production `/api/brain-chat` ventral path
  is unchanged); and the imprint arm underperforms both trace and random (redundant single-sample patches) — a known
  HMAX-imprinting weakness, not load-bearing for any claim here.

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u -m \
  research.runners._vision_hmax_hierarchy_derisk --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/perception/vision_hmax_hierarchy_6seed.json
```

CAPABILITY GO gate (per seed, PRIMARY = HMAX-TRACE): held-object decode >= chance + 0.15; beats V1-direct-held AND
flat-pool by 0.1 (architecture load-bearing); position pooled out (object decodable, position ~chance off C2);
pixel-scramble does not decode; same-vs-held invariance gap <= 0.2. Reported separately (predicted to FAIL, and it
does): template_learning_load_bearing = HMAX-trace beats HMAX-random (a random projection) by 0.1.

## Sources

- Riesenhuber, M. & Poggio, T. (1999). Hierarchical models of object recognition in cortex. *Nature Neuroscience*
  2:1019-1025.
- Serre, T., Wolf, L., Bileschi, S., Riesenhuber, M. & Poggio, T. (2007). Robust object recognition with
  cortex-like mechanisms. *IEEE TPAMI* 29:411-426. (learned/imprinted S2 patches)
- Foldiak, P. (1991). Learning invariance from transformation sequences. *Neural Computation* 3:194-200.
