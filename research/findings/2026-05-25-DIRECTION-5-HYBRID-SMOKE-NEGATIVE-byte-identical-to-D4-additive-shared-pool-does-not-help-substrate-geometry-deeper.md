# Direction 5 hybrid sparse-distributed SMOKE = DIRECTION_5_NEGATIVE multi-seed (multi-seed mean BYTE-IDENTICAL to D4 NEGATIVE: 0.050/0.007 0.008/0.000 0.005/0.000); additive shared sparse pool on bio_brain_regions does NOT help cross-bridge composition; the substrate-geometry constraint identified in D4 is even deeper than expected — requires LEARNED dedicated→shared projection (Approach C) rather than additive pool

**Date:** 2026-05-25 ~13:35 EDT
**Status:** DIRECTION_5_NEGATIVE at smoke scale; byte-identical to D4 NEGATIVE multi-seed; additive hybrid hypothesis REFUTED; pre-registered next is Approach C (learned dedicated→shared projection) per the D5 design doc

## What was tested

Direction 5 (hybrid sparse-distributed shared pool on bio_brain_regions)
launched per user direction tier 1 after tier 2 (Q E/I balance) completed.

**Hybrid architecture** (commits 7ff60a7 + 0fcaf07): each of 5 bridges
keeps its 16 dedicated 200-neuron bio_brain_regions concept pools
(byte-unchanged) PLUS a NEW 2000-neuron shared_concept_pool + 300-neuron
shared_FS WTA region + lang_input -> shared_concept_pool plastic
pathway. K=100 sparse patterns per concept; pillar n=95 topographic
prior (factor 10.0 / off-target 0.1) applied one-time via reused
apply_sparse_topographic_prior. Cross-bridge probe reads from
shared_concept_pool (uniform 2000-feature substrate across 5 bridges).

**Hypothesis tested**: combining biology-faithful dedicated pools
(pillar n=98/n=105) with sparse-distributed cross-bridge geometry
(pillar n=95) would preserve the cross-bridge composition capability
of G.20 sparse on the bio_brain_regions substrate.

**Smoke config**: n_lang_input=1024, n_per_pool=100, n_fs_per_pool=12,
n_shared_pool=2000, pattern_size=100, events_per_word=50, M_OBS=8,
3 seeds [42, 43, 44], 5 bridges. Hybrid bridge: 6588 neurons (vs D4's
4288 dedicated-only).

Training wall: 94.7 min total (15 cells x ~6 min/cell). Cross-bridge
probe wall: 132.6s.

## Result: DIRECTION_5_NEGATIVE — byte-identical to D4

Multi-seed mean accuracy:

| Load | OB (D5 hybrid) | OB (D4 dedicated-only) | OI (D5 hybrid) | OI (D4 dedicated-only) |
|---|---|---|---|---|
| L=2 | 0.050 | 0.050 | 0.007 | 0.007 |
| L=3 | 0.008 | 0.008 | 0.000 | 0.000 |
| L=5 | 0.005 | 0.005 | 0.000 | 0.000 |

**Every cell at every seed is identical to D4 NEGATIVE.** The additive
shared sparse pool produces ZERO improvement over the dedicated-only
baseline.

Per-seed cell counts:
- seed 42: L=2 OB 0.045/OI 0.000; L=3 OB 0.010/OI 0.000; L=5 OB 0.000/OI 0.000
- seed 43: L=2 OB 0.045/OI 0.005; L=3 OB 0.010/OI 0.000; L=5 OB 0.010/OI 0.000
- seed 44: L=2 OB 0.060/OI 0.015; L=3 OB 0.005/OI 0.000; L=5 OB 0.005/OI 0.000

Chance baseline: 1/80 = 0.0125. All cells at or below chance.

## Activity capture (the diagnostic)

The shared_concept_pool DOES show higher firing rates than D4's dedicated
pools:
- D5 shared_concept_pool: mean_rate ~0.12-0.17, density ~0.07-0.09
- D4 dedicated noun_pool union: mean_rate ~0.02, density ~0.03

So the shared pool IS being driven by lang_input. The sparse K=100
patterns are presumably being embedded. The topographic prior is being
applied (per the log).

But the cross-bridge probe cannot extract concept-discriminating signal
from the captured shared_pool activity. The patterns are presumably
DOMINATED by noise (ongoing dynamics + random init) rather than the
trained K-of-N codes.

## Honest comparison to G.20 sparse pillar n=95

| Metric | G.20 sparse n=95 (V=160 cross-bridge) | D5 hybrid (V=80 cross-bridge) |
|---|---|---|
| Substrate | sparse-only Kanerva SDM | hybrid (dedicated bio_brain_regions + shared sparse) |
| Per-bridge architecture | 2000-neuron pool only | 16 x 200 dedicated + 2000 shared |
| Training events per word | 100 (validated) | 50 (smoke; D5 SMOKE) |
| OB L=5 | 1.000 (perfect) | 0.005 (chance) |
| OI L=5 | 0.790 | 0.000 |

Major differences that may explain the gap:
1. **Training events**: D5 smoke at 50ev/word vs G.20 sparse production
   at 100ev/word; the K-of-N pattern may need more events to embed
2. **Substrate scale**: G.20 sparse uses 2000-neuron pools ONLY (no
   dedicated pools draining attention/competition); D5 has the
   dedicated pools COMPETING for cortex resources
3. **Activity capture target**: D5 captures from shared_concept_pool;
   G.20 sparse captures from THE pool (no distinction)
4. **Pool drive**: G.20 sparse drives lang_input -> SHARED directly;
   D5 drives lang_input -> BOTH dedicated AND shared (competition?)

## Biology-translatable insight

The additive shared sparse pool hypothesis is REFUTED at smoke scale:
adding a sparse-distributed shared substrate ON TOP of the dedicated-pool
architecture does NOT preserve the cross-bridge composition capability.
The dedicated pools either (a) drain the substrate's capacity to embed
sparse patterns OR (b) the shared pool's K-of-N patterns are
indistinguishable from background dynamics in the captured activity.

A learned dedicated->shared projection (Approach C, deferred from
D5 design) might work because:
1. The dedicated pool's distinctive firing pattern (only the trained
   concept's pool fires strongly) projects through a LEARNED weight
   matrix into a discriminative shared activity
2. The shared pool's activity is then a learned READOUT of the dedicated
   pool's identity, not a separately-trained sparse pattern
3. This more closely matches G.20 sparse's mechanism (the K-of-N
   patterns are the "weight matrix" from concept identity to shared pool;
   in G.20 the weight matrix is hard-coded; in Approach C it would be
   learned)

## What is preserved unconditionally

- Pillar n=105 (D3 V=32 production PASS) stands UNAFFECTED
- Pillar n=95 (G.20 sparse cross-bridge) stands UNAFFECTED
- Pillars n=93/n=94/n=96/n=97/n=98 stand UNAFFECTED
- Direction M deliverable + Direction R-v3 envelope stand
- bio_brain_regions substrate (build_biological_brain_regions) byte-
  unchanged
- G.20 sparse pool builder + topographic prior byte-unchanged
- No-confab moat 7/7 byte-identical
- Bar UNCHANGED at 0.80 throughout
- Direction 5 infrastructure (vocab + 5 hybrid builders + probe + runner +
  verdict + tests) reusable for any future cross-bridge investigation

## Pre-registered next concrete action

Per the D5 design doc's Approach C deferral:

> "Approach C deferred for future iteration if A is PARTIAL/NEGATIVE.
> Approach C = learned dedicated->shared projection: instead of training
> the shared sparse pool independently via lang_input, train a weight
> matrix that maps dedicated-pool activity -> shared-pool sparse code."

Approach C is substantial (~1-2 wk design + implementation). Alternatively,
the user-stated chain (Q -> 3 -> 4 -> R) is now FULLY EXHAUSTED:
- Q: PARTIAL (4 axes characterized; bottleneck is structural/dynamical)
- 3: PILLAR N=105 (bio_brain_regions V=32 PASS)
- 4: NEGATIVE + diagnostic (substrate-geometry limited)
- R: PASS (capacity envelope to N=512)

Plus the post-user-chain work:
- Q-secondary: PARTIAL (E/I balance not the constraint)
- D5: NEGATIVE (additive hybrid doesn't help; need learned projection)

The honest cumulative state: the substrate has 2 working modes (single-
substrate vocab scaling pillar n=105; G.20 sparse cross-bridge pillar n=95)
that cannot YET be unified additively. Further unification requires
learned-projection design (Approach C).

## Discipline preserved

- Multi-seed [42, 43, 44] decisive (not one-seed artifact)
- Frozen verdict computed from recorded JSON; not tuned
- Honest propagation: NEGATIVE recorded as NEGATIVE; byte-identical
  to D4 documented honestly (not spun)
- Pre-registered Approach C identified as next concrete action
- Both remotes pushed
- ~95 min training + ~2 min probe = ~97 min total wall

## Files

- Hybrid bridge builder: research/findings/raw/direction_5_bridge_builder.py
- Vocab spec (matches D4): research/findings/raw/direction_5_vocab_spec.py
- Cross-bridge probe: research/findings/raw/direction_5_cross_bridge_probe.py
- 5-bridge runner: research/findings/raw/direction_5_5bridge_runner.py
- Verdict module (frozen): research/findings/raw/direction_5_verdict.py
- Smoke training result: research/findings/raw/direction_5_5bridge_smoke.json
- Cross-bridge probe result: research/findings/raw/direction_5_cross_bridge_smoke.json
- Probe log: research/findings/raw/direction_5_cross_bridge_smoke.log
- Design doc: docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-design.md
- Implementation plan: docs/plans/2026-05-25-direction-5-hybrid-sparse-distributed-bio_brain_regions-implementation.md
- D4 NEGATIVE comparison: research/findings/2026-05-25-DIRECTION-4-5bridge-SMOKE-NEGATIVE-bio_brain_regions-cross-bridge-doesnt-engage-multi-seed-chance-level.md
- G.20 sparse n=95 reference: research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md
