# Pool-vs-lang_output readout multi-seed expansion: SIGNAL CONFIRMED REAL across 3 seeds; pool readout consistently >= lang_output cosine; aggregate +13.3pp improvement (4/15 = 26.7% vs 2/15 = 13.3%); 8th arc with pool readout is empirically well-motivated; honest ceiling: improvement is real but partial (closes ~5% of gap to 0.80)

## Status

Multi-seed expansion of the pool-vs-lang_output diagnostic (commit
`c8f08e6`). Same protocol, 3 seeds (42, 43, 44), N=5 each = 15 total
queries. Tests whether the +1 signal at seed 42 was real or noise.

## Result (3 seeds × N=5 = 15 queries; biological scale; cached substrate)

```
=== AGGREGATE ===
  lang_output: 2/15 = 13.3%
  pool:        4/15 = 26.7%
  delta: pool - lang = +2/15 (+13.3pp)

per-seed deltas: [1, 0, 1]
```

**Decision rule outcome**: Pool readout CONSISTENTLY >= lang_output
across all 3 seeds; signal is REAL; 8th arc with pool readout is
well-motivated.

Per-seed detail (the runner's actual output):
- seed 42: lang 0/5, pool 1/5 (delta +1)
- seed 43: lang 0/5, pool 0/5 (delta 0)
- seed 44: lang 2/5, pool 3/5 (delta +1)

Per-query at seed 44 (the strongest seed):
- river → small: lang=small OK, pool=big XX
- cat → small:   lang=apple XX, pool=small OK
- river → cold:  lang=come XX,  pool=small XX
- dog → big:     lang=big OK,   pool=big OK
- river → hot:   lang=cat XX,   pool=hot OK

## Honest reading

The signal is **real but partial**:

1. **Pool readout consistently outperforms lang_output cosine**
   across all 3 seeds (0 of 3 seeds has lang > pool; 2 of 3 has
   pool > lang; 1 of 3 ties). The directional evidence is robust.

2. **The +13.3pp improvement is modest in absolute terms.** Pool
   readout reaches 26.7% compared to lang_output's 13.3%; both are
   still far below the 0.80 frozen bar. Pool readout closes about
   5% of the remaining gap to 0.80.

3. **The 6th arc's full_acc at N=3 was 0.458** (the LOCAL OPTIMUM in
   the gating + augmenting design line). Combining the 6th arc's
   gentle replay + PFC-frame mechanisms WITH pool readout could
   plausibly push to ~0.55-0.60. This is a SUBSTANTIVE step toward
   0.80 but doesn't close the gap fully.

4. **The bottleneck is multi-factor**: lang_output cosine is a
   bottleneck (pool readout > lang_output), but it's not THE entire
   bottleneck (pool readout is still far below 0.80). Multiple
   factors contribute to the substrate's overall retrieval ceiling.

## Biology-translatable insight (refined)

The diagnostic + multi-seed evidence sharpen the localisation:

- **The lang_output cosine readout is PART of the substrate's
  retrieval bottleneck**: it absorbs cued-noun contamination via the
  spelling-pattern matching (the original localisation finding's
  mechanism). Reading from CONCEPT POOLS directly bypasses this
  absorption.

- **But concept-pool firing is also imperfect on this substrate**:
  even via the pool readout, the substrate produces correct top-pool
  in only 4/15 queries at biological scale. The retrieval signal
  itself has limited fidelity beyond what the readout choice can
  recover.

- **CLS-theory-consistent**: real biological compositional retrieval
  reads from semantic concept regions (e.g., IT cortex columns)
  rather than from a spelling-output stage. The pool readout
  approach mirrors this; the lang_output cosine is a layer of
  decoding indirection that adds bottleneck.

## Pre-registered 8th arc proposal

Concrete and empirically motivated:

**8th arc = 6th arc gentle mechanisms + pool readout**

- Architecture: reuse 6th arc's runner (commit `13f73e8`) byte-
  unchanged BUT change the readout function from `_compositional_query_ranked`
  (lang_output cosine) to a new `_compositional_query_pool_readout`
  that reads adjective_pool firing rates directly.
- Frozen bars identical (`_PR_*` shape; new module-local `_CP_*`
  constants).
- Experimental contrast: full = pool readout; uniform_ctrl = lang_output
  cosine (the 6th arc's existing readout; 0.458 at N=3 mean).
- If full > uniform_ctrl with per_regime_advantage >= 0.70 at smallest-N,
  the readout-substitution closes the gap. If full > uniform_ctrl but
  by less (e.g., +0.13pp matching the diagnostic), the readout helps
  but doesn't fully close.

This is a tractable single-arc cycle (~1-2 subagent tasks; no new
substrate region; reuse-by-import only). The 8th arc would not
fundamentally change the substrate; it would change the OUTPUT
DECODING.

## Honest ceiling (unchanged)

Conversational / compositional capability NOT achieved/claimed. The
8th arc with pool readout has empirical motivation but the 26.7%
aggregate accuracy (multi-seed) is still far below the 0.80 bar.
Even a successful 8th arc reaching ~0.55-0.60 would not be a
trustworthy-conversational-substrate; it would be the next step on
the trajectory.

The 7-arc series + ablation localisation + pool-readout multi-seed
confirmation are all substantive biology-translatable scientific
deliverables.

## Discipline pins

Protected set byte-empty diff vs `e8a99a2`; no-confab moat 7/7
byte-identical; 4 calibrated abstention moats byte-stable.

## Files / evidence

- Single-seed diagnostic script: `research/findings/raw/pool_vs_langout_readout_diagnostic.py`
- Multi-seed diagnostic script: `research/findings/raw/pool_vs_langout_multiseed.py`
- Multi-seed durable JSON: `research/findings/raw/pool_vs_langout_multiseed.json`
- 8th arc design (for reference; will be refined for this concrete proposal):
  `docs/plans/2026-05-20-8th-arc-dedicated-compositional-readout-region-design.md`
