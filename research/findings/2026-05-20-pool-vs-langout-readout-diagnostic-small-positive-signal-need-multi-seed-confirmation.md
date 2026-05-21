# Pool-vs-lang_output readout diagnostic: small positive signal for the 8th arc Direction A; pool readout 1/5 correct vs lang_output 0/5 at seed 42 N=5; signal is directionally consistent with the localisation finding but too weak at single-seed to confidently motivate the full 8th arc

## Status

Controller-only diagnostic following the 7-arc + ablation + day-
consolidated findings (commit `351366f`). The 8th arc design (commit
`be78d14`) proposed Direction A (dedicated compositional-readout
region) but discovered the brain-region framework doesn't support
post-construction `add_region`. The simpler alternative: use existing
adjective_pool_* regions via `cp_firing_states` reads instead of
lang_output cosine. This diagnostic tests whether that simpler
alternative actually helps.

## Diagnostic result (seed 42; N=5; biological scale; cached substrate)

For each of 5 (cue, bound-adj) pairs encoded via the standard
`_encode_facts` helper, then queried via TWO readouts:

| Query | Target | lang_output top | lang correct? | pool top | pool correct? |
|-------|--------|-----------------|----------------|-----------|----------------|
| apple | cold | go | XX | small | XX |
| apple | hot | big | XX | small | XX |
| apple | small | go | XX | big | XX |
| cat | small | go | XX | big | XX |
| cat | big | dog | XX | **big** | **OK** |

**Summary:**
- lang_output readout: **0/5 correct**
- pool readout: **1/5 correct**
- Pool readout HELPS marginally (+1)

## Interpretation

The signal is **directionally positive**: pool readout outperforms
lang_output cosine by 1 query out of 5 at seed 42 N=5. This is
consistent with the localisation finding (lang_output cosine is
bottlenecked by cued-noun-diffuse-drive contamination; pool firing is
more selective).

But the signal is **small and single-seed**:
1. +1 out of 5 = 20% absolute improvement at single-seed; statistical
   power is low.
2. Both readouts are still below random-chance threshold (5
   adjectives among 4 possible top words; random guess accuracy is
   ~25%). Pool readout at 20% is essentially random; lang_output at
   0% is below random.
3. The lang_output 0/5 here is lower than the 6th arc's N=5 mean of
   0.273 (~1.4/5). The diagnostic's protocol may not be perfectly
   apples-to-apples with the 6th arc runner's eval (specifically: the
   diagnostic does cue + tag stim in sequence; the 6th arc runner
   does them with specific timing per its `_compositional_query_ranked`).

## Honest reading

The +1 signal is **too weak alone to commit to a full 8th arc build**.
At biological scale a single seed's small N=5 has high variance; the
+1 could be noise as easily as real signal.

The honest pre-registered next step has two paths:

(A) **Multi-seed diagnostic expansion**: re-run the same diagnostic at
seeds 43 and 44; if the +1 advantage persists or grows (e.g., +2 or +3
per 5 across seeds), the signal is real and motivates building the
full 8th arc with pool readout. If it varies (e.g., +1 / 0 / -1 across
seeds), the +1 was noise and the 8th arc isn't well-motivated.

(B) **Honest closure of the gating + augmenting + readout-variation
composition design line**: the 7-arc + ablation + pool-readout
diagnostic series IS the scientific deliverable. No combination of
gating + augmenting + readout-variation crosses the 0.80 bar at
biological scale on this substrate using only already-validated
subsystems. Future work requires fundamentally different mechanisms
(new connectivity; new consolidation primitives; new substrate
architecture).

Path (A) is a small additional investment (~15 min controller-only
diagnostic) that resolves the noise-vs-signal question. Path (B) is
the responsible terminus given the substantive findings already
propagated.

Per the standing autonomy + iterate-following-biology + the small
positive signal here, **path (A) is the cheap-next-step** that
informs which path (8th arc OR closure) is right.

## Files / evidence

- Diagnostic script: `research/findings/raw/pool_vs_langout_readout_diagnostic.py`
- Diagnostic durable JSON: `research/findings/raw/pool_vs_langout_readout_diagnostic.json`
- 8th arc design (for reference): `docs/plans/2026-05-20-8th-arc-dedicated-compositional-readout-region-design.md`

## Discipline pins

Protected set byte-empty diff vs `e8a99a2`; no-confab moat 7/7
byte-identical; 4 calibrated abstention moats byte-stable. Honest
ceiling unchanged.
