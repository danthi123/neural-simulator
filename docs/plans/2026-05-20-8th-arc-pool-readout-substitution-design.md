# 8th arc: pool-readout substitution (empirically motivated; supersedes the earlier "dedicated region" design)

> **Note:** This design SUPERSEDES the earlier `docs/plans/2026-05-20-8th-arc-dedicated-compositional-readout-region-design.md` (commit `be78d14`). The earlier design proposed a dedicated readout REGION but discovered the brain-region framework doesn't support post-construction `add_region`. The empirical evidence from the pool-vs-lang_output multi-seed diagnostic (commit `4d6a3a6`) showed that the simpler approach -- reading directly from existing concept pools via `cp_firing_states` -- ALREADY consistently outperforms lang_output cosine by +13.3pp. This design pivots to that simpler implementation.

## Status

Pre-registered NEW design grounded in two empirical findings:

1. **Cross-arc trajectory (commit `9693685` + `0ef9b6e`)**: 6th arc was
   the LOCAL OPTIMUM at N=3 full_acc = 0.458; 7th arc REGRESSED
   (-0.095) due to over-consolidation; the gating + augmenting
   composition design line is asymptotically capped without readout
   refinement.

2. **Pool-vs-lang_output multi-seed diagnostic (commit `4d6a3a6`)**:
   Pool readout CONSISTENTLY outperforms lang_output cosine across all
   3 seeds (per-seed deltas [+1, 0, +1]; aggregate +13.3pp; 4/15 vs
   2/15). The signal is REAL. The readout choice is part of the
   bottleneck.

## 1. The capability under test (falsifiable; identical frozen bars)

Same compositional retrieval task. Frozen bars identical to all
prior arcs (`_CP_*` module-local constants, values UNCHANGED):
- `_CP_FULL_MIN = 0.80`
- `_CP_UNIFORM_CTRL_MAX = 0.10`
- `_CP_DIRECT_RETAIN_MIN = 0.80`
- `_CP_ABSTAIN_CORRECT_MIN = 0.90`
- `_CP_SCALE_TOL = 0.10`
- `_CP_LADDER = (2, 3, 5)`
- `_CP_MIN_SEEDS = 3`

CP = Compositional Pool readout.

## 2. The mechanism being added (load-bearing)

**Pool-readout substitution**: replace the `_compositional_query_ranked`
function (lang_output cosine pattern match) with a new
`_compositional_query_pool_readout` function that reads the
adjective_pool_* firing rates directly via `cp_firing_states` and
ranks by firing rate.

### `_compositional_query_pool_readout` (the genuine net-new piece)

```python
def _compositional_query_pool_readout(
    bridge, cue_noun, tag_name, dims, recall_steps
):
    """Read compositional output via adjective_pool firing rates after
    cue + tag stim. BYPASSES lang_output cosine.

    1. Drive lang_input(cue) for recall_steps (cue-drive phase)
    2. Stim engram tag for recall_steps (tag-stim phase)
    3. Measure adjective_pool_BIG/SMALL/HOT/COLD firing rates from
       accumulated spike counts during the tag-stim phase
    4. Rank by firing rate; return as ranked list
    """
    from research.runners.compose_concept_engram import (
        lang_output_pattern_during_input,
    )
    # Phase 1: drive cue (uses helper for protocol consistency)
    lang_output_pattern_during_input(
        bridge, cue_noun,
        n_lang_input=int(dims["n_lang_input"]),
        sparsity=float(dims["sparsity"]),
        n_words_for_orthogonal=int(dims["n_words_for_orthogonal"]),
        stim_steps=int(recall_steps),
    )

    # Phase 2: stim engram tag + accumulate firing per adjective pool
    pool_names = [
        ("big", "adjective_pool_BIG"),
        ("small", "adjective_pool_SMALL"),
        ("hot", "adjective_pool_HOT"),
        ("cold", "adjective_pool_COLD"),
    ]
    pool_indices = {
        word: bridge.region_manager.indices(region_name)
        for word, region_name in pool_names
    }
    pool_spike_counts = {word: 0 for word, _ in pool_names}

    if tag_name is not None and tag_name in {
        t["name"] for t in bridge.list_engram_tags()
    }:
        bridge.stimulate_tag(tag_name, drive_pA=1500.0)
        for _ in range(int(recall_steps)):
            bridge._run_one_simulation_step()
            firing = bridge.cp_firing_states
            if hasattr(firing, "get"):
                firing = firing.get()
            for word, indices in pool_indices.items():
                pool_spike_counts[word] += int(firing[indices].sum())
        bridge.clear_tag_drive(tag_name)

    # Convert spike counts to firing rates (normalised by neurons × steps)
    n_per_pool = int(dims["n_per_pool"])
    rates = {
        word: float(pool_spike_counts[word]) / float(n_per_pool * recall_steps)
        for word, _ in pool_names
    }
    ranked = sorted(
        [(word, rate, "pool_readout") for word, rate in rates.items()],
        key=lambda x: -x[1],
    )
    return ranked
```

### Experimental contrast

- **FULL arm**: 6th arc mechanisms (gentle 20-cycle replay + 10-step
  PFC-frame) + pool readout
- **UNIFORM_CTRL arm**: 6th arc mechanisms + lang_output cosine readout
  (the existing 6th arc baseline; 0.458 at N=3 mean)

If full > uniform_ctrl by per_regime_advantage >= 0.70 at smallest-N,
the readout substitution closes the gap. If smaller (e.g., +0.13
matching the diagnostic), readout substitution helps partially. The
8th arc tests how much of the 0.342 gap to 0.80 the pool readout can
close beyond the 6th arc's 0.458.

## 3. Implementation route (purely runner-side; no substrate change)

Mirror the 6th arc runner structure exactly. The ONLY changes:

1. Add the new `_compositional_query_pool_readout` function (above).
2. In the eval arm: FULL uses `_compositional_query_pool_readout`;
   UNIFORM_CTRL uses the existing `_compositional_query_ranked`
   (reused byte-unchanged from the unified runner).
3. Structural-effect probe: replay-effect probe (same as 6th arc)
   + pool-vs-lang_output readout probe (NEW; verify the two readouts
   produce DIFFERENT ranked outputs on the same bridge state).
4. RNG isolation (per 8th-review lesson) + cache-scale validation
   (per 13f73e8 lesson) preserved.

## 4. Pre-registered next staged step

Mirror all prior arcs:
- Task 0: grounding pin
- Task 1: frozen verdict (`_CP_*`)
- Task 2: net-new runner with pool readout
- Task 3: 13th adversarial review
- Task 4: no-harm verification
- Task 5: controller-only decisive run + smell-test + honest propagation

## 5. Honest ceiling

- A full_acc > 0.80 at N=3 would be the FIRST architecture to cross
  the bar. Possible but unlikely given the multi-seed diagnostic
  showed only +13.3pp improvement.
- A full_acc ~ 0.55-0.60 at N=3 would be the most likely outcome
  per the trajectory analysis (6th arc baseline 0.458 + diagnostic
  improvement). This continues the cross-arc gap-closure trajectory
  but doesn't fully close.
- A full_acc <= 0.458 (matching 6th arc) would indicate the readout
  substitution doesn't add net value at biological scale; the
  diagnostic's single-query measurement may not generalize to
  multi-pair encoding settings.
- A FAIL with no further gap-closure -> the substrate's compositional
  retrieval mechanism is genuinely asymptotic; honest closure of the
  design line.

## 6. Discipline pins (mirror prior 7 arcs)

NO bar change; NO protected file modification; NO autograd; NO
declare-unfit; mandatory dedicated adversarial review BEFORE no-harm
BEFORE decisive; honest propagation every outcome both remotes;
same-turn discipline; 4 calibrated moats + no-confab moat
byte-stable.
