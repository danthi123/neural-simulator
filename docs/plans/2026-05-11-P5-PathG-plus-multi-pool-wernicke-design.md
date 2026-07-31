---
type: plan
status: live
date: 2026-05-11
---

# Path G+ — Multi-pool wernicke design (P5 architectural fallback)

**Date:** 2026-05-11
**Phase:** P5 fallback design (used if iter H fails)
**Catalog:** G.11 + G.13 (ventral semantic stream)
**Estimated effort:** 2-3 hours (code + smoke test)

## Why this exists

P5 iter A-G all failed because wernicke fires uniformly for both
apple and river concepts. The lang→wernicke pathway at density
0.30 averages out per-word differences so all wernicke neurons
receive similar drive intensity. FS lateral inhibition (iter G)
just picks the same intrinsic-property winners for both concepts.

Iter H tests lower density (0.05) for natural per-concept
structural variance. If that fails too, Path G+ is the next
move: PRE-ALLOCATE wernicke sub-pools per concept.

This is the same pattern that worked for Tier 1 motor pools
(2026-05-06): per-action motor_N/E/S/W pools + FS lateral
inhibition + topographic bias. 5/6 seeds PASS W↔A binding.

## Design

### Region structure (replace wernicke)

Old: single wernicke region (200 neurons)

New: 4+ wernicke sub-pools (50 neurons each), with FS:
- `wernicke_pool_0` (50 neurons) — apple-biased
- `wernicke_pool_1` (50 neurons) — river-biased
- `wernicke_pool_2` (50 neurons) — concept 3 (future)
- `wernicke_pool_3` (50 neurons) — concept 4 (future)
- `wernicke_fs_pool_0` (12 PV-FS) — inhibits OTHER pools when pool 0 active
- `wernicke_fs_pool_1`, `wernicke_fs_pool_2`, `wernicke_fs_pool_3` (12 each)

Total: 4 * 50 + 4 * 12 = 248 neurons (vs original 200; slight scale-up).

### Pathways

```
lang_input -> wernicke_pool_X  # density 0.30, with topographic bias
wernicke_pool_X -> wernicke_fs_pool_X  # excite own FS
wernicke_fs_pool_X -> wernicke_pool_Y  # for Y != X (cross-inhibit)
wernicke_pool_X -> semantic_cortex     # density 0.30, weight 4.0 (unchanged)
```

The cross-inhibition (each FS inhibits OTHER pools, NOT own) creates
winner-take-most across concepts.

### Topographic bias

Need a new function `apply_wernicke_topographic_bias(bridge, concepts)`:

```python
def apply_wernicke_topographic_bias(
    bridge,
    concepts: list[str],  # ["apple", "river", ...]
    topographic_factor: float = 1.5,
    off_target_factor: float = 0.7,
):
    """Bias lang_input -> wernicke_pool weights so concept_i's
    active lang neurons preferentially target wernicke_pool_i.
    Mirror of apply_topographic_bias (motor pools) at semantic level.
    """
    for i, concept in enumerate(concepts):
        pool_name = f"wernicke_pool_{i}"
        active_lang = get_active_lang_neurons(concept)
        for other_pool in [f"wernicke_pool_{j}" for j in range(len(concepts)) if j != i]:
            scale_weights(bridge, active_lang, other_pool, off_target_factor)
        scale_weights(bridge, active_lang, pool_name, topographic_factor)
```

### Test methodology

Same as iter A-H: drive lang_input(apple), tag semantic_cortex,
test reactivation. Weight inspection across pools.

NEW weight diagnostic: per-pool reactivation rate. Apple should
preferentially activate wernicke_pool_0 (its assigned pool).

### Implementation steps

1. Add new params to `build_biological_brain_regions`:
   - `enable_multi_pool_wernicke: bool = False`
   - `n_wernicke_pools: int = 4`
   - `n_per_wernicke_pool: int = 50`
   - `n_wernicke_pool_fs: int = 12`

2. In the `enable_ventral_semantic` block, branch:
   - If `enable_multi_pool_wernicke=True`: add per-pool regions + cross-inhibition
   - Else: existing single wernicke region

3. Add `apply_wernicke_topographic_bias` to `text_minimal_isolation.py`.

4. Update `validate_ventral_semantic.py`:
   - CLI flag `--enable-multi-pool-wernicke`
   - When enabled: call topographic bias after build, before training
   - Weight inspection: per-pool selectivity

5. Test smoke build at NumPy backend.

6. Run iter G+ at seed 42:
   ```bash
   python -m research.runners.validate_ventral_semantic \
       --seed 42 --n-train-events 300 --n-replay-cycles 30 \
       --strict-two-stage --drive-lang-during-replay \
       --semantic-cortex-recurrent-density 0.25 \
       --semantic-cortex-recurrent-weight 4.0 \
       --drive-steps 300 \
       --enable-multi-pool-wernicke \
       --out research/findings/raw/g11_bg/p5_iterGplus_seed42.json
   ```

7. If passes: launch 43/44 for multi-seed.

## Caveats

- Doesn't scale to large vocabularies (4 pools = 4 concepts). For
  larger vocab, would need many more pools OR a different scheme.
- Defeats some of the catalog G.13 architectural intent (one
  Wernicke's area, not many). Compromise: it works.
- Requires per-concept pool allocation at architecture-build time.
  Adding new concepts requires rebuild + retrain.

## Pragmatic vs ATL-faithful

The Patterson 2007 ATL hub theory has ONE semantic hub that
encodes all concepts via sparse distributed codes. Path G+
violates that — each concept has a dedicated pool.

But Tier 1's success showed that for the project's current scale
(4-8 concepts), dedicated pools WORK reliably. Multi-seed PASS
5/6. Tier 1 was the only architecture that passed multi-seed
for word→action binding.

The user explicitly accepted "high-risk research bets" but also
wants PROGRESS. Path G+ is a high-leverage pragmatic compromise:
sacrifice ATL biological faithfulness for working bindings.

The catalog-faithful approach (single semantic hub) can be
revisited once the basic conversational loop works at multi-pool
level. Then SCALE UP to many concepts via gradual replacement
of per-concept pools with sparse distributed coding.

## Alternative if iter H AND Path G+ both fail

Possible deeper issues:
1. The training paradigm itself (paired-stim with all gates open)
   doesn't produce per-concept selectivity because both apple
   and river training drive the same wernicke neurons.
2. Need contrastive training: train apple while ACTIVELY
   suppressing river's tag (LTD). Requires new gate logic.
3. Need pre-trained orthogonal codes (Hopfield-style) instead
   of emergent codes.

These are deeper architectural changes (~1+ day each). At that
point, escalate to user for direction.
