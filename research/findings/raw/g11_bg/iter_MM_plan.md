# iter MM contingency plan — stronger topographic bias

**Triggers:** if iter LL (scale-only) fails or marginally passes.

## Hypothesis

iter KK's failure (canon amplifies structural bias) revealed that the
problem is per-seed structural bias amplifying through symmetric
architecture. iter LL tests "scale without canon".

If iter LL still has structural bias (e.g. pool_1 wins for both
concepts), the topographic bias prior (currently 1.5/0.7 = 2.14x
ratio) may be too weak to overpower seed-dependent random
connectivity asymmetries.

Tier 1's biological scale Tier 2.1 architecture worked with the same
1.5/0.7 factor on motor pools — but motor pools have explicit per-
action substrate (somatotopy is real biology). For abstract concepts
(apple/river), there's no real somatotopy — the topographic prior
IS the only thing preventing pool collision.

iter MM: stronger topographic bias (3.0/0.33 = 9x ratio) to force
clean per-concept channelization at lang_input → wernicke.

## Configuration

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --apply-wernicke-topographic \
    --wernicke-topographic-factor 3.0 \
    --wernicke-off-target-factor 0.33 \
    --n-recognition-trials 5 --inter-trial-rest-steps 100 \
    --out research/findings/raw/g11_bg/iter_MM/iter_MM_seed42.json
```

Only differences from iter LL: factor 1.5→3.0, off_target 0.7→0.33.

## Biology

Stronger topographic biases are biology-faithful when the cortex has
clearly-organized maps. Pulvermüller 2001-2003 cortical somatotopy
shows ~5x activation ratio between word-class-specific cortical
zones; our 9x ratio falls within biological range for highly-
organized maps.

## Expected outcome

If structural bias was the issue: iter MM should fix it (5-6/6 BIDIR).
If something deeper is wrong: iter MM will still fail and we need to
rethink (drop semantic_cortex from recognition, or anchored
concepts via visual cortex, or full unified-wernicke + sparse coding).
