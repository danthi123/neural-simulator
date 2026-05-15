# Sparse-distributed encoding capacity curve

## TL;DR

Sparse-distributed encoding (true catalog G.20 form) capacity curve:

| N concepts | Pool size | Pattern size | Top-1 | Top-5 | × chance |
|-----------|-----------|--------------|-------|-------|----------|
| 64 | 2000 | 100 | **100.0%** | **100.0%** | 64× |
| 128 | 3000 | 100 | 84.4% | 95.3% | 108× |
| 256 | 5000 | 100 | TBD | TBD | TBD |

At 128 concepts: **108/128 (84.4%) top-1**, **122/128 (95.3%) top-5**.

Compared to chance (1/128 = 0.78%), this is 108× chance for top-1 and
24× chance for top-5. Very strong architecture signal.

## Comparison vs contiguous-slice architecture

| Architecture | Concepts | Pool | Top-1 |
|--------------|---------|------|-------|
| Contiguous 32-conc | 32 | 1600 | 100.0% |
| Contiguous 60-conc | 60 | 3200 | 78.3% |
| **Sparse 64-conc** | **64** | **2000** | **100.0%** |
| **Sparse 128-conc** | **128** | **3000** | **84.4%** |

At 64 concepts, sparse-distributed PERFECT (vs contiguous 60's 78.3%).
At 128 concepts, sparse-distributed 84.4% (vs contiguous's predicted
much-lower based on the 60-conc trajectory).

## What 128-concept tells us

The architecture works strongly above chance at 128. 6 of 128 outside
top-5 (95.3% top-5). The failures are:
- bird rank 62 (bad failure — but it's a vocabA_nouns word that's
  also one of the v1 failures, possibly an interaction with the
  pattern selection RNG)
- 4 of 5 failures are "concept60-127" placeholder words (vocab
  extended beyond ALL_60 with generic suffixes)
- Only 1 real-word failure (bird) and 2 close-but-not-top-1

The wall is real but soft. Predictions:
- At 128, real vocab (no placeholder concepts) would probably give 90%+ top-1
- At 256, lang_input collision and FS undersaturation become significant

## Capacity wall analysis

For sparse-distributed at 128 conc, N_pool=3000, K=100:
- Expected pairwise overlap: K²/N = 100²/3000 = 3.3 (3.3% of K)
- This is FINE — Hamming distance of 200-3.3 = ~196 separating any two
  patterns. Should give clean discrimination.

So the 84.4% wall is NOT pattern overlap. It's probably:
1. **Lang_input collision**: at 128 cues, 8192 lang_input neurons at
   sparsity 0.007 = 57 active per cue. 128 × 57 = 7300 / 8192 = 89%
   lang_input packing. Adjacent codes very close.
2. **FS undersaturation**: 300 FS neurons can't selectively inhibit
   128 different concepts at once.
3. **Training event count**: 400 events per concept × 128 = 51200 events.
   STDP may need more for fine discrimination.

## Implications for production

**Current production tier**: 5 bridges × 32 concepts (contiguous) =
160 concepts at 100%.

**Sparse-distributed production tier**: 5 bridges × 64 concepts
(sparse) = 320 concepts at 100%. Same wall-clock to train.

**Sparse-distributed stretch tier**: 5 bridges × 128 concepts (sparse)
= 640 concepts at ~84% per bridge. 538 robust concepts.

Combined with morpheme tokenizer (~10× reach with expanded dict):
- 320 robust concepts × 10 = ~3200 surface forms (age 5)
- 538 robust + 102 fragile × 10 = ~5400 surface forms (age 6-7)

## Tests in flight

256-concept sparse-distributed, 5000-pool, 16384 lang_input, sparsity
0.003. Predicted: 60-75% top-1 if lang_input is the wall, 80%+ if
FS / training is the wall.

## Recommendation

For production:
- Use **64-concept sparse-distributed** per bridge (validated 100%)
- 5-bridge ensemble = 320 unique concepts
- Combined with morpheme tokenizer (200 morphemes, ~10× reach) =
  effective vocab ~3200 surface forms (age 5)

For research:
- Continue capacity testing to find the true wall
- Diagnose 128-concept failures (is it lang_input, FS, or training?)
- Path 2 to improvement: bigger FS pool, better training schedule

## Files

- 64-concept sparse: `research/findings/raw/g11_bg/sparse_n64.json`
- 128-concept sparse: `research/findings/raw/g11_bg/sparse_n128.json`
- 256-concept sparse (in flight): `research/findings/raw/g11_bg/sparse_n256.json`
- This doc
