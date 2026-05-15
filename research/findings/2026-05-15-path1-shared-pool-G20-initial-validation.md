# Path 1 (Catalog G.20) shared-pool distributed encoding — initial signal

## TL;DR

The catalog-G.20-style "single shared pool with per-concept engram tags"
architecture **discriminates above chance** in initial smoke tests:

| N concepts | Pool size | top-1 PASS | Chance | Multiple of chance |
|---|---|---|---|---|
| 8 | 800 neurons | 4/8 (50%) | 12.5% | **4.0×** |
| 16 | 1000 neurons | 7/16 (43.8%) | 6.2% | **7.0×** |
| 32 | 1600 neurons | (in flight, expected ~30-40%) | 3.1% | TBD |

**Per-substrate efficiency vs v16:**
- v16 (16 concepts, 16 dedicated pools × 200 neurons = 3200 neurons):
  77.5% multi-seed PASS → 0.0039 PASS-concepts/neuron
- Shared-pool (16 concepts, 1 pool × 1000 neurons):
  43.8% PASS → 0.0070 PASS-concepts/neuron
- **78% more efficient per neuron** (single-seed comparison)

The catalog G.20 hypothesis — "distributed coding in shared substrate
is more substrate-efficient than pool-per-concept" — receives initial
empirical support.

## Implementation summary

`research/runners/concept_pool_demo_shared.py`:

```python
build_shared_pool_bridge(
    n_lang_input=8192,
    n_shared_pool=1600,   # ONE shared pool for N concepts
    n_shared_fs=200,      # lateral inhibition (catalog J PV-FSI)
    n_lang_output=8192,
)
```

Architecture:
- `language_input` → `shared_concept_pool` (plastic, gated)
- `shared_concept_pool` → `language_output` (reciprocal, plastic, gated)
- `shared_concept_pool` ↔ `shared_FS` (WTA via lateral inhibition)
- No motor pools, no NMDA, no dlpfc — minimal test of distributed encoding

Per-concept training (catalog G.20 + D.14 mechanism):
1. Apply **topographic prior**: word N's lang_input band gets 10× weight
   to shared_pool slice N (e.g. neurons [N*50:(N+1)*50]); 0.1× to others
2. **Interleaved training** (matches v16 recipe): drive lang_input(N) +
   teacher current on slice N + teacher on lang_output(N) for STDP
3. After training, **engram-tag** each concept: drive lang_input(N),
   capture top-K cofiring neurons in shared_pool (catalog D.14 Tonegawa)
4. Eval: stim each engram tag, measure firing per slice; PASS if target
   slice ranks 1

Critical hyperparams (different from v16):
- `topographic_factor=10.0` (vs v16 default 3.0) — need stronger bias
  to dominate random init in shared substrate
- `off_target_factor=0.1` (vs v16 default 0.3) — stronger dampening
- `slice_size=50` — gives 1600 / 50 = 32 concepts max per pool

## Initial results

### 8 concepts (initial smoke)

```
[RESULTS] 4/8 top-1 (50.0%), 6/8 top-5 (75.0%)

word         rank  tgt_rate   max_off
apple            1      183.0       35.0  ← strong
cat              1      461.0       85.0  ← very strong
go               1      192.0      148.0
look             1      138.0       60.0
river            3       38.0      117.0
come             5       27.0      112.0
dog              7        1.0      131.0  ← anomaly: target slice barely fires
stop             8        0.0       54.0  ← anomaly: target slice silent
```

### 16 concepts

```
[RESULTS] 7/16 top-1 (43.8%), 12/16 top-5 (75.0%)
  apple, cat, come, stop, big, cold, tree all top-1
  dog rank 3, sun rank 8, bird rank 5
```

7× chance top-1; 75% top-5 means even when target is not rank 1, it's
usually in top 5 of 16 (chance 31%).

## Open questions

1. **Why do 2/8 and ~9/16 concepts have near-zero target firing?**
   - Specific concepts (dog, stop, sun, moon) consistently lose. May be
     interaction with random init variance + topographic prior application
     order. Worth investigating per-concept.

2. **Does it scale to 32, 64, 128 concepts?**
   - 32 in flight (architecture: 9992 neurons, ~4.3M synapses).
   - Predicted: 30-40% top-1 at 32, 20-30% at 64.
   - If 64 concepts hold ≥ 20%: clear path to vocab expansion beyond
     current multi-bridge ceiling.

3. **Is lang_output cosine readout fixable?**
   - Initial test showed near-random lang_output cosine despite working
     slice discrimination. STDP timing or weight magnitudes may need
     adjustment.
   - Current PASS metric uses slice firing directly; downstream
     lang_output is for chat-REPL spelling readout (deferred).

4. **Multi-seed robustness?**
   - All initial smokes at seed 42. Need seeds 43-46 to confirm.

## Comparison to prior tested architectures

| Architecture | Vocab tested | Result | Substrate (concept neurons) |
|---|---|---|---|
| v16 concept-pool | 16 | 77.5% (5-seed) | 3200 (16 × 200) |
| v17 28-pool | 28 | NEGATIVE (8 hypotheses) | 5600 (28 × 200) |
| Encoding-axis 64-word | 64 (4 dir × 16 syn) | 62.5% primary, 17.5% syn | 8000 (4 × 2000) |
| Encoding-axis 96-word | 96 | NEGATIVE (25% floor) | ~12K |
| **Shared-pool 16 (NEW)** | **16** | **43.8% (seed 42)** | **1000 (1 × 1000)** |

Catalog G.20 explicitly predicted shared-pool distributed coding would
improve substrate efficiency. Initial result supports this.

## Recommended next steps

1. **32-concept smoke** (in flight): test capacity scaling. If 30%+: GO.
2. **Multi-seed (3 seeds × 16 concepts):** confirm seed-42 result isn't
   a fluke. ~30 min wall clock.
3. **Per-concept diagnostic:** why do some concepts have 0 target
   firing? Investigate random init interaction with topographic prior.
4. **Fix lang_output readout:** required for end-to-end chat-REPL
   integration. Currently slice firing is the success metric.
5. **64-concept smoke at full architecture:** if 32 holds, push to
   encoding-axis parity (64 words in 1 shared pool vs encoding-axis
   64 words in 4 motor pools).
6. **Multi-bridge composition:** if single-bridge holds 64+, then
   5 shared-pool bridges × 64 = 320 unique concept words trivially.

## Catalog reference status

- **G.20 Pulvermüller distributed cortical word ensembles**: prior
  status PARTIALLY MISSING. Initial implementation now exists +
  validates above chance at 16 concepts. Catalog status proposed
  update: PROTOTYPE IN PROGRESS.
- **D.14 Tonegawa engram cells**: reused from prior validation. Engram
  tag mechanism is the storage primitive.
- **J (PV-FSI lateral inhibition)**: shared_FS region provides
  cross-concept WTA. Not yet ablated to test contribution.

## Files

- Runner: `research/runners/concept_pool_demo_shared.py`
- Raw data: `research/findings/raw/g11_bg/shared_pool_n{8,16,32}.json`
- Strategic plan: `docs/plans/2026-05-15-vocab-scaling-paths-1-2-3.md`
- This finding doc
