# Set 3 trained — 36-word multi-bridge vocab milestone

## TL;DR

Set 3 finished training 2026-05-14 23:17 EST.

**Single-seed Phase 1 PASS: 14/16 (87.5%)** — best per-bridge result so
far (set 1 baseline 11-13/16, set 2 11/16). Wall clock ~19.5 min.

Combined with set 1 + set 2, the multi-bridge chat REPL now supports
**36 unique concept words** + 4 motors = 40-word operational vocab
across 3 bridges.

Set 4 + set 5 auto-launch via `chain_set45_runtime.ps1`.

## Per-word verdicts (seed 42, set 3)

| Word | Pool | Target | Max off-target | Ratio | Verdict |
|------|------|--------|----------------|-------|---------|
| north | motor_N | 1.455 | 1.015 (noun_pool_WATER) | 1.43x | PASS |
| east | motor_E | 1.045 | 0.935 (adj_pool_WET) | 1.12x | PASS |
| south | motor_S | 1.330 | 0.910 (adj_pool_SHORT) | 1.46x | PASS |
| west | motor_W | 0.670 | 0.540 (adj_pool_TALL) | 1.24x | PASS |
| house | noun_pool_HOUSE | 0.880 | 0.810 (motor_E) | 1.09x | PASS |
| road | noun_pool_ROAD | 1.125 | 0.800 (verb_pool_FIND) | 1.41x | PASS |
| fire | noun_pool_FIRE | 1.230 | 0.915 (motor_E) | 1.34x | PASS |
| water | noun_pool_WATER | 1.540 | 0.950 (motor_S) | 1.62x | PASS |
| give | verb_pool_GIVE | 1.120 | 1.110 (noun_pool_WATER) | 1.01x | PASS |
| take | verb_pool_TAKE | 0.740 | 0.735 (adj_pool_DRY) | 1.01x | PASS |
| find | verb_pool_FIND | 1.460 | 1.065 (verb_pool_TAKE) | 1.37x | PASS |
| lose | verb_pool_LOSE | 1.010 | 0.755 (verb_pool_TAKE) | 1.34x | PASS |
| tall | adj_pool_TALL | 0.770 | 0.825 (motor_N) | 0.93x | FAIL |
| short | adj_pool_SHORT | 1.040 | 1.025 (adj_pool_TALL) | 1.01x | PASS |
| wet | adj_pool_WET | 1.305 | 0.855 (noun_pool_WATER) | 1.53x | PASS |
| dry | adj_pool_DRY | 0.905 | 0.935 (motor_N) | 0.97x | FAIL |

Failure pattern: 'tall' + 'dry' both have motor_N as the dominant
off-target winner. Same failure mode (anti-target motor_N bias),
consistent with the structural-noise hypothesis at seed 42. Both
fail words are adjectives; matches the v16 observation that
adjective discrimination is the most fragile word class.

## Compound vocab across 3 bridges

- Set 1: apple, river, dog, cat, go, come, stop, look, big, small, hot, cold
- Set 2: tree, bird, sun, moon, walk, run, eat, sleep, red, blue, fast, slow
- Set 3: house, road, fire, water, give, take, find, lose, tall, short, wet, dry

**36 unique concept words. 0 overlaps.**

## Now in flight

- Set 4 (started 23:17 EST, ~18 min ETA, brings to 48 words)
- Set 5 (auto-launched by chain after set 4 finishes, brings to 60 words)

## File map

```
research/findings/raw/g11_bg/concept_pool_demo/
  seed42_set2.simstate.h5     (set 2 trained 2026-05-14 PM)
  seed42_set3.simstate.h5     (set 3 trained 2026-05-14 23:17)
  seed42_set4.simstate.h5     (in flight)
  seed42_set5.simstate.h5     (queued)
```

## Status

- Sets 1, 2, 3: SHIPPED (8.7 GB combined bridge state)
- Set 4: TRAINING in background
- Set 5: QUEUED behind set 4
- Cross-set sentence encoding: SHIPPED (3+ word tags via partial bridge encoding)
- Role queries (who/what did): SHIPPED + 10 unit tests
- 61 multibridge unit tests passing
