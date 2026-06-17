# Free latency lever: the resonate window shortens 208 → 40 steps with no accuracy loss (6.2× fewer steps)

**Date:** 2026-06-17 (the resonator-paper bonus lever, on top of the CYCLE-152 batched scan)
**Status:** **GO (validated lever).** The composer's resonate window can drop from `period=200` (208 steps) to
`period=32` (40 steps) — **6.2× fewer steps, ~3.9× faster/query** — with who/what recall AND the no-confab moat
still at full accuracy (3 seeds). Free: a constructor knob, NO `sim/` edit. Adoption (changing the default) is
gated on re-running the full conversational suite at the shorter window.
**Runner:** `research/runners/_phaseB_resonate_period_sweep.py` · **Raw:** `research/findings/raw/_resonate_period_sweep.json`

## Why

The owner-supplied resonator paper (arXiv:2208.12880) represents a phasor on Loihi with a **16-timestep cycle**,
where our `RFPhasorComposer` runs **208** resonate steps per op. The op cost is ~entirely the per-step loop
(CYCLE-151 profile: 97.7%), so fewer steps = proportionally faster, for free.

## Result — period sweep, 3 seeds (42, 43, 44), an 8-fact store

| period | steps | who/what acc | moat | ms/query |
|---|---|---|---|---|
| 8 | 16 | 0.000 | ok | 49 |
| 16 | 24 | 0.000 | ok | 64 |
| 24 | 32 | 0.792 | ok | 120 |
| **32** | **40** | **1.000** | **ok** | **129** |
| 48 | 56 | 1.000 | ok | 158 |
| 64 | 72 | 1.000 | ok | 192 |
| 100 | 108 | 1.000 | ok | 306 |
| 200 | 208 | 1.000 | ok | 507 |

(ms inflated by concurrent CPU load during the run; the **ratio** is the clean signal — fewer steps ∝ faster.)

## Reading it

- **The phase read needs ≥ ~32–40 steps, not 208.** Below ~32 the zero-crossing read collapses (0.000); at 32 it
  is already at full accuracy. So **~6× of the 208-step window was headroom**, not load-bearing — a free speedup.
- **It stacks multiplicatively** with the batched scan (CYCLE 152) and will stack again with the CUDA-graph
  refactor: each op is both cheaper (fewer steps) and there are fewer ops per turn (batched).
- **Honest adoption gate:** the sweep validated *who/what + moat*; the full stack (negation, clauses, multi-hop,
  reconsolidation, multi-turn) uses the same unbind+cleanup, so it should hold — but changing the **default**
  `period` (in `RFPhasorComposer` and the hardcoded `period=200` in `BrainConversationalAgent`) must be gated on
  re-running the full conversational suite at the shorter window (and the `test_rf_*` golden outputs assume 208,
  so those goldens move). Recommend `period=48` (a safe margin above the 32 cliff) for the adoption, pending that
  re-validation. Until then it is a validated knob, not the default.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_resonate_period_sweep --seeds 42,43,44
```
