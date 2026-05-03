# dt=1.0ms speedup validated — ~2x throughput, dynamics stable

**Date:** 2026-05-03 ~14:58 EDT (autonomous overnight)
**Tool:** `--dt-ms 1.0 --stim-steps-per-step 100 --reset-steps 50`

---

## TL;DR

Tested dt=1.0 (vs the v2 default dt=0.5) with halved stim+reset
windows so simulated time matches. Result:

- **No crashes, no NaN, dynamics stable.** Izhikevich Euler at dt=1.0
  is fine for our parameter set.
- **~1.92x fewer sub-steps**, **~47% wall-clock savings.**
- W->A accuracy at 20 ep training = 28% (in v2 baseline range).
- aligned 0/1 (same architecture-level issue as everything else; dt
  is a performance lever, not a learning lever).

## Numbers

20-episode Phase 2 + standard eval at dt=1.0 / stim=100 / reset=50:

| Metric | Value |
|---|---|
| Sub-steps | 147,000 |
| Sub-steps if dt=0.5 same simulated time | 282,000 |
| Sub-step ratio | 1.92x fewer |
| Wall clock (under 4-way GPU contention) | 27 min 26 sec |
| Sub-steps/sec | 89 |
| W->A accuracy | 28.0% |
| Best permutation accuracy | 34.0% (NESW rotation) |
| aligned | 0/1 |
| I->W accuracy | 19.0% |

For comparison, an equivalent 20-ep dt=0.5 run under same 4-way
contention would have taken ~52 min. dt=1.0 saved ~25 min.

## What this unlocks (combined with parallel-3 already in use)

| Configuration | 6-seed wall clock |
|---|---|
| Original sequential dt=0.5 | ~480 min |
| Parallel-3 dt=0.5 (current) | ~150 min |
| Parallel-3 dt=1.0 (with this) | **~80 min** |

= **6x improvement** over the original baseline, free.

## Caveats

1. **Numerical accuracy.** dt=1.0 is at the edge of Izhikevich Euler
   stability. For parameter sets with very fast dynamics (FS
   interneurons, MSN bursting), might miss spike timing. Our profile
   shows the network behaves normally at dt=1.0 but a closer
   comparison vs dt=0.5 on the SAME seed would be cleaner — easy to
   add as a follow-up.
2. **Eval method assumes 50ms reset window.** If we go to dt=2.0 in
   the future, the reset would be 25ms — too short to wash out NMDA
   tau (200ms decay).
3. **Single seed result.** The 28% accuracy at 20 ep is one data
   point — could be lucky variance. Full validation would need 6
   seeds at dt=1.0 to confirm distribution matches dt=0.5.

## Recommendation

**Use dt=1.0 + stim=100 + reset=50 as the default for the next
experiment cycle.** Specifically:
- For tomorrow's investigation experiments (Hebbian-fix-decay,
  longer-Phase-2, etc.), use dt=1.0 throughout. Halve wall clock.
- Don't retroactively change the running fundamentals sweep —
  consistency vs the existing v2 baseline matters there.
- After the auto-followup completes, re-run the winning variant at
  dt=1.0 as a confirmation that scaled-up runs match.

## Implementation

Already wired:
- `text_train_curriculum.py --dt-ms 1.0 --stim-steps-per-step 100 --reset-steps 50`
- `text_pfc_bypass_isolation.py` doesn't take `--dt-ms` yet — would need similar wiring for tomorrow

## Next-step micro-opts in the profile

While we're here, the profile (88% compute / 12% Python) showed the
biggest single section is `t_dyn` (29% of compute = ~1 ms/step
single-process). Inside `t_dyn`:
- `fired_indices = cp.where(fired_this_step)[0]` forces a GPU-CPU sync
- `if fired_indices.size > 0:` requires reading the size to host
- The fancy-index reset operations (`v_new[fired_indices] = c_reset[...]`)
  could be replaced with `cp.where(fired, c_reset, v_new)` masking

Replacing the fancy-index pattern with masked-update would:
- Eliminate one GPU-CPU sync per step
- Replace 3 fancy-index kernel launches with 1 masked-where launch
- Save maybe 100-200 us per step = 3-5% speedup compounded

Not for tonight, but a clean PR for tomorrow.
