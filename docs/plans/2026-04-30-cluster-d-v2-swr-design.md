# Cluster D v2 — Sharp-Wave-Ripple replay for offline CA3 cleanup

**Date:** 2026-04-30
**Status:** DESIGN (not yet implemented)
**Branch context:** `main`, builds on Cluster D v1 (`--enable-cluster-d-hippocampus`, scaffolded 2026-04-29)

## Why now

After F v2 closed NO-GO (3× worse than baseline due to PF-scale mismatch), the cluster-stacking strategy has 4 NEUTRAL/NEGATIVE results in a row past A+E (6.97 ± 0.83 multi-goal det). D v2 is the last *per-step plasticity* candidate before pivoting to compartmentalized DA (Cluster C v2) or scaling work.

D v2 differs from F v2 in mechanism. F v2 added a *new* anti-Hebbian rule globally on every step. D v2 modifies *when* existing plasticity fires inside the hippocampus during sleep — a gating change, not a new rule. That gives it independent failure modes from the F branch.

## The problem D v2 should solve

The existing sleep replay (Cluster D v1, NREM trajectory + REM random) trains downstream weights from arbitrary trajectory snippets. The SCIENCE_ROADMAP note says content quality is the bottleneck: random trajectory replay doesn't help, recency-weighted current-goal-only might.

But "content" is one lever. The other is **the gating of plasticity itself**: in real hippocampus, replay-related learning happens only during sharp-wave-ripple (SWR) bursts, not continuously. Outside SWR events, CA3 recurrent collaterals fire at low rate and STDP is suppressed by ACh tone. Inside SWR bursts (~50–100 ms windows of synchronous high-rate firing), STDP fires with high gain.

The mechanistic claim: **STDP gated by SWR events selectively reinforces patterns that the recurrent autoassociator already partially learned, while suppressing reinforcement of random/noise correlations**. This is the autoassociator "cleanup" property — strong attractor basins get deeper, shallow ones decay. Without burst-gated plasticity, every co-firing event reinforces noise.

## Design — three options, recommendation

### Option 1: Burst-gated CA3 plasticity (recommended)

- Add a `_detect_ca3_burst()` probe that runs every step during sleep. Burst = CA3 mean firing rate > μ + 2σ where μ, σ are running stats over the last 200 ms.
- Add a new plasticity gate `ca3_swr_burst` that gates the CA3 internal recurrent + CA3→CA1 Schaffer + CA1→cortex pathways. Default 0.1 (suppressed) during sleep; thaw to 1.0 during burst windows; back to 0.1 between bursts.
- During wake the gate is held at 1.0 (D v1 behavior preserved).

Implementation footprint: ~50 LOC in `g11_bg_runner.py`, ~30 LOC of unit tests. No new bridge fields. Reuses the existing `set_plasticity_gate()` infrastructure.

Risk: if the existing sleep replay drive doesn't actually elicit bursts in CA3, no consolidation happens. Mitigation: smoke test asserts at least one burst per 50 sleep steps; otherwise increase the replay drive amplitude.

### Option 2: Reverse-order trajectory replay (content-side)

- Modify the NREM replay sampling to drive `(x_t, y_t, gx, gy)` in *reverse* time order (goal → start). Real hippocampus does this for credit assignment; trajectories near the goal get more dopamine in our scheme via the reverse-time ordering.
- No new gates, no new mechanisms. Just changes which trajectory is sampled.

Implementation footprint: ~30 LOC. Risk: addresses content quality but not the gating problem. May not interact constructively with Option 1.

### Option 3: Phase-locked slow-oscillation + burst detector (full original design)

- Add a 1 Hz slow oscillation via global excitability modulation during sleep.
- Detect CA3 bursts that are phase-locked to up-state troughs.
- Combined gating: plasticity opens only at burst × up-state coincidences.

Implementation footprint: ~150 LOC + 2 hyperparameters (slow-osc amplitude, burst threshold). Most biologically faithful but largest tuning surface.

### Recommended: Option 1 standalone first

Reasons:
1. **Smallest scope**, easiest TDD coverage. ~50 LOC + tests is one focused change.
2. **Independent failure mode** from existing sleep-replay content. If Option 1 lands neutral or negative, Option 2 is a clean follow-up because the two changes don't interact.
3. **Mechanism-level claim is clean**: gate STDP on autoassociator burst events. Falsifiable in 6-seed eval.
4. Option 3's slow-osc is biologically nicer but adds tuning we can't justify until Option 1 has signal.

If Option 1 is NEUTRAL or NEGATIVE on 6-seed multi-goal det, **stop**. We've now exhausted 5 cluster attempts past A+E and the pattern is real — buildout strategy isn't beating the ceiling. Pivot to Cluster C v2 (compartmentalized DA) or scale work.

## Implementation plan (Option 1)

### Code changes in `research/runners/g11_bg_runner.py`

1. **New CLI flag**: `--enable-cluster-d-v2-swr` (mutually compatible with `--enable-cluster-d-hippocampus`; warns if v2 is set without v1).
2. **New plasticity gate** `ca3_swr_burst` registered on three pathways when v2 enabled:
   - CA3 internal (region.plastic_internal continues being True; gate added via plasticity_gate field)
   - CA3 → CA1 (Schaffer)
   - CA1 → place_cells (currently static; consider plasticity if v2 needs it — design decision, see below)
3. **Burst detector** function called once per step inside the sleep_replay block:
   ```python
   def _ca3_burst_active(bridge, ca3_indices, history) -> bool:
       rate = bridge.recent_firing_rate(ca3_indices)  # exists in bridge
       history.append(rate)
       if len(history) > 40:  # ~200ms at dt=5ms
           history.popleft()
       if len(history) < 10:
           return False
       mu = np.mean(history)
       sd = np.std(history) + 1e-6
       return rate > mu + 2.0 * sd
   ```
4. **Gate flip**: at sleep step start, gate = 0.1; if `_ca3_burst_active()` returns True, set gate = 1.0; otherwise stay at 0.1.

### Tests (TDD order — all must fail first)

| # | Test | Check |
|---|------|------|
| 1 | `test_burst_detector_no_burst_low_rate` | flat 5 Hz CA3 firing → False all steps |
| 2 | `test_burst_detector_detects_2sigma_spike` | inject burst at step 30 → True at step 30, False before/after |
| 3 | `test_v2_flag_creates_swr_gate` | CLI `--enable-cluster-d-v2-swr` registers `ca3_swr_burst` gate |
| 4 | `test_swr_gate_default_low_in_sleep` | first sleep step before any burst → CA3 plasticity gain ≈ 0.1 |
| 5 | `test_swr_gate_thaws_during_burst` | force CA3 burst → CA3 plasticity gain == 1.0 that step |
| 6 | `test_swr_gate_unchanged_during_wake` | wake step (not in sleep) → gate stays at 1.0 even with no burst |
| 7 | Smoke test (manual): `--enable-cluster-d-v2-swr --n-steps 1800 --seed 42 --sleep-replay-after-step 900 --sleep-replay-steps 300` runs to completion, returncode 0 |

### Eval plan

| Tier | Conditions | Seeds | Wall-clock | Decision |
|---|---|---|---|---|
| Smoke | A+D+v2 single seed n_steps=400 | 1 | 5 min | runs cleanly + at least 3 burst events fire |
| Tier 2 | A+D vs A+D+v2 vs A+E+D+v2 | 3 | 30 min via replicated runner | go/no-go for tier 3 |
| Tier 3 | Above 3 conditions, 6 seeds | 6 | 60 min via replicated runner | NO-GO if mean ≥ baseline; GO only if Δmean ≤ −1.0 vs A+E (6.97) |
| Tier 4 | If tier 3 GO, deterministic 6-seed | 6 | 90 min | confirm; commit as new flagship |

Decision threshold matches Cluster A and the cheat-5 strategy doc: Δmean ≤ −1.0 *and* Δstd ≤ baseline std. Anything else is NO-GO.

### Open question (decide during implementation)

Should CA1 → place_cells become plastic under v2 (currently static, line 1234)? Two arguments:

- **Yes**: SWR consolidation should propagate to readout. Keeping it static breaks the consolidation chain at the last step.
- **No**: Place cells are coordinate readout; making them plastic risks drifting the agent's spatial representation during sleep replay.

Default: keep static for v2 implementation, revisit if signal is partial. Documented as a follow-up question.

## What success looks like

- 6-seed multi-goal det A+D+v2 mean ≤ 5.97 (≥1.0 Δ vs A+E ceiling 6.97)
- Std ≤ 0.83 (A+E baseline std) — variance NOT increased
- 6/6 seeds beat baseline 7.77; ≥4/6 beat A+E 6.97

## What partial success looks like

- A+D+v2 NEUTRAL on cheat-5 mean (within ±1 of A+E) but reduces phase-2 collapse seeds (the high-variance seeds in baseline like seed 101 = 12.70)
- → reasonable to keep `--enable-cluster-d-v2-swr` opt-in for variance control, like F v1 was kept
- → also worth following up with Option 2 (reverse-order replay) as content companion

## What failure looks like

- A+D+v2 mean ≥ A+E mean (no improvement) AND no variance reduction
- → close NO-GO, do not stack v2 on flagship
- → triggers the pivot decision: Cluster C v2 next, or scale work

## Files this will touch

- `research/runners/g11_bg_runner.py`: add CLI flag, burst detector, gate management in sleep loop
- `tests/test_cluster_d.py` (new): 6 unit tests
- `tests/test_g11_bg_runner_flags.py`: add CLI flag wiring test
- `research/findings/2026-04-30-cluster-d-v2-results.md` (new, post-eval): findings doc
- `CLAUDE.md`: cluster status propagation post-eval

## Out of scope for v2

- Slow-oscillation phase-locking (Option 3 territory)
- Engram tagging API (deferred to v3 in original design doc)
- Reverse-order trajectory replay (Option 2 — separate follow-up if v2 is partial-signal)
- Multi-goal-conditioned replay buffers
- Hippo-PFC interaction (separate cluster, not D)
