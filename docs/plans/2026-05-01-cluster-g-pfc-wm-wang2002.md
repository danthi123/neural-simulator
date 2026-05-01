# Cluster G v1 — PFC Working Memory (Wang 2002 NMDA-mediated bistability)

**Date:** 2026-05-01
**Status:** DESIGN
**Predecessor:** existing `dlpfc_wm` region in `build_bg_brain_regions` (cluster D-adjacent, single AMPA+GABA dynamics)

## Goal

Give the PFC region true **persistent activity** via NMDA-mediated recurrent excitation, per Wang 2002. Test on delayed-response: agent retains goal information across a "silence" window where goal_cells are zeroed.

## Biology source

Primary: **Wang 2002**, *J Neurosci* "Synaptic basis of cortical persistent activity: the importance of NMDA receptors to working memory."

Key claims:
- PFC pyramidal neurons sustain elevated firing (~20 Hz) for seconds during delay periods
- Bistability emerges from recurrent E-E excitation with a slow component
- AMPA alone (τ ≈ 5 ms) is too fast — recurrent excitation decays before next spike volley
- NMDA (τ ≈ 100 ms) is slow enough to bridge spike intervals → sustained drive
- NMDA:AMPA ratio elevated in PFC (~0.5) vs other cortical areas (~0.1)
- Voltage-dependent Mg²⁺ block keeps NMDA quiet at rest, opens at depolarization

Secondary: **Goldman-Rakic 1995** (cellular basis), **Funahashi 1989** (delayed response), **Compte 2000** (computational ring attractor with NMDA).

## Existing infrastructure

The bridge already supports NMDA:
- `cfg.enable_nmda: bool = False` (default off)
- `cfg.nmda_ratio: 0.4` (NMDA:AMPA conductance ratio)
- `cfg.nmda_tau_decay: 100.0 ms` ← Wang 2002 calibration
- `cfg.nmda_tau_rise: 3.0 ms`
- `cfg.nmda_mg_concentration: 1.0 mM`
- `fused_nmda_update_and_current()` GPU kernel with Jahr & Stevens 1990 Mg²⁺ block
- Bridge step 4133-4146: NMDA conductance update + current contribution

Currently: NMDA is global (all regions) when enabled. Wang 2002's biology suggests PFC-specific NMDA dominance, but for v1 we accept the global approximation.

## Implementation

### CLI flag

```python
ap.add_argument("--enable-pfc-nmda", action="store_true",
    help="Cluster G v1 (Wang 2002): enable NMDA-mediated recurrent "
         "excitation globally with PFC-typical 0.5 NMDA:AMPA ratio. "
         "Combined with --enable-pfc, gives the dlpfc_wm region true "
         "persistent activity for working-memory delays. Default off.")
```

### Kwarg + plumbing

- `enable_pfc_nmda: bool = False` kwarg to `run_moving_goal_episode`
- When True:
  - `cfg.enable_nmda = True`
  - `cfg.nmda_ratio = 0.5` (Wang 2002 PFC calibration; default 0.4 was a conservative starting point)
  - Other defaults unchanged

### Composes with

- `--enable-pfc` (must be on; otherwise no PFC region exists)
- All other clusters (orthogonal mechanism)

## Test plan

### Unit tests (already covered by existing NMDA tests if any)

Verify cfg.enable_nmda kwarg is forwarded; verify fused_nmda_update_and_current activates when on.

### Integration test 1: persistent activity emerges

Setup:
- `--enable-pfc --enable-pfc-nmda --heuristic-single-pool`
- Cheat-5 multi-goal det

Expected:
- dlpfc_wm region's mean firing rate during goal_silence window is non-zero (vs current behavior where it returns to baseline after goal removal)

Validation: instrument `bridge.cp_firing_states[dlpfc_wm_indices].sum()` during a goal_silence period, verify non-zero. Compare to `--enable-pfc` without NMDA (should return to baseline within ~50 ms).

### Integration test 2: delayed-response

Setup:
- `--enable-pfc --enable-pfc-nmda --heuristic-single-pool --goal-silence-after-step 1500 --goal-silence-duration 100`
- Cheat-5 multi-goal det, n=3 seeds

Expected:
- With PFC NMDA: agent retains goal info, phase-3 finalQ stays low (<2)
- Without PFC NMDA: agent drifts during silence, phase-3 finalQ degrades

### Integration test 3: cheat-5 baseline impact

Setup:
- `--enable-pfc --enable-pfc-nmda --heuristic-single-pool` vs no-NMDA control
- Cheat-5 multi-goal det, n=6 seeds

Expected outcomes:
- **Best case:** NMDA gives small additional improvement (~5-10% over A+E SP 5.02)
- **Likely case:** Neutral or marginal — without an explicit delayed-response task, persistent activity may not be expressed
- **Worst case:** Slight regression — NMDA might add cortical noise

## Stretch (defer to v2 if v1 lands cleanly)

- **Per-region NMDA:** add `BrainRegion.nmda_ratio_override` field. Set high in PFC, low elsewhere. Better biology, more code change.
- **Wang 2002 ring attractor:** explicit topographic recurrent connectivity (orientation-selective neurons in a ring) for spatial working memory
- **Compte 2000 generalization:** continuous attractor for multi-position WM (not just goal vs not-goal)

## Files to touch

- `research/runners/g11_bg_runner.py`: add CLI flag + kwarg + plumbing
- `tests/test_g11_bg_runner_flags.py`: add 1-2 flag-acceptance tests
- (optional) Test NMDA conductance trace during goal_silence: probe in same test file
- `CLAUDE.md`: propagate after eval lands

## Estimated effort

- ~30 LOC implementation
- ~30 min eval (3 conditions × 6 seeds, sequential 3-batch)
- Findings doc + propagation if positive: ~30 min

Total: ~1-2 hours.
