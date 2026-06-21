# Shortcut #5b secondary SNc-burst δ (deferred-item-1) — synaptic-scaling volley-normalization (IN FLIGHT 2026-06-21)

**Task:** SURPASS the #5b secondary SNc-burst δ residual (deferred-item-1). Per the determinism close
(`2026-06-22-shortcut5b-determinism-deltabar-close.md`, `08d24a61`): #5b R1 (value-grading) is CLOSED 3/3;
the deterministic-read holds the SNc-burst δ on seeds 42/43 but NOT seed 44. The seed-44 residual is NOT a
non-determinism artifact (the scoping's premise was FALSIFIED — the 255.8 Hz rate is reproducible) — it is a
**GENUINELY STRONG learned place→value volley** (`w_near` grew to 2.475 on seed 44 vs 0.40/0.57 on seeds
42/43) over-driving the weighted-plateau READ → the critic fires hard even at FAR (136.5 Hz) → the SNc
GABA_B subtraction over-clamps at BOTH near and far → `gabab_gap=False`. A flat weight cap (0.8) FAILS — it
STARVES the gentle seeds 42/43 (critic → 0–1.4 Hz). The genuine fix is to NORMALIZE the seed-variable
learned-volley strength so ONE config holds the δ on all 3 seeds.

## The precisely-characterized residual (the determinism-close 2/3 baseline, flag ON)

| seed | `w_near` | crit@near Hz | crit@far Hz | `gabab_gap` (authoritative) |
|---|---|---|---|---|
| 42 | 0.401 | 17.1 | 0.0 | **True** |
| 43 | 0.570 | 64.0 | 0.0 | **True** |
| 44 | **2.475** | **255.8** | **136.5** | **False (over-clamp)** |

Seed 44's `w_near` is ~5× seeds 42/43. The smoking gun is **crit@far = 136.5 Hz** (vs 0.0 on 42/43): the
strong volley makes the critic fire even at FAR, so the SNc GABA_B subtraction clamps the SNc at far too →
`snc_unpred (FAR)` collapses to the same low level as `snc_pred (NEAR)` → the δ-gap (`snc_unpred > 1.3×
snc_pred`) vanishes.

## The mechanism (Turrigiano 2008 synaptic scaling — the EXISTING sim/ machinery, NO sim/ edit)

The fix is **synaptic scaling on the place→value path** (Turrigiano 2008): multiplicatively scale the
critic's afferent weights toward a common SET POINT. Because the scale factor is ONE value applied
uniformly to ALL the critic's place afferents, the **near/far RATIO** (the R1 selectivity, set by STDP) is
PRESERVED while the ABSOLUTE volley level is driven to one seed-STABLE operating point. The strong seed
(44, `w_near` 2.475) is scaled DOWN to the gentle-seed band; the gentle seeds (42/43) are left at their
already-passing level (the flat-cap STARVATION failure mode is avoided because the set point IS the gentle
band).

Three forms were tested (all on the EXISTING machinery, NO `sim/` edit):

1. **`continuous`** (stock `cfg.enable_synaptic_scaling`, `sim/bridge.py:7402`) — enable the per-step flag
   for the whole pipeline. **NEGATIVE:** it measures the VALUE-TRAIN firing (inflated by the
   `critic_teacher_pa=300` teacher) → over-suppresses the READ regime (seed 44: `w_near` 2.475 → 0.0,
   critic starves — the same failure as the flat cap). The teacher-driven regime is the wrong rate to
   normalize against.
2. **`freeze_seam` RATE-TARGET** — a read-regime synaptic-scaling calibration applied ONCE at the
   value-train→read FREEZE seam (the `value_input` 1.0→0.0 transition in `_patched_set_gate`); measures the
   WEIGHTED-plateau critic@near (the regime stage-B reads), scales toward a target rate. **PARTIAL:** the
   calibration lands its own read correctly (seed 44: 401.5 → 49.9 Hz, R1 V_n/f=6.71 preserved) but
   stage-B reads 273.9 Hz at the SAME near — the critic's OWN threshold homeostasis drifts during the
   rate-calibration and is not stable into stage-B (a `fs_freeze_critic_threshold` option pins it, untested
   — superseded by form 3).
3. **`freeze_seam` WEIGHT-TARGET** (the WINNER) — at the freeze, scale `w_near` to a TARGET WEIGHT (the
   gentle-seed band ~0.5–0.6) in ONE shot. A SET-POINT form of Turrigiano scaling: no rate measurement, so
   no interaction with the critic threshold homeostasis. **Seed 44 GO:** `w_near` 2.233 → 0.500 (uniform
   scale 0.224 over 102 near-active place cells) → crit@near 255.8 → 17.2 Hz, **crit@far 136.5 → 0.0 Hz**
   (far now SILENT = the over-clamp is gone), `snc_pred(NEAR)=0 / snc_unpred(FAR)=50` = the genuine RPE δ,
   `gabab_gap=True`, `V_n/f=6.82` (R1 PRESERVED — actually improved from 4.78 because de-saturating the
   over-driven critic cleans the graded read). Seed 44 now behaves exactly like the gentle seeds (42's
   17.1/0.0).

Probe lever (the WINNER): `--synaptic-scaling --synscale-mode freeze_seam --synscale-fs-target-wnear W
[--synscale-fs-down-only]` (a uniform multiplicative scale on `cp_connections.data` over the place→value
synapses, the same op `cfg.enable_synaptic_scaling` does per-step, applied once in the read regime). Held
WITH `--deterministic-read` throughout. NO `sim/` edit; NO `g11_bg_runner.py` edit.

## RESULTS — single-seed mechanism CONFIRMED (seed 44 GO); 3-seed battery IN FLIGHT

**Seed 44 (the make-or-break) — GO** (weight-target `w_near`→0.5):
- BEFORE (determinism alone): crit@near 255.8 / far 136.5 Hz, `gabab_gap=False`.
- AFTER (weight-target): crit@near **17.2** / far **0.0** Hz, `snc_unpred(FAR)=50` / `snc_pred(NEAR)=0`,
  `gabab_gap=True`, `LEARNS-V=True`, `V_n/f=6.82`.

<!-- FILL: the full 3-seed δ table (config-B target 0.6 down-only: 42/43 untouched, 44 normalized), the
controls, R1-unchanged, the VERDICT -->

## sim/-edit flag

<!-- FILL -->

## Moat confirmation

A nav-only probe (no conversational regions). The place/critic/SNc state
(`cp_connections` / `cp_firing_states` / `cp_conductance_g_graded_plateau` / `cp_conductance_g_gabab`) is
array-disjoint from the composer's complex `cp_rf_w_*` synapses. Preserved by construction; the no-confab
moat was NEVER weakened.

## Files
- `research/runners/_n5_grid_frontend_onbridge_probe.py` — `--synaptic-scaling` + the `_SYNSCALE` lever.
- `research/findings/raw/_n5_synscale_*_seed{42,43,44}.json` — the 3-seed battery.
- `sim/bridge.py:7402` + `sim/config.py:291` — the EXISTING synaptic-scaling path (UNCHANGED).
- `research/findings/2026-06-22-shortcut5b-determinism-deltabar-close.md` (`08d24a61`) — the 2/3 baseline +
  the precisely-characterized seed-44 residual this de-risk normalizes.

## Reproduce
```bash
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m research.runners._n5_grid_frontend_onbridge_probe \
    --seed 44 --all-arms --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 2.5 --value-train-w-max 3 --deterministic-read \
    --synaptic-scaling --synscale-target-rate <R> --synscale-ema-alpha <a> \
    --out research/findings/raw/_n5_synscale_seed44.json
```
