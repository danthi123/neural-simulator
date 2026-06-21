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

`cfg.enable_synaptic_scaling` (`sim/bridge.py:7402`) multiplicatively scales each postsynaptic neuron's
excitatory afferent weights toward a TARGET firing rate
(`scale_factor = 1 + synaptic_scaling_rate × (homeostasis_target_rate − activity_ema)`, clipped 0.95–1.05
per step). The critic (`striosome_value`) is the postsynaptic neuron of the place→value path. Because the
scale factor is ONE value per postsynaptic neuron applied to ALL its afferents, the **near/far RATIO** (the
R1 selectivity, set by STDP) is PRESERVED while the ABSOLUTE volley level is driven to one seed-STABLE
operating point. The strong seed (44) is scaled DOWN to the gentle-seed regime; the gentle seeds (42/43) are
NOT starved (the flat-cap failure mode) because scaling targets a RATE, not a fixed ceiling. Point-neuron,
biology-grounded, held ON through the value-train + the reads (the volley settles to target and stays:
scale_factor → 1 at target → no read-time drift).

Probe lever: `--synaptic-scaling --synscale-target-rate R [--synscale-rate r] [--synscale-ema-alpha a]`
(set in `_n5_grid_frontend_onbridge_probe.py:_patched_init` via the existing `cfg.enable_synaptic_scaling`
path). Held WITH `--deterministic-read` throughout. NO `sim/` edit; NO `g11_bg_runner.py` edit.

## RESULTS — IN FLIGHT

<!-- FILL: the 3-seed δ table (all 3 + no starvation), the controls, R1-unchanged, the VERDICT -->

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
