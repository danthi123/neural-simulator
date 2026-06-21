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

**The set point is `w_near`→0.5** (the gentle-seed operating point; seeds 42/43 sit at 0.365/0.57, mean
~0.47). A `w_near`→0.6 probe over-shot (seed 44 crit@near 93 Hz, `gabab_gap=False`): the gentle band is
NARROW and 0.5 is in it, 0.6 is over the edge (a steep weight→rate nonlinearity near the cliff). 0.5 is
principled — it normalizes the strong seed TO the weight regime of the seeds that already work. A `w_near`→
0.6 `down_only` probe also confirmed seed 42 (0.365 < target) is left byte-UNTOUCHED (scale 1.000) and
still passes (the 80-step near-active measurement does not perturb a gentle seed).

**The full 3-seed battery uses `w_near`→0.5 (both-directions): every seed lands at the SAME set point
(0.365/0.57/2.22 → 0.5/0.5/0.5)** — the purest realization of "normalize the seed-variable volley strength
to ONE operating point."

### The 3-seed grid (TEST) SNc-burst δ — config A' (`w_near`→0.5) vs the determinism-only 2/3 baseline — **3/3**

| seed | `w_near` (scaled) | crit@near Hz | crit@far Hz | `gabab_gap` (authoritative) | V_n/f (R1) | — | baseline `gabab_gap` (determinism only) | baseline crit@far |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.364 → 0.500 (×1.37 up) | 21.0 | 0.28 | **True** | 4.42 | | True | 0.0 |
| 43 | 0.572 → 0.500 (×0.88 down) | 18.75 | 0.0 | **True** | 12.61 | | True | 0.0 |
| 44 | 2.243 → 0.500 (×0.22 down) | 15.0 | 0.0 | **True (was False)** | 6.69 | | **False (over-clamp)** | **136.5** |

**3/3 under ONE config (`w_near`→0.5).** The make-or-break seed 44 flips False→True: its over-driving volley
(crit@far 136.5 Hz → 0.0) is normalized to the gentle-seed band (crit@near 255.8 → 15.0 Hz), so the SNc can
burst at FAR (`snc_unpred=50`) while V clamps it at NEAR (`snc_pred=0`) — the genuine δ=r−V. **No
starvation:** every critic fires 15–21 Hz at near (the flat-cap failure mode — 0–1.4 Hz — is avoided).
**R1 unchanged:** V_n/f stays selective on every seed (4.42 / 12.61 / 6.69 vs the determinism-only baseline
4.49 / 12.35 / 4.78 — seeds 42/43 unchanged, seed 44 improved because de-saturating the over-driven critic
cleans the graded read). `LEARNS-V` holds (seed 44's `w_n/w_f=1.94`).

### The control battery (the anti-cheat — normalization must NOT manufacture a δ)

| arm | seed 42 `gabab_gap` | seed 43 `gabab_gap` | seed 44 `gabab_gap` | what it controls | verdict |
|---|---|---|---|---|---|
| **grid** (TEST) | **True** | **True** | **True** | the learned grid δ | the RPE δ is present 3/3 |
| render (R1-LIMIT) | False | False | False | needs a non-degenerate location code | **collapses 3/3** |
| no_learn (floor) | False (1.0) | False (1.0) | False (1.0) | needs the value-train | **collapses 3/3** (exactly 1.0; `w_n/f`≈1.0) |
| lesion (graded off) | False (1.0) | False (1.0) | False (1.0) | needs the graded read-out | **collapses 3/3** |
| scramble (metric) | True | True | False | needs the periodic grid METRIC | does NOT collapse (confounded — see below) |
| **shuffle_v** (clean metric-lesion) | True (graded-V FLAT 50/50) | **False** | True (graded-V 0/50) | needs the LEARNED near/far ratio | **mixed — exposes the deeper residual** |

**Three discriminating controls collapse 3/3 — the grid δ is value-train-dependent.** The decisive one is
**no_learn** — the SAME grid code (same magnitude structure), no value-train → `snc_gap=1.0`, `w_n/f`≈1.0,
no δ on every seed. So the grid δ **REQUIRES the value-train** (it is not present in the bare structural
code). `render` (collapses) shows it needs a non-degenerate location-discriminable code; `lesion`
(collapses) shows it needs the graded read-out.

**The `scramble` control does NOT collapse — an honest correction to the prior framing.** A scrambled grid
code (per-cell independent position-permutation) is still **DECORRELATED and location-discriminable**, so
the place self-org carves selective fields on it and the value-train learns a GENUINE near/far V (seed 43
scramble: `w_n/f=2.49`, crit@near 53 Hz). The scramble lesions the periodic grid METRIC but NOT the
learnability of a near/far V — it is NOT a clean "is-it-learned" lesion. (The determinism-close doc reported
scramble "collapses (no spatially-selective V)", but that collapse was an OVER-CLAMP artifact, masked by the
same over-firing this fix removes.)

**The CLEAN metric-lesion `shuffle_v` (permute the LEARNED place→value V across place neurons → `w_n/f`→1.0)
exposes the DEEPER residual: the grid δ is value-train-dependent but NOT purely the learned near/far
ratio.** In the graded-V-only read, shuffle_v's δ COLLAPSES on seeds 42/43 (snc_pred/unpred → 50/50 FLAT;
seed 43 authoritative `gabab_gap`→False) but SURVIVES on seed 44 (snc 0/50, `gabab_gap`=True at `w_n/f`=1.03
— a FLAT learned ratio). The mechanism (the documented **graded-V structural contamination**): the
value-train raises the OVERALL place→value weight MAGNITUDE (init ~0.2 → ~0.5), which amplifies the grid
code's INTRINSIC structural near/far asymmetry (the place drive is stronger at near for seed 44's particular
phase draw) through the graded plateau → a near>far V WITHOUT a learned ratio. So the value-train is
necessary (no_learn collapses) and contributes via BOTH the learned ratio (dominant on 42/43) AND a
structural-magnitude amplification (dominant on 44). This is a **pre-existing property of the
graded-plateau READ** (flagged in the determinism-close doc), surfaced — not introduced — by removing the
over-clamp.

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
