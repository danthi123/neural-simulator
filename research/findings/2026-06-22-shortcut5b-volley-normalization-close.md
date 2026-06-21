# Shortcut #5b secondary SNc-burst δ (deferred-item-1) — synaptic-scaling volley-normalization CLOSES the over-clamp 3/3, AND a rigorous anti-cheat isolates a deeper boundary: the grid δ READOUT is structurally-driven (2026-06-21)

## VERDICT (one line)

**The SPECIFIED #5b residual — seed 44's volley OVER-CLAMP — is CLOSED:** the freeze-seam synaptic-scaling
volley-normalization (`w_near`→0.5) holds the SNc-burst `gabab_gap` δ **3/3 under one config** (seed 44
crit@far 136.5→0.0), with **no starvation** (critics 15–21 Hz) and **R1 V_n/f preserved** (4.42 / 12.61 /
6.69). The three discriminating controls (render / no_learn / lesion) collapse 3/3; moat array-disjoint; NO
`sim/` edit. **BUT** a rigorous added anti-cheat (the magnitude-matched `shuffle_v` lesion + place-drive
normalization) isolates a DEEPER, PRE-EXISTING boundary: the grid-frontend δ **READOUT** (the graded
plateau) conflates the place code's structural near/far MAGNITUDE asymmetry with learned value — the δ
survives even when the learned weight ratio is destroyed (magnitude-matched `w_n/f`→1.0, δ still True), and
the cleanest decoupling (place-drive normalization) removes the structural asymmetry but COUPLES to (kills)
the value-learning. So the over-clamp fix is genuine, but the δ it restores is **structurally-influenced,
not a clean learned RPE** — a characterized substrate boundary (an honest negative), equally true of the
determinism-close 2/3 baseline. **Disposition:** the over-clamp deferred-item is CLOSED; the
"is-the-δ-purely-learned" question is the precisely-isolated deeper residual (the graded-plateau readout
limitation), recorded as the next frontier.

---


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

## RESULTS — the volley-normalization closes the over-clamp 3/3 (the deeper anti-cheat is in the control battery + NEXT MOVE below)

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
| **shuffle_v** mag-matched (clean lesion) | True (snc 0/50) | True (snc 0/50) | True (snc 0/50) | needs the LEARNED ratio | **does NOT collapse at flat `w_n/f` → δ is STRUCTURAL** |

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

**The CLEAN metric-lesion `shuffle_v`, run MAGNITUDE-MATCHED (permute the LEARNED place→value V THEN
normalize `w_near`→0.5 = grid's level, so the ONLY difference vs grid is the learned ratio), is DECISIVE:
the grid δ at `w_near`=0.5 is ENTIRELY graded-V STRUCTURAL, NOT the learned ratio.** The magnitude-matched
shuffle_v shows `gabab_gap=True` on ALL 3 seeds with `snc_pred/unpred = 0.0/50.0` (a CLEAN δ) DESPITE
`w_n/f` = 1.005 / 0.817 / 1.049 (flat — and INVERTED on seed 43, far>near weights!). So **the δ does NOT
require the learned near/far weight ratio at all.** The mechanism (the documented **graded-V structural
contamination**): the graded plateau reads the grid code's INTRINSIC structural near/far MAGNITUDE asymmetry
(the place drive fires more cells at near), amplified by the overall weight magnitude (`w_near`=0.5),
INDEPENDENT of learning. `no_learn` collapses only because its weights are at INIT (~0.2, too low to amplify
the asymmetry); raise the magnitude (value-train OR the normalization) and the structural δ appears
regardless of what was learned. **Consequence:** the over-clamp removal is genuine (seed 44 crit@far
136.5→0), but the δ that now passes is the structural graded-V gradient, not a learned RPE — and this was
equally true in the determinism-close 2/3 baseline (seeds 42/43 "passed" structurally; the over-clamp masked
it on seed 44). This is the precisely-isolated genuine residual: **the graded-plateau READ conflates the
structural place-drive magnitude asymmetry with learned value.**

### NEXT MOVE — place-drive divisive normalization (decouple the δ from the structural asymmetry)

Per the no-boundary rule, the next move: a per-location DIVISIVE (L1) normalization of the grid place drive
to a CONSTANT total (Carandini-Heeger; point-neuron, biology-grounded), applied EVERYWHERE `place_sensors`
is driven (self-org + value-train + reads). This removes the structural per-location magnitude asymmetry →
the only near/far V left is the LEARNED weight ratio. The decisive prediction: with normalization ON, the
GRID δ HOLDS (learned ratio) while the magnitude-matched `shuffle_v` COLLAPSES (no structural asymmetry
left). Probe lever `--normalize-place-drive`.

**Result — the normalization confirms the inseparability (a FUNDAMENTAL finding).** With place-drive
normalization ON (every location → constant total drive), **BOTH** the grid arm AND the magnitude-matched
`shuffle_v` COLLAPSE (`gabab_gap=False` 6/6), `crit@near`≈0, AND — the key — the grid arm's value-train did
NOT learn a near/far ratio (`w_n/f` = 0.97 / 1.00 / 0.84 ≈ flat). So on this substrate the learned near/far
V is **inseparable** from the structural magnitude asymmetry: the value-train learns the near/far V BY
amplifying the place code's intrinsic structural magnitude differences (near fires more cells); remove that
asymmetry and there is NOTHING for the value-train to grow from → flat `w_n/f` → no δ. **Disambiguation (rules out "weak drive"):** the `drive_scale`×2 run (`_n5_synscale_norm5_*`) makes the
critic fire HARD (122–227 Hz) under the normalized drive, yet the grid arm's value-train STILL gives
`w_n/f` = 0.99 / 1.00 / 0.97 (`learns_v=False`) on every seed → the flat `w_n/f` is the genuine
inseparability, NOT weak drive. **Nuance — there IS real learning in the winning config:** the
NON-normalized value-train DOES grow `w_n/f` from ~1.0 (no_learn) to 1.43–1.95 (grid); the learned ratio is
real. The boundary is in the graded-plateau **READOUT**, which reflects TOTAL magnitude asymmetry
(structural + learned), not the learned increment selectively — so the gap survives when only the structural
part remains, and the only way to remove the structural part (place-drive normalization) also removes the
asymmetry the value-train learns FROM.

## sim/-edit flag

**NONE.** The whole arc is probe-only, reuse-by-import: the volley-normalization is a uniform multiplicative
scale on `cp_connections.data` over the place→value synapses (the same op the EXISTING
`cfg.enable_synaptic_scaling` does per-step, applied once at the freeze, on the EXISTING
`cfg.deterministic_transpose_matvec` deterministic-read path); the place-drive normalization is a
per-location renorm of the grid sensory drive in the probe's `place_sensors` monkeypatch. `sim/bridge.py`
and `g11_bg_runner.py` are BYTE-UNCHANGED.

## Moat confirmation

A nav-only probe (no conversational regions). The place/critic/SNc state
(`cp_connections` / `cp_firing_states` / `cp_conductance_g_graded_plateau` / `cp_conductance_g_gabab`) is
array-disjoint from the composer's complex `cp_rf_w_*` synapses. Preserved by construction; the no-confab
moat was NEVER weakened.

## Files
- `research/runners/_n5_grid_frontend_onbridge_probe.py` — the levers: `--synaptic-scaling --synscale-mode
  freeze_seam --synscale-fs-target-wnear` (the volley-normalization WINNER), `--deterministic-read`
  (the determinism-close base), `--arm shuffle_v` / `--with-shuffle-v` (the clean metric-lesion),
  `--normalize-place-drive` (the structural-decoupling next move).
- `research/findings/raw/_n5_synscale_battery_w05_seed{42,43,44}.json` — the 5-arm battery (grid δ 3/3 +
  render/no_learn/lesion collapse 3/3 + scramble confound).
- `research/findings/raw/_n5_synscale_shufflev_magmatched_seed{42,43,44}.json` — the DECISIVE
  magnitude-matched clean lesion (δ survives flat `w_n/f` → structural).
- `research/findings/raw/_n5_synscale_norm{,5}_{grid,shufflev}_seed{42,43,44}.json` — the place-drive
  normalization + disambiguation (value-train can't learn on flat drive → inseparability confirmed).
- `sim/bridge.py:7402` + `sim/config.py:291,300` — the EXISTING synaptic-scaling +
  deterministic-transpose-matvec paths (UNCHANGED).
- `research/findings/2026-06-22-shortcut5b-determinism-deltabar-close.md` (`08d24a61`) — the 2/3 baseline +
  the precisely-characterized seed-44 over-clamp residual this de-risk closes (and the same graded-V
  structural contamination it flagged, now fully isolated).

## Disposition / next frontier
- **CLOSED:** the #5b deferred-item-1 over-clamp residual — volley-normalization holds the δ 3/3 (seed 44
  crit@far 136.5→0), no starvation, R1 preserved.
- **The precisely-isolated deeper residual (next frontier):** the grid-frontend δ READOUT (the graded
  plateau) conflates the place code's structural near/far magnitude asymmetry with learned value. A clean
  learned-only δ needs a READOUT that reads the LEARNED weight increment rather than total magnitude (e.g.,
  a baseline-subtracted / contrast-normalized value read, or a read that is invariant to the afferent's
  total drive). That is a READ-mechanism redesign, distinct from the volley-normalization that closed the
  over-clamp — recorded here as the next move.

## Reproduce
```bash
# The volley-normalization that CLOSES the over-clamp (δ 3/3, no starvation, R1 preserved):
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m research.runners._n5_grid_frontend_onbridge_probe \
    --seed 44 --all-arms --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 2.5 --value-train-w-max 3 --deterministic-read \
    --synaptic-scaling --synscale-mode freeze_seam --synscale-fs-target-wnear 0.5 \
    --out research/findings/raw/_n5_synscale_battery_w05_seed44.json
# (repeat --seed 42, 43; the δ holds 3/3 under this one config)

# The DECISIVE clean lesion (magnitude-matched flat-ratio → δ survives → structural):
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m research.runners._n5_grid_frontend_onbridge_probe \
    --seed 44 --arm shuffle_v --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 2.5 --value-train-w-max 3 --deterministic-read \
    --synaptic-scaling --synscale-mode freeze_seam --synscale-fs-target-wnear 0.5 \
    --out research/findings/raw/_n5_synscale_shufflev_magmatched_seed44.json

# The structural-decoupling next move (place-drive normalization → value-train can't learn on flat drive):
SIM_BACKEND=cupy CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m research.runners._n5_grid_frontend_onbridge_probe \
    --seed 44 --arm grid --readout-only --multi-goal --value-train-trials 40 \
    --grid-drive-scale 5.0 --value-train-w-max 3 --deterministic-read \
    --synaptic-scaling --synscale-mode freeze_seam --synscale-fs-target-wnear 0.5 \
    --normalize-place-drive \
    --out research/findings/raw/_n5_synscale_norm5_grid_seed44.json
```
