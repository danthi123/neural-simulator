# #5 place-code SPARSIFY — sparsification FIXES value-learning but a read-out-regime selectivity wall caps the δ (honest BOUNDARY, 2026-06-19)

**Follow-on to** `2026-06-18-merged-neural-place-code-delta-probe-NEGATIVE.md` (the dense-field root cause) +
`2026-06-18-merged-neural-place-code-SCOPE-GO.md` (the `nav_critic_place_selforg` builder path). The NEGATIVE
found the self-org place fields were **too DENSE (sparsity 0.46) + overlapping (cos near/far 0.67)** → the
value-train potentiated overlapping cells uniformly → uniform V → flat δ (1.0, WORSE than the host Gaussian's
~1.3). This task: sharpen the self-org → sparse, separated fields → position-graded value → δ > 1.3 → deploy
as the merged default (replacing the host-Gaussian `vs_place_context` shortcut).

## VERDICT — BOUNDARY (honest NEGATIVE on the δ-lift; a substantially more informative one than the prior NEGATIVE)

**Sparsification WORKS and FIXES the prior NEGATIVE's root cause** (LEARNS-V `w_near/w_far` improved from the
NEGATIVE's **1.01 → 1.91x** at `place_sensors_to_place_weight=10`), **but the δ does NOT cross 1.3** — a DEEPER,
distinct wall is exposed: in the **FS-PING-OPEN operating regime that the value-train and the critic read in**,
the sparse cells are **NOT location-selective** at nav scale (a few dominant place cells fire at MANY locations,
so the near/far ensemble overlap is `cos ≈ 0.42-0.78` REGARDLESS of self-org sparsity), and the all-or-none
weighted coincidence-plateau read-out has only two reachable regimes, neither of which grades:
- **low weights** (sparse code → ~15 cells → `w_near ≈ 2.4-2.7`): the critic fires in a *physiological* band
  (~5 Hz) but **cannot discriminate** (critic@near ≈ critic@far, ratio 0.87-0.99) → flat δ ≈ 1.04-1.12.
- **high weights** (more cells / more trials → `w_near ≈ 40-86`): the critic **over-fires** (~98-238 Hz,
  unphysiological) → **over-clamps the SNc GABA_B to 0 for BOTH near and far** → δ collapses to 0.0.

**⇒ Do NOT flip the merged default critic afferent to the self-org place code.** The host-Gaussian
`vs_place_context` (position-specific BY CONSTRUCTION) remains the documented better-δ scaffold (δ ~1.3), exactly
the prior NEGATIVE's recommendation. The self-org place is a validated-but-costly brain-based replacement (it
COMPOSES — SCOPE-GO value `a`, committed — and now LEARNS a real V gradient, but its read-out δ underperforms the
host shortcut). Per the BRAIN-BASED-ONLY standard, this neural-underperforms-host mapping IS the deliverable.

## Iteration harness

`research/runners/_n5_place_sparsify_probe.py` — captures the EXACT deployed NEGATIVE-repro kwargs (intercepts
`run_moving_goal_episode` inside `main()`), applies sparsity-lever overrides, runs the standalone g11
`stage_a_smoke` (~15s self-org + ~10s read = sparsity+cos) or `stage_b_smoke` (~50-190s value-train + δ probe).
`--deterministic-selforg` ON → the CuPy-non-deterministic place self-org draws the SAME code per seed
(reproducible measurement; STEP-1 numbers byte-match the NEGATIVE at the default `place_sensors_to_place_weight=28`).

## STEP-1 root cause CONFIRMED + the dominant sparsity lever = the afferent threshold-WTA (a pure kwarg)

Baseline (NEGATIVE-repro, `place_sensors_to_place_weight=28`): STEP-1 sparsity **0.458**, diff-loc cos **0.672**.
The dense code is set by the afferent drive (`place_sensors_to_place_weight=28` × density 0.5) overdriving ~46%
of the `place` pool past threshold; two locations then share ~46% of cells by density alone → cos 0.67.

**Afferent-weight sweep (stage-A, seed 42, GPU, deterministic self-org):**

| `place_sensors_to_place_weight` | STEP-1 sparsity | STEP-1 diff-loc cos | stage-A read sparsity | stage-A read cos |
|---|---|---|---|---|
| 28 (baseline) | 0.458 | 0.672 | 0.587 | 0.718 |
| 18 | 0.385 | 0.612 | 0.473 | 0.739 |
| 12 | 0.168 | 0.569 | 0.200 | 0.770 |
| 11 | 0.110 | 0.559 | 0.117 | 0.619 |
| **10** | **0.0625** | **0.219** | **0.078** | 0.571 |
| 9 | 0.050 | 0.231 | 0.038 | 0.401 |
| 8 | 0.0075 | 0.000 (knife-edge: near-empty) | 0.007 | 0.000 |

- Lowering the afferent weight sharply sparsifies: **W=10 hits the sparsity target (0.0625 ≈ 6%)**.
- W=8 is a knife-edge (~1.5 cells, ensembles near-empty/disjoint) → too few cells to fire the coincidence
  detector (needs ≥K coincident spikes); the volley dies. W=9/10 keep ~10-15 active cells.

## The decisive finding: the FS-PING-OPEN read regime is NOT location-selective (a regime mismatch)

Two FS-open reads of the SAME frozen W=10 code disagree: the STEP-1 cos (n_meas=80, fresh) = **0.22**, but the
settled stage-A read = **0.57**, and the value-train (settled FS-open regime) operates at the ~0.57 overlap. The
sparse cells that survive the FS-PING competition fire at near AND far → the value-train potentiates overlapping
cells. **Confirmed across every regime lever**, none of which lowered the operative read cos below ~0.42:
- self-org WITH FS-PING open (matched regime, `N5_SPARSIFY_FS_DURING_SELFORG=1`, runner-local env-gated default-off
  edit): sparsity 0.06-0.10 but read cos 0.74-0.78 (no improvement; the FS-PING is a gamma synchronizer, not a WTA).
- stronger FS→place GABA_A during read (`fs_to_place_weight` 16/24/40): read cos 0.42/0.52/0.74 (worse at 40).

## STAGE-B δ sweep (seed 42, GPU) — every config either can't discriminate or over-clamps

| config | w_near/w_far | LEARNS-V | critic@near/far (Hz) | GABA_B δ gap | verdict |
|---|---|---|---|---|---|
| NEGATIVE baseline (W=28, k=12) | 1.01 | FALSE | 364 / 378 | 1.00 | dense → uniform V |
| W=10, multi-goal, k=12 | **1.91** | **TRUE** | 4.58 / 4.17 | 1.04 | sep V, but critic can't discriminate |
| W=10, multi-goal, k=4 | 1.91 | TRUE | 12.6 / 12.8 | 1.12 | k too low → both fire → flat |
| W=10, init=0.5, k=12 | 1.74 | TRUE | 5.28 / 5.56 | 1.04 | weights too small for plateau |
| W=10, init=0.3, k=8 | 1.86 | TRUE | 7.64 / 8.75 | 0.88 | far(=goal) fires more |
| W=11, single-goal, 80tr, init=0.5, k=20 | 1.68 | TRUE | 98 / 81 | 0.00 | OVER-CLAMP (critic 98 Hz → SNc=0) |
| W=11, single-goal, 80tr, fs_w=40, k=20 | 1.06 | FALSE | 238 / 189 | 0.87 | over-clamp + de-separated |
| W=10, single-goal, 150tr, GIRK cap=10, k=15 | **0.79** | FALSE | 7.50 / 6.11 | 1.04 | read-overlap: far cells potentiate during near-train |
| W=10, multi-goal, GIRK cap=12, k=6 | **1.91** | TRUE | 10.4 / 11.7 (physiological!) | **0.95** | best-effort: critic IN BAND but far(=goal) fires more → flat |

**Key cross-config diagnostics:**
- `near/far = far(1,1)` is the point-reflection of `near(6,6)` — but `(1,1)=sw` IS a scheduled goal, so in the
  multi-goal value-train far is ALSO valued (a probe-contrast artifact). The single-goal runs (only (6,6) trained,
  far untrained) are the clean capability test — and there **w_far GREW ABOVE w_near (LEARNS-V 0.79, FALSE)**
  because the FS-open read regime fires the same dominant cells at near AND far, so the (6,6) pairing potentiates
  far's "exclusive" cells too. ⇒ the value-train cannot localize V to the trained location's cells at nav scale.
- The over-clamp (W=11 high-weight) is unmitigated by the `place_fs→striosome_value` divisive clamp
  (`critic_fs_weight` 16→40 made it WORSE) AND by the GIRK saturation cap (`critic_gabab_max`) within the tested
  range — the two reachable regimes (under-discriminating vs over-clamping) bracket the narrow physiological+graded
  window without opening it.

## Levers tested (all on seed 42, GPU, deterministic self-org) — exhaustive

afferent weight `place_sensors_to_place_weight` {8,9,10,11,12,18,28} · `fs_to_place_weight` {8,16,20,24,40} ·
`N5_SPARSIFY_FS_DURING_SELFORG` {0,1} · `vs_place_to_value_weight` (init V) {0.2,0.3,0.5} · `value_train_trials`
{40,60,80,150} · `critic_warmup_all_goals` {True (multi), False (single)} · `coincidence_threshold` k
{4,6,8,12,15,20} · `critic_gabab_max` (GIRK cap) {0,10,12} · `critic_fs_weight` {16,40}.

## Is it the dendrite wall or a tuning limit?

**Closest to a tuning/architecture limit with a dendritic flavor.** The IMMEDIATE blocker is the FS-PING-open
read regime's NON-selectivity (a few dominant cells fire everywhere) + the all-or-none coincidence-plateau
read-out (binary, not graded), which over-clamps the SNc when driven hard. The DEEPER cause — the point-neuron
`place` pool cannot form MANY distinct, location-selective sparse codes from heavily-overlapping egocentric
landmark sensors at nav scale — is the same analog/dendritic-computation limit the project repeatedly hits
(Mikulasch-Priesemann; the conversational decorrelation/whitening wall). A genuinely sparse, location-selective
place code (real place cells ~1-5% AND selective) would plausibly need the dendritic substrate (per-cell
nonlinear input integration to carve selective fields), OR a fundamentally different read-out than the all-or-none
plateau (a graded rate read-out that scales smoothly with V, so a modest near>far weight gradient → a modest
near>far critic rate → a graded GABA_B δ, without the over-clamp). Both are out of this task's runner/op-point
scope and are the specified next moves if the δ-lift is re-prioritized.

## Deployment decision

`nav_conv_merged_bridge.py` is **UNCHANGED** (byte-identical; I own it for this task and leave it clean — the
self-org place stays the SCOPE-GO opt-in `nav_critic_place_selforg`, NOT the default critic afferent, because the
GO bar δ>1.3 is not met). The host-Gaussian `vs_place_context` remains the merged critic's better-δ scaffold.
The merged BUILD moat structure was reconfirmed (numpy CPU): `place` + `striosome_value` present, host
`vs_place_context` ABSENT under `nav_critic_place_selforg=True`, and `place`/`striosome_value` array-DISJOINT from
`parse_role`/`dlpfc_wm` (the no-confab moat's conv slices) — so a future sparse-config deployment would inherit the
disjointness, IF a read-out that grades is found.

## Files
- `research/runners/_n5_place_sparsify_probe.py` — the iteration harness (NEW).
- `research/runners/g11_bg_runner.py` — one runner-local, env-gated, default-OFF addition
  (`N5_SPARSIFY_FS_DURING_SELFORG`, the FS-during-self-org sparsify lever; byte-identical when unset). Tested
  NEGATIVE (it densifies / does not improve read-regime selectivity), kept as the documented research lever.
- `research/runners/nav_conv_merged_bridge.py` — UNCHANGED (deployment not made; GO bar not met).
- `research/findings/raw/_n5_sparsify_W10_best.json` — the best-effort δ probe dump.

## Reproduce (the winning sparsity recipe + the best-effort δ)
```bash
# sparse, separated fields (the value-learning fix): place_sensors_to_place_weight=10
SIM_BACKEND=cupy python -m research.runners._n5_place_sparsify_probe --seed 42 \
  --overrides place_sensors_to_place_weight=10          # stage-A: sparsity 0.0625, STEP-1 cos 0.22, LEARNS-V 1.91x
SIM_BACKEND=cupy python -m research.runners._n5_place_sparsify_probe --seed 42 --stage-b \
  --overrides place_sensors_to_place_weight=10,coincidence_threshold=12   # δ probe (flat ~1.04)
```
