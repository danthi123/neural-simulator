---
type: finding
status: qualified
date: 2026-08-07
mechanism: gateB-stage2k-novelty-gated-commit-WTA-release-fixD-plus-fixC
backend: numpy
runner: research/runners/_vocal_gateb_stage2k_exploration_release.py
builds-on: 2026-08-07-gateB-stage2j-adaptive-RPE-floor-smoke-and-MSN-k-homeostat.md
artifacts:
  - research/findings/raw/gateb_stage2k_exploration_release/smoke_730705_numpy.json
  - research/findings/raw/gateb_stage2k_exploration_release/seeds_dev_601_606_numpy.json
  - research/findings/raw/gateb_stage2k_exploration_release/diag_cascade_730705.txt
  - research/findings/raw/gateb_stage2k_exploration_release/diag_expression_wall_730705.txt
---

# Gate B Stage 2k: a novelty-gated commit-WTA release (FIX D) gets the un-sampled action SELECTED, REWARDED and its D1 route strongly potentiated on 730705 — but the learned route CANNOT EXPRESS at test, relocating the residual from selection to a bistable cortical-WTA expression wall

## Verdict (QUALIFIED — NOT a GO; smoke, not full battery)

**FIX D achieves Stage 2k's stated training goal on 730705 and does not regress dev, but
730705 steer does NOT flip.** Authoritative backend = **numpy** (a cupy run tests a
different brain per seed — see the 2j backend note). Honest smoke, not the frozen battery.

> ✅ **PARENT-VERIFIED (2026-08-07, per-seed parallel numpy).** Independent re-run confirms every
> claim here: (1) **byte-identity FIX D off ≡ Stage 2j** — `--mode byte` on 730703/730705/730606
> returns `all_byte_identical=true` (mismatch `{}`), so the Stage-2j GO is unaffected; (2) **FIX C+D
> dev 6/6, held-out 5/6** (730705 the only miss: count_c1=[37,3] — action 1 IS selected 3× during
> training — but D_contingent=0, `test_rate_c1`=0 → does not express at test); (3) **no dev
> regression** (730606 engaged, 730601 non-engaged, both steer=True). Held-out 5/6 = same as 2j's
> FIX B' GO, so FIX D is a diagnostic advance (residual relocated), NOT a new steer gain.

| seed (numpy) | FIX C+D | count_c1 | n_released | test_rate_c1 | D_contingent | steer |
|---|---|---|---|---|---|---|
| 730705 (held-out, the miss) | on | [37,3] (K=3) → [0,40] (K=40) | 3 → 40 | **0.000 at every K** | 0.0 / −0.15 | **False** |
| 730606 (dev, FIX C engages) | on | [4,36] | 3 | 1.0 | 1.0 (D_yoked −0.0) | True (no regression) |
| 730601 (dev, no engage) | on | [3,36] | 0 | 1.0 | 1.0 | True (byte-identical to 2j) |

**The one held-out miss (730705) is NOT closed.** FIX D makes action 1 win the WTA *during
training* (count_c1 flips off [40,0]; the proposal_1→str_d1_1 route potentiates 40→**315** at
K=40), so it IS selected and rewarded — but at test (FIX D off) action 1 reverts to
`test_rate_c1=0.000` regardless of how strongly the route was potentiated. The blocker is
**expression at test**, not selection or learning.

## Substrate diagnosis (5 instrumented probes — the residual is precisely located)

Full numbers: `raw/gateb_stage2k_exploration_release/diag_expression_wall_730705.txt` +
`diag_cascade_730705.txt`.

1. **Where the signal dies.** With FIX C (str_d1_1 gain ×3) str_d1_1 fires 138 onset
   spikes, gpi_1 pauses (212→51), **thal_1 releases (0→186)** — but **commit_1 = 0**: the
   incumbent's commit-level cross-inhibition `commit_fs_0 → commit_1` (weight 60, driven by
   commit_0=452) vetoes it, so motor_1 = 0. The death is at the **cortical commit WTA**, not
   the striatum or gpi.
2. **The proposal-novelty CURRENT is counterproductive here** — driving proposal_1 with
   350 pA drops str_d1_1 from 138→31 (it over-drives str_fsi / the indirect arm). So
   candidate (a), an un-satiable *proposal* novelty floor, makes 730705 **worse**. Refuted
   by measurement.
3. **Only a FULL release of the veto flips the WTA, dirtily.** Scaling `commit_fs_0→commit_1`
   ×0.75/0.5/0.25 does **nothing** (motor stays [856,0] — the cross-inhibition is
   near-saturating); ×0.0 flips to motor=[431,544] (winner 1, but a dirty win).
4. **FIX D (the training mechanism)** applies exactly that release, gated to the FIX-C dead
   channel and un-satiable until it has ≥K real selections, with the proposal novelty current
   suppressed for that channel. It works *as a selection aid*: K=3→40 gives count_c1
   (37,3)→(0,40), n_released 3→40, w1 40→315.
5. **But the learned route cannot express at test.** With str_d1_1 manually potentiated to
   300 and the circuit intact (release off), down-regulating the incumbent's gain — proposal_0
   (no effect at any k), commit_0 (only k×0.1 flips it, dirtily), or a two-sided striatal
   homeostat down-regulating str_d1_0 (even k×0.05 leaves motor=[550,84], winner 0) — does
   **not** cleanly recover action 1. The commit WTA is a strong **bistable latch** that
   proposal_0's intrinsic saturation always resolves to action 0.

## Why a persistent release would be a SHORTCUT (and is not used)

Making the commit-WTA release permanent (train + test) would flip 730705's test, but the
`acquisition_plasticity_share` control would then attribute the contingency to the
hand-rewire rather than to D1 plasticity (freezing acquisition-time D1 would leave action 1
winning by construction). FIX D is therefore **training-only and default-OFF**; expression at
test must come from learning through an intact WTA.

## FIX D properties (additive, default-OFF, byte-identical when off)

`--fix-d` (requires `--fix-c`): for the homeostat's dead channel, while it has < K=3 real
selections, transiently scale `commit_fs_{other}→commit_{dead}` to 0 for the trial (restored
after — commit_fs routes are non-plastic, so the restore is byte-exact) and suppress the
proposal novelty current into that channel. It is a novelty-gated *disinhibition* of an
inhibitory synapse in the SELECTION loop — distinct from the refuted FIX A (current injection
into str_d1). Gated to the FIX-C dead channel ⇒ on any seed where FIX C does not engage, FIX
D never engages. `_assert_fixd_off_byte_identical` (asserted in the smoke, verified on 730703;
730601 confirms 0 releases → identical to 2j) checks `--fix-d` OFF reproduces Stage 2j exactly.

## Next mechanism (no-defer, banked)

The residual is now **expression through a bistable cortical commit WTA that overrides the BG
(thalamic) action-selection signal on the extreme-bias 730705 draw** — thal_1 fires 186 at
test yet loses the WTA. Stage 2l must make the commit competition REFLECT the thalamic/BG
drive rather than raw cortical bias — e.g. divisive normalisation of the commit WTA, or a
thalamus-gated de-latching so a BG-selected channel can seize the cortical winner — so that a
learned str_d1_1 expresses at test through an intact, learning-legitimate circuit (the
acquisition-lesion control must still attribute the contingency to D1 plasticity). FIX C
(wakes the MSN) and FIX D (selects+rewards it in training) are the first two thirds; the
expression stage is the third.
