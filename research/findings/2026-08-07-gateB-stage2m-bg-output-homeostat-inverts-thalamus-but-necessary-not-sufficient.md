---
type: finding
status: no-go
date: 2026-08-07
mechanism: gateB-stage2m-FIXE-bg-output-intrinsic-homeostat
backend: numpy
runner: research/runners/_vocal_gateb_stage2m_bg_output_homeostat.py
builds-on: 2026-08-07-gateB-stage2l-commit-normalization-refuted-residual-relocated-to-BG-output-readout.md
artifacts:
  - research/findings/raw/gateb_stage2m_bg_output_homeostat/smoke_730705_numpy.json
  - research/findings/raw/gateb_stage2m_bg_output_homeostat/diag_730705.txt
---

# Gate B Stage 2m: FIX E (a BG-output intrinsic-excitability homeostat) INVERTS the thalamic drive that Stage 2l called the wall, but is NECESSARY-NOT-SUFFICIENT to flip 730705 standalone — the residual relocates to the commit's integration of a thalamic temporal head-start, and (with FIX D off) to training-time exploration

## Verdict (NO-GO for FIX E standalone; a measured RELOCATION, plus a concrete legitimacy-preserving closing stack)

FIX E is the direct downstream analogue of FIX C (Stage 2j MSN k-homeostat): a target-blind
intrinsic-excitability homeostat on the BG-output pools (GPi + thalamic relay) that scales the
Izhikevich gain `cp_izh_k` per channel toward a COMMON per-region baseline set-point
(Desai 1999 / Turrigiano 2011; NOT current injection). Additive, DEFAULT-OFF, byte-identical
when off (`--mode byte`, ASSERTED `all_byte_identical=true` on 730703/730705). Authoritative
backend = **numpy**. The Stage-2j/2k GO is unaffected.

**The real standing smoke (fix_c on, fix_d off, FIX E on) does NOT flip 730705**:
`test_rate_c1 = 0.0`, `count_c1 = [40, 0]`, `steer = false` (smoke_730705_numpy.json). But the
diagnosis Stage 2l built its wall on is **refuted in part**: FIX E DOES invert the thalamic
aggregate drive — the very quantity 2l measured as unfixable.

## What FIX E actually does (raw: diag_730705.txt)

On 730705 FIX E engages on BOTH BG-output regions under the extreme baseline asymmetry gate:

| region | baseline f0 | baseline f1 | set-point | k-scale applied |
|---|---|---|---|---|
| gpi  | 37  | 215 | 126 | gpi_1 ×0.15 (down) |
| thal | 203 | 0   | 101 | thal_0 ×0.15 (down) |

It partially equalises the baseline (gpi [37,215]→[37,143]; thal [203,0]→[151,0]) and, on the
FIX-D-trained bridge, **inverts the thalamic aggregate at test**: `thal [273,215] → [215,228]`
— thal_1 now EXCEEDS thal_0. Stage 2l's headline ("on 730705 the thalamic drive favors action
0, so no commit competition that reflects thalamic drive can select action 1") is therefore
false as an absolute: a **target-blind intrinsic homeostat at the BG output CAN invert the
thalamic drive**.

## Why FIX E is still necessary-not-sufficient

Despite `thal_1 > thal_0` in aggregate, the motor winner does NOT flip (`motor [795, 0]`,
action 0). The commit WTA **integrates thal_0's TEMPORAL head-start**: thal_0 fires first
(entering the onset primed) and commit_0 ignites and latches before thal_1's later spikes
accumulate. Removing the commit latch (Stage-2l soft-WTA de-latch, scale 0.0) still gives
`commit [388, 359]` → action 0, because the head-start's early spikes already bias the commit
integrator. The head-start is set at BASELINE by the str_d1 firing asymmetry
(str_d1_0 ~86 vs str_d1_1 ~0 → gpi_0 paused → thal_0 primed to 272), and the str_d1 baseline is
NOT k-reducible (str_d1_0 stays ~86 at k×0.1 — consistent with 2j: k does not silence an
already-firing MSN).

## The concrete, legitimacy-preserving closing stack (probe-level, FIX D trained bridge)

Three standing test-time properties TOGETHER flip it — **FIX E + Stage-2l commit de-latch + an
onset entry-state equalisation** (reset both gpi AND both thal channels to a common membrane
value at onset, removing the temporal head-start): `thal [214,255]`, `commit [371,399]`,
`motor [772,779]` → **11/12 action 1**. Each piece is required: FIX E alone (commit latches),
FIX E + de-latch alone (temporal head-start), and de-latch + entry-eq without FIX E (thal not
inverted) all stay action 0. **Legitimacy holds**: on an untrained (no-D1-learning) bridge the
same stack stays action 0 (`motor [742,135]`, thal_1 only 24) — it needs the trained str_d1_1
onset drive, so it does not manufacture the policy. The onset entry-equalisation is a
scaffold-flavoured membrane reset (biological analogue: a TRN/feedforward inhibitory reset
between selection epochs) and is offered as a lead to validate, not a closed mechanism.

## Legitimacy of FIX E itself (the shortcut check — PASSED)

The frozen acquisition-lesion control with FIX E ON: `D_contingent_acq_lesion = 0.0`,
`p_action0(target=1) = 1.0`, `test_rate_target1 = 0.0` — action 1 does NOT win without the D1
learning (smoke_730705_numpy.json → `legitimacy_acq_lesion`). FIX E equalises baselines
target-blind; it does not own the contingency.

## Two honest sub-walls, by config

1. **FIX D off (the requested standing-property test):** the wall is TRAINING-TIME EXPLORATION
   — `count_c1 = [40, 0]` means action 1 is never sampled during training, so no policy forms
   to express. FIX E's standing BG-symmetrisation does not overcome the channel-0-open lock
   during ordinary (non-novelty) trials, so exploration still fails. (FIX D was precisely the
   training-time exploration-release that produced the trained policy in the 2l/2k diagnosis.)
2. **FIX D on (the diagnostic bridge):** the trained policy exists; FIX E inverts the thal; the
   remaining residual is the commit's integration of the thalamic temporal head-start, closed
   by the three-part stack above.

## Mechanism properties (additive, default-OFF, byte-identical when off — ASSERTED)

`FIX E` engages only under an extreme BG-output baseline asymmetry (ratio > 5.0); on 730705 it
fires on gpi and thal, on well-behaved seeds it need not fire. k-scales are calibrated on a
same-seed PROBE bridge (training-bridge RNG untouched) and applied via a build-time wrapper
around Stage-2k's `build_stage2_bridge` (2k/2l stay intact). FIX E OFF is a no-op wrapper →
byte-identical: `_assert_fixe_off_byte_identical` measured `all_byte_identical=true`
(mismatch `{}`) on 730703/730705.

## Banked method + no-defer next step

BANKED (refuted standalone): a BG-output intrinsic-excitability homeostat inverts the thalamic
drive but cannot by itself flip 730705 — the commit integrates a baseline-set thalamic temporal
head-start, and with FIX D off the prior wall (exploration) re-emerges. NEW method (validated
at probe level, awaiting a full standing smoke): **FIX E + Stage-2l commit de-latch + an
onset entry-state equalisation** (a TRN-like inhibitory selection-epoch reset), which flips
730705 11/12 while preserving contingency. The deeper origin — the str_d1 baseline firing
asymmetry that sets the channel-0-open lock and is not intrinsic-k-reducible — is the true
upstream residual for the arc owner to weigh at this checkpoint.

## Parent validation commands (numpy, orphan-proof)

```bash
export PYTHONPATH=$PWD SIM_BACKEND=numpy
# byte-identity when off (must be all_byte_identical=true -> GO protected):
.venv/bin/python -m research.runners._vocal_gateb_stage2m_bg_output_homeostat --mode byte \
  --out research/findings/raw/gateb_stage2m_bg_output_homeostat/byte_numpy.json
# standing FIX E smoke on the held-out miss (SMOKE_730705_test_rate_c1_flips must be false):
.venv/bin/python -m research.runners._vocal_gateb_stage2m_bg_output_homeostat --mode smoke \
  --smoke-seeds 730705 \
  --out research/findings/raw/gateb_stage2m_bg_output_homeostat/smoke_730705_numpy.json
# dev no-regression under FIX E (steer_passes should not fall below Stage-2k):
.venv/bin/python -m research.runners._vocal_gateb_stage2m_bg_output_homeostat --mode seeds \
  --dev-seeds 730601 730602 730603 730604 730605 730606
# full frozen battery under FIX E (dev steer + acquisition lesion + reversal):
.venv/bin/python -m research.runners._vocal_gateb_stage2m_bg_output_homeostat --mode full
# evidence dump (the tables above):
.venv/bin/python -m research.runners._vocal_gateb_stage2m_bg_output_homeostat --mode diag \
  --diag-seed 730705
```
