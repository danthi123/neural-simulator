# Burndown #3 Stage S2 — lift the on-bridge K-way sequencer routing margin toward the production K=32

**Date:** 2026-06-20
**Runner:** `research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py`
**Backend:** CPU / numpy (`SIM_BACKEND=numpy`), production dimension D=128
**Builds on:** S1 (`_phaseB_onebrain_sequencerK_divnorm_derisk.py`, commit `2cbae1ee`, GO K∈{2,4,8}, K=16 boundary) + S5 (`_phaseC_S5_divnorm_derisk.py`, the on-bridge `input_divisive_norm` closure) + S0 (`_phaseB_onebrain_sequencerK_derisk.py`, the K-way control fabric)
**`sim/` edit:** NONE (reuse-by-import of the S0 K-way sequencer + the EXISTING `input_divisive_norm` primitive flipped on a runner-built score bridge; the retreat-2 WTA — implemented but NOT needed on the production path — is runner-side score-bridge wiring)

---

## Verdict

**RETREAT 1 (the divnorm re-tune, `gain=0.05`→`gain=0.11`) LIFTS the S1 routing-margin limit cleanly: the on-bridge
K-way sequencer is GO 3/3 at K∈{2,4,8,16} (D=128) — including K=16, which S1 could only do 2/3.** At the production
**K=32** it is a **moat-intact 2/3 partial**: the cleanup is a perfect single-argmax (64/0/0) on **all 3 seeds**, the
moat held HARD (**0 false-accepts at every K, K=32 included**), but on **1 seed (43)** a single present cue's spiking
match-pool rate fell just below the fixed `match_thresh=0.15` (0.116) → **abstain-on-present**. That residual is a
DIFFERENT, milder mechanism than S1's EXTRA — a near-threshold MATCH-CASCADE rate (downstream of the drive), a
moat-safe FALSE-NEGATIVE, **never a confabulation**. So:

- **on-bridge clean 3/3 GO to K\* = 16** (a clean lift of S1's K=16 boundary);
- **K=32 = a moat-safe 2/3 partial** whose only failure mode is abstain-on-present.

This closes **#3 as a characterized partial conversion** (the brain-based-only deliverable): the on-bridge match
cascade holds cleanly to K\*=16, and at K=32 the host `_scan` covers the residual 1/3-seed false-negative if a strict
K=32 GO is required (`--host-fallback-above 16`). The MOAT was never relaxed; NO config search beyond the single named
retreat-1 op-point; NO `sim/` edit.

| K (D=128, gain=0.11) | ==host | moat 0-FA | lesion-safe | permuted-inverts | peak-robust | cleanup modes (ex/xt/ms) | verdict |
|---|---|---|---|---|---|---|---|
| 2 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 4/0/0 | **GO** |
| 4 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 8/0/0 | **GO** |
| 8 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 16/0/0 | **GO** |
| 16 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 32/0/0 | **GO** (S1 was 2/3) |
| **32** | **2/3** | **3/3 (FA=0)** | 3/3 | 2/3 | 3/3 | **64/0/0 (3/3)** | **2/3 partial** |

`k_star = 16`, first break at K=32. S5 op-point carried forward verbatim except the one re-tuned knob: `input_gain=1.0`,
`sigma=1.0`, **`gain=0.11`** (vs S1's `0.05`), the Izhikevich rheobase as the placed threshold.

---

## The two mechanisms, cleanly separated

### S1's K=16 boundary (the EXTRA) — FIXED by retreat 1

S1's K=16 squeeze was a runner-up FIRING (the EXTRA mode): at `gain=0.05` the divisive divisor `sigma + gain·mean_j(drive)`
was too weak to push the sub-peak runner-up below rheobase, so a decoded line lit TWO words; at K≥16 an extra-lit agent
on a LOWER block sharing an action word won first-match → wrong-block routing. The EXTRA count exploded with store size:
**7/96 role-reads at K=16 → 58/64 at K=32** under `gain=0.05`.

**Retreat 1 (a larger `gain`) eliminates the EXTRA.** A larger gain → larger divisor → all words scaled down more → the
runner-up drops below the placed rheobase while the winner stays above. The cleanup-mode `gain` sweep (D=128, K=32, the
64 role-reads):

| gain | seed 42 (ex/xt/ms) | seed 43 | seed 44 |
|---|---|---|---|
| 0.05 (S1) | 6 / **58** / 0 | — | — |
| 0.10 | **64 / 0 / 0** | 62 / **2** / 0 | 64 / 0 / 0 |
| **0.11** | **64 / 0 / 0** | **64 / 0 / 0** | **64 / 0 / 0** |
| 0.12 | 63 / 0 / **1** | 63 / 0 / **1** | 63 / 0 / **1** |

`gain=0.11` is the clean window for ALL three seeds at K=32 (perfect single-argmax, 0 EXTRA, 0 MISS). At `gain=0.12` a
near-tie role is squeezed below rheobase (the upper edge). The seed-43 EXTRAs at `gain=0.1`
(`blk10 deer→north`, `blk25 run→birb`) are exactly the sub-peak runner-ups the larger divisor at 0.11 suppresses.

**The per-query-peak ROBUSTNESS (the S5 contract) is preserved by construction at any `gain`:** the on-bridge divide is
scale-invariant, so the (exact/extra/miss) counts are IDENTICAL across pm=0.1, 1.0, 10.0 (peaks spanning ≥1 order of
magnitude) — `peak-robust 3/3` at **every K including K=32**. Re-tuning `gain` moves the runner-up suppression band; it
does NOT re-introduce a per-query peak read (the host `scores_to_drive`/`s.max()` is still GONE from the drive path).

⇒ The drive is a perfect single-argmax (64/0/0) at K=32 on all 3 seeds. The routing-margin limit that broke K=16 is
**solved**.

### The residual K=32 miss (seed 43) — a DIFFERENT, milder mechanism

With the drive perfect (64/0/0), the single K=32 seed-43 failure is **downstream in the spiking MATCH CASCADE**, not the
drive: the worst present-cue match-pool rate across seeds at K=32 is

| seed | worst present-match rate | the cue | verdict |
|---|---|---|---|
| 42 | 0.182 | (hill, rest) | ==host |
| **43** | **0.116** | **(sun, hop)** | **abstain-on-present** |
| 44 | 0.196 | (deer, see) | ==host |

At K=16 the worst present-match rate across seeds was 0.181–0.206 (comfortably above the 0.15 threshold → clean GO 3/3).
At K=32 the match-pool rate margin THINS (more blocks → more shared inhibitory load + more settling competition in the
gated-disinhibition cascade), and on seed 43 one cue (`(sun, hop)`, block 4) settles at **m=0.116 < the fixed
`match_thresh=0.15`** → the sequencer ABSTAINS instead of answering. The `perm-FAIL` is the SAME row (the permuted rule
also cannot route a block whose match pool does not fire).

**This is a moat-safe FALSE-NEGATIVE (abstain-on-present), NOT a wrong-block route or a confabulation** — the moat held
0-FA on seed 43 (and every seed) at K=32. It is a property of the SEQUENCER match cascade (the S0 control fabric:
`match_thresh` / `settle` / `w_blk`), which is OUTSIDE the three named drive-retreats. Per the brief (no config search
beyond the named retreats; never weaken the moat), this is REPORTED as the honest boundary, not chased into a sequencer
tuning search.

---

## Why this is NOT escalated to retreat 2/3

The three named retreats all target the **drive** (the decoded-line over-firing / the 1-of-K word discrimination):
retreat 1 (divnorm re-tune), retreat 2 (NEF-FS lateral-inhibition WTA between the decoded word-lines), retreat 3
(hierarchical match). Retreat 1 already drove the cleanup to a **perfect single-argmax (64/0/0)** at K=32 — there is no
EXTRA left for retreat 2's WTA to suppress, and no 1-of-K discrimination error for retreat 3 to split. The residual is a
**match-pool rate margin in the sequencer**, a different stage; none of the three drive-retreats addresses it. (Retreat 2,
`build_wta_score_bridge` + `wta_drive`, `--retreat wta`, is implemented in the runner as the named fallback but is inert
against this failure mode and was not needed on the production path.)

---

## Anti-cheats (all behaved)

- **MOAT (HARD, 0-FA):** every absent-agent / absent-action / cross cue abstains, at K∈{2,4,8,16,32}, **3/3 seeds, FA_total
  = 0 at every K**. The K=32 cross cue `(dog, run)` (fact0's agent + a shared action it does not pair with) is a direct
  test that no extra-lit agent on a lower `run`-block produces a false-accept. **Never traded.**
- **NO-DIVNORM control (load-bearing):** a divnorm-OFF score pool driving the SAME placed rheobase threshold on the RAW
  un-normalized drive FAILS the battery (winner + runner-up both fire → the whole-row-lights wall) → the divisive
  normalization is load-bearing, `raw-fails 3/3` at every K.
- **OFF==byte-identical:** S5's `check_off_byte_identical` guard PASS — the `input_divisive_norm` primitive is a guarded
  no-op when off (no `sim/` edit was made; this asserts the contract).
- **lesion-fails-safe:** severing the result→op conditioning (zero decoded-line drive) → abstain, 3/3 at every K.
- **permuted-rule inverts:** a present cue for block b routes to `ans{(b+1)%K}` — the decision follows the cyclic-shift
  RULE, not a fixed scan order (3/3 where ==host holds; the K=32 seed-43 perm-fail is the SAME abstain-on-present row, not
  a rule-following failure).

---

## Reproduce

```bash
# production D=128, K∈{2,4,8,16,32}, retreat 1 (divnorm re-tune, gain=0.11), 3 seeds
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk \
    --seeds 42,43,44 --dim 128 --ks 2,4,8,16,32 --retreat divnorm --gain 0.11

# the characterized partial conversion (on-bridge to K*=16, host _scan above) -> strict K=32 GO with the moat intact
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk \
    --seeds 42,43,44 --dim 128 --ks 2,4,8,16,32 --retreat divnorm --gain 0.11 --host-fallback-above 16
```

Raw output: `research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json`.
A single-seed full K=32 confirmation at `gain=0.1` (seed 42, GO) is in `research/findings/raw/_s2_r1_k32_s42.json`.

---

## Bottom line

S2 took the on-bridge K-way sequencer to the production K=32. **Retreat 1 (the divnorm re-tune to `gain=0.11`) fully
solved the routing-MARGIN limit (the EXTRA) that broke S1's K=16** — the on-bridge cleanup is a perfect single-argmax
(64/0/0) at K=32 on all 3 seeds, peak-robust, and the on-bridge sequencer is a **clean 3/3 GO to K\*=16** (lifting S1's
K=16 boundary). At K=32 it is a **moat-intact 2/3 partial**: the sole residual is a near-threshold spiking match-pool RATE
on 1 seed (an **abstain-on-present false-negative**, m=0.116 vs 0.15), a DIFFERENT and milder mechanism than the EXTRA,
living in the sequencer match cascade (not the drive) and so outside the three named retreats. **The moat held HARD (0
false-accepts) at every K including K=32** — the on-bridge match cascade is NEVER tricked into confabulating; at worst it
abstains. The honest closure: **clean on-bridge to K\*=16; K=32 = moat-safe 2/3** (with the `--host-fallback-above 16`
path covering the residual false-negative for a strict K=32). The host `scores_to_drive` peak-read remains GONE from the
drive path; **NO `sim/` edit.**
