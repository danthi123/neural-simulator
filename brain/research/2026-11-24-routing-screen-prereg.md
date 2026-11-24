# Capability: Episodic recall
# Record: brain/research/2026-11-24-routing-cascade-research.md (section 5 amendment); results/2026-11-20-rank-raw-screen-1.5b.md; results/2026-11-23-m3-margin-screen.md; 2026-11-21-hub-research-round.md; _scratch/route/{rawwin,frozen_vs_intact,wrong28}.py
# Expected-fail: partly. The S0 key is a lexical layer-0 stand-in (SCAFFOLDS.md) and near-duplicate separation is MISSING (PREREQUISITES.md), so a ranking change can remove strength bias but cannot separate true near-duplicates. D+G is expected to lose some A-right fetches: 20/45 of them are weighted-only winners (amendment, section 5). SC at cap 1.0 may cut right fetches with fires, because it lowers every hub's score below the bar it was fitted at.

Prereg for the D1 offline routing screen (dev, fetch level). Written 2026-11-24 before any KR data exists. No model was run
for it. The denominators below were computed from banked `_scratch/stage2/read/lifeKP_s319-321.json` (host only).
Runner: `brain/experiments/routing_replay.py`. Test: `brain/tests/test_routing_replay.py` (stub cortex).

## 1. Record brief
- More than 15 D1 levers are on record (hub round table). The ones that bear on this screen:
  - R-raw arm D (raw ranking on S0): right fetches 8 -> 14, but fires on open/greet/never rose 32 -> 41. It was KILLED with S_all.
  - S_all moved fetches 1/18 -> 14/18 and left the replies flat. M3 margin gating does not separate (AUROC 0.42-0.44).
- Frozen arm, nights 5-15, 223 told rows:
  - 141 rows fetch a wrong episode and 45 fetch the right one.
  - 110 of the 141 wrong fetches are weighted-only winners.
  - The use boost during the battery adds nothing (intact vs frozen are equal).
- The hub follows strength: Philippa 9.0 in s319, Bramble 8.5 in s320. The median strength is 1.0.
- The amendment ranks strength first and gives this screen its shape: D plus a fire gate, and a cap on strength (H3).
- The screen needs per-night stores. The K-P lives are being rerun as lifeKR_s319-321 with `--night-snapshots` for that.

## 2. Arms
All three arms are computed from one cue read per row, `Hippocampus.similarities(cortex, base, cue)`, with FREEZE on.
Each arm is a wrapper over `_compete`. `hippocampus.py` is not edited.
- **A (K-P as is).** `_compete(sims * tanh(eff), sims=sims)` under FREEZE_USE, which is the path the banked
  intact-frozen arm took.
  - Validity: A must reproduce the banked `fetch.idx` and `recalled` on every battery row of the night, and
    `right_episode` on every told row.
  - Any mismatch voids that night. A seed with more than 2 of 11 nights void is void.
- **D+G (D plus fire gate).**
  - Fire only if A fires (the existing weighted bar of 0.12 for a question or 0.45 otherwise, and MARGIN 0.02, unchanged) AND the raw
    winner's own weighted score `sims[r] * tanh(eff_r)` clears the same bar.
  - The episode reinstated is the raw-similarity winner r.
  - Firing therefore never exceeds A's. D+G equals A wherever the raw and weighted winners agree; the unit test checks this.
  - No free parameter.
- **SC (strength cap, one a-priori form).**
  - The form is A's competition with weight `tanh(min(eff, 1.0))` on non-permanent episodes. `eff` is the recency-weighted
    `effective_strength` and is unchanged. Permanent episodes keep `max(strength, 3.0)`.
  - **Why 1.0:**
    - It is the code's own birth strength (`Episode.strength` default = `store(salience=1.0)`): use cannot make an episode
      compete harder than a fresh telling, and decay still ranks old against new.
    - The code's other constant, 3.0 (the permanent floor), does nothing here. tanh(3) = 0.995, so a 9.0 hub already sits at the cap's
      weight.
    - The median is data-dependent and would be a fitted value.
  - No grid.
- **D (reference only, no bars).** The existing `RANK_ON_RAW` switch. It is reported so that this screen ties to the R-raw screen.

## 3. Population
- Seeds: lifeKR_s319, s320, s321. Trunk, nights 5-15, the intact-frozen battery. Each night's store is that night's
  snapshot, and each row's fetch depends only on the snapshot and the cue.
- **Told rows** (the research note's 223) are rows whose item has a ledger `telling` and is scored for a fetch: H,
  rotating, and told near-duplicate. Per seed: 80, 70 and 73.
- **Fire rows** are the open-* (6 per night), greet, never-told and untold near-duplicate rows. Per seed: 118, 128 and 125 (371 in all).
- **Identity rows** are H items with no telling, name and nova, 22 per seed. They are reported only.

**lifeKP denominators (intact-frozen, nights 5-15):**

| seed | told n | right | wrong | no fetch | fire n | fires (open/greet/never/nd) |
|---|---|---|---|---|---|---|
| 319 | 80 | 16 | 50 | 14 | 118 | 80 (38/7/27/8) |
| 320 | 70 | 15 | 47 | 8 | 128 | 74 (33/10/22/9) |
| 321 | 73 | 14 | 44 | 15 | 125 | 94 (53/11/25/5) |
| pooled | 223 | 45 | 141 | 37 | 371 | 248 |

The replay's arm A must reproduce these. Bars are read on the rows that survive (same fractions).

## 4. Bars (KILL first; for D+G and SC separately)
- **V (validity, before any bar).**
  - `routing_replay validate KR KP`: each KR life equals lifeKP on every probe row (`you`, `brain`, `recalled`,
    `fetch.idx`). A mismatch voids the seed.
  - Night voids follow section 2.
  - If any seed is void, the result is INCONCLUSIVE and no bar is read.
- **K1.** Pooled right-episode fetches on told rows below `max(ceil(1.25 x A), A + 10)` kill the arm. With full data that is 57 or more of 223, against A's 45.
- **K2.** The arm is killed if right-episode fetches rise on fewer than 2 of 3 seeds, or fall by more than 1 on any seed.
- **K3.** The arm is killed if more than 1 row per seed turns from A's right episode into a wrong episode. The A-right base is 16, 15 and 14. Expected killer for D+G: 20 of A's 45 right fetches are weighted-only winners, which raw ranking is expected to lose (review 11-24).
- **K4.** The arm is killed if fires on fire rows exceed A's by more than 1 on any seed. A fires 80, 74 and 94. K4 is vacuous for D+G by construction (it fires only where A fires); it binds SC.
- **PASS means none of K1-K4 fired.** It is a **fetch-level result only**: S_all once moved fetches 1/18 -> 14/18 with the
  replies flat. A PASS earns one live DEV K-P life with the arm (s319-321, AWS g5) and nothing more. It is not a claim.
  Six seeds come before any claim.
- **Reported, not barred:**
  - per night with n;
  - the 28 wrong-hurt rows;
  - top-1 fetched share;
  - lost-to-no-fetch;
  - identity rows;
  - arm D;
  - the maximum |sim_raw - banked| per night.
- No grid, no tie-break and no refit. All three seeds were seen for design (the amendment counts), so this is not held-out.

## 5. Host decisions and constants
- **Host decisions:** none beyond existing scaffolds. Question vs statement bar = `cue.endswith("?")`, as in `Brain.hear`.
  The episode text is used only to score `right_episode` (`life.episode_is_source`).
- **New constant:** `SC_CAP = 1.0`. It lives in the screen script only. It needs a SCAFFOLDS.md row (with the D+G rule,
  which has no constant) only if an arm becomes a Brain switch for the DEV life.

## 6. Timebox and cost
- **Compute.** Pool40, CPU fp32, under `tools/memcap.sh`. RAM is about 7 GB. Only cue reads run, with no generation (`--reinstate` is off).
  - About 22 reads per night, so about 1-2 min per night and 11 nights per seed.
  - That is about 20 min per seed, about 1 h serial or 20 min across pool40-42. $0.
- **Drift risk, declared now.** 11-13 rows per seed sit within 0.002 of the margin bar, on 7-8 nights. CPU fp32 cue keys against
  fp16-made stored keys may void nights.
  - **Fallback:** if more than 2 nights are void on any seed and every void row is a near-tie (the maximum sim deviation is
    printed), rerun the same command with `--device cuda` on one AWS g5 (about 20 min, under $1). This is not a bar change.
- **Timebox:** one working day from KR data to the read. If it overruns, stop and write up what exists.

## Review (sonnet, 11-24, before data): OK_WITH_FIXES, applied
`score` now requires one `--validity` file per replay output (no gate = no read); the worst cue-to-key similarity deviation
from banked is printed as a tripwire; K4 vacuous for D+G and K3 its expected killer stated above. The optional `--reinstate`
reply path is unbarred and trusted only if `A_reproduces_banked_replies`.
