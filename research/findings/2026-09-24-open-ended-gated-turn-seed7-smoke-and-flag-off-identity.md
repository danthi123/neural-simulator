---
type: finding
status: measured
date: 2026-09-24
lane: load-bearing (open-ended conversation; plan step S16, lane A3)
mechanism: open-ended-gated-turn (BRAIN_OPEN_ENDED_GATED, default OFF)
seeds: [7]
verdict: dev-seed integrity checks only. No gate seed has run. Nothing here is a capability verdict.
---

# Open-ended gated turn: seed-7 dev smoke and flag-off identity (S16 integrity checks, 2026-09-24)

seed-waiver: seed 7 is the development seed. These are the integrity checks the pre-registration lists as NOT governed.
No claim here generalises beyond seed 7.

Pre-registration: [`2026-09-24-open-ended-gated-turn-PREREGISTRATION.md`](2026-09-24-open-ended-gated-turn-PREREGISTRATION.md)
(committed `b4588775a`; Amendments 1 `98c4912e2`, 2 `7a39851cd` and 3 `6994e80f2`, each before any gate-seed run).
Owner decision (2026-09-24): Qwen stays fact-free. The rest is this lane's own inference, not a separate owner
ruling on this flag: the flag stays default OFF whatever the gates read (pre-registration A3.3).
The 2026-09-23 a3 NO-GO on the ungated turn stands:
[`2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md`](2026-09-23-open-ended-production-turn-a3-6seed-harvest-NO-GO.md).
Branch `research/open-ended-gated-turn`. `BRAIN_OPEN_ENDED_GATED` stays default OFF; nothing here flips a default.

## What was measured

All runs: numpy backend, stub renderer, LLM disabled, `BRAIN_CHAT_SEED=7`, the real `webapp.server.brain_chat`
through the regression battery's own worker. Runner: `research/runners/_open_ended_gated_turn_smoke.py`.

1. **Conditioning smoke after Amendment 1** at `3ef5ec08` (pool2, LTM off; turns `unknown`, `emo`, `sw_open`).
   Arms: intact_a, intact_b (independent rebuild, the null), gate lesion, affect lesion, GNW workspace lesion.
   Scored file: `research/findings/raw/_open_ended_gated/smoke_amend1/s7/smoke_summary.json`.
2. **Flag-off identity** (flag unset) at `3ef5ec08` vs its merge-base with main `306ef27d`.
   Compare: `research/findings/raw/_open_ended_gated/identity/compare_off5_ltmoff_s7_306ef27d_vs_3ef5ec08.json`.
3. **A descriptive race-history read** of the speak/abstain selector (no brain), `--bg-order` at `b97dea71`:
   `research/findings/raw/_open_ended_gated/bg_curve/bg_order_dev_s7_11_13.json`.
4. **Flag-OFF GNW-lesion read of `chase`** (added in the review fix round): intact and `BRAIN_GNW_2ORGAN_WS_LESION=1`,
   flag unset, seed 7, LTM off, through the battery worker. Files: `research/findings/raw/_open_ended_gated/smoke_gnwturn_flagoff/s7/`.

## Results

**The null is clean.** intact_a and intact_b agree on every conditioning field and on the reply, on all three turns
(`smoke_summary.json`: `null_clean` true, `null_diffs` empty).

**The GNW workspace lesion moves the gated turn's ROUTE on a KB-hit turn. The reply change is the pipeline's, not
the gated turn's.** (Corrected after the 2026-09-24 review; PREREG Amendment 3. This paragraph first said the lesion
"moves the gated turn's decision" and counted three decision-field diffs.)
On `sw_open` ("what does the dog chase") the route goes grounded -> withheld, the BG action SPEAK -> STAY_SILENT, the
reply kind grounded -> withheld_abstain, and the answer goes from "the dog chases the cat" to "I don't know about that."
(`smoke_summary.json`, `changes_vs_intact.gnw_lesion.sw_open`). The cut held at read time: `cut.ws_lesion` is true in `gnw_lesion.json`.
Those changes have one cause. On a withheld route `decide()` keeps the pipeline's answer whatever the race commits.
The BG action moves only through the host FAM table (withheld -> familiarity 0), and `withheld_abstain` is a direct map
of the route. The answer change is the pipeline's own gate abstain.
With the flag OFF the same lesion makes the same reply change on `chase`: "the dog chases the cat — worth going
further here." -> "I don't know about that. — worth going further here."
(`research/findings/raw/_open_ended_gated/smoke_gnwturn_flagoff/s7/intact_off.json` and `gnw_lesion_off.json`, `chase.answer`;
seed 7, LTM off, local numpy. The intact arm ran at `7a39851c` plus this round's uncommitted edits, none on the flag-off
path; the lesion arm at `6994e80f2`. `webapp/server.py` is the same at both.)
Under Amendment 3 the row `open-ended-turn-gnw-drive` decides on `route` only. On its registered turn `chase` it scores
load-bearing at seed 7 with 1 decision-field diff and 0 in the null
(`research/findings/raw/_open_ended_gated/smoke_gnwturn/s7/smoke_summary.json`, `rows.open-ended-turn-gnw-drive`, re-scored at `6994e80f2`).
On the off-KB turns (`unknown`, `emo`, `question`, `open`) the GNW lesion changed nothing at seed 7.

**The affect lesion moves the afferent, not the decision, at seed 7.** On `emo` the Gate-B valence sign goes + -> 0
(a trace copy the lesion sets, excluded from the row by the pass-by-construction audit).
The BG speak salience goes 2/3 (0.667 <!--derived-->) -> 0.0 (`conditioning.emo.affect_lesion.salience`; `cut.affect_lesioned` true).
The BG action is STAY_SILENT in both arms and the reply is identical, so the row `open-ended-turn-affect-drive`
scores not load-bearing at seed 7: 0 decision-field diffs, clean null (`rows.open-ended-turn-affect-drive`).

**Why the intact emo race held at s = 2/3.** <!--derived--> The selector persists across a session's turns on the session's private
RNG timeline, so the emo race is the second race, after the `unknown` turn's race at s = 0.
`--bg-order` replays exactly that order outside the pipeline and gets STAY_SILENT at seed 7 as well
(`bg_order_dev_s7_11_13.json`, `per_seed.7.smoke_order_race2_s_hi`).
So the in-pipeline decision matches the isolated selector read. It is one draw, not a wiring fault.
At s = 2/3 the pooled P(SPEAK) over dev seeds 7/11/13 is 17/24 when each race follows an s = 0 race and 15/24 over
consecutive races (`pooled_speak_after_s0`, `pooled_speak_fresh_consecutive`).
The ascending `--bg-curve` read 21/24 at 0.67 (`bg_curve_dev_s7_11_13.json`, `pooled`).
Amendment 1's expected intact P(SPEAK) for the affect row was taken from that curve, so it is too high by about 4-6
races in 24. At these dev rates a single-shot affect row would read no change (STAY_SILENT in both arms) roughly a
third of the time, even if the afferent coupling is real. The rate in the LBF depends on which races precede `emo`
in that process, which this read does not fix.

**The gate lesion (the afferent cut) does not move the BG action at seed 7, and its marker change is by construction.**
(Corrected after the 2026-09-24 review; this paragraph first reported the marker change as an effect.)
The saliences are cut to (0.5, 0.5) on all three turns (`cut.baseline_applied` true), and the BG action is unchanged on
each at seed 7. On `emo` the marker register goes 2 -> none (`changes_vs_intact.gate_lesion.emo`).
That is not an effect of the cut: the same lesion runs the marker WTA with its own `lesion=True`, which returns None on
(almost) every trial. Amendment 3 removes `marker_level` from the row's decision fields.
The row `open-ended-turn-faculty-drive` probes `unknown`, where it scores not load-bearing at seed 7.
There the marker is never read (a neutral mood is not sent to the circuit), the intact race sits at s = 0 and the cut
race at s = 0.5. A single race at 0.5 is a coin flip (12 of 24 dev races spoke; `bg_curve_dev_s7_11_13.json`, `pooled`).
So Amendment 3 declares the row descriptive only: its verdict is reported and never counted toward a load-bearing claim.

**Flag-off identity holds on the turns checked.** With the flag unset, the response JSON of 5 turns (`well`, `unknown`,
`sw_open`, `rich_well`, `rich_open`; LTM off) is identical at `3ef5ec08` and at `306ef27d`.
Both hash to sha256 `fc3b716d195e...` (`compare_off5_ltmoff_s7_306ef27d_vs_3ef5ec08.json`, `identical` true).
The same 5-turn check was identical at `928d0d82` vs `f554056e` (`research/findings/raw/_open_ended_gated/identity/compare_off5_ltmoff_s7.json`).
**Scope deviation, disclosed:** the pre-registration's integrity section names the full probe roster.
These 5 turns reach both hooked paths (single-fact and rich), but they are not the roster.
The 26-turn roster pair (`49b8fb12` vs its merge-base `355db9c7`, seed 7, LTM off) was staged on the pool at 16:29; at
17:25 one half was running and the other queued, so it is not reported here.
`webapp/server.py` at the fix-round head `b728777a` is identical to `49b8fb12` (`git diff` empty), and the gated module
is never imported with the flag off, so that pair covers the lane's flag-off footprint at the new head when it lands.

## Against the plan's S16 success check

The check reads: conditioning state changes under the affect lesion and under the GNW lesion, a clean null, and
`byte_identical_off` true.
- GNW lesion: met only for the gated turn's route label on the KB-hit turns (`sw_open`, `chase`). The reply change
  there is the pipeline's own abstain. On the off-KB turns nothing moved.
- Affect lesion: met only at the trace and afferent level (valence sign, speak salience). The BG decision did not move
  at seed 7. This is not evidence that the affect afferent drives the turn.
- Clean null: met.
- Flag-off identity: met on 5 turns, exact compare in the data. Not run on the full roster.

## Integration defect found at the merge with main (Part B)

The AG-REG row hook merged on main (`338d9ecee`) parks any row whose probe turn is not in
`onebrain_regression_battery.PROBE_TURNS`. `sw_open` is a label-only turn (kept out of the default roster by the
swap-drives preregistration), so `open-ended-turn-gnw-drive` is parked: `LBF_ROW_MERGE_REPORT.parked` names it,
and only the faculty-drive and affect-drive rows enter the registry.
As registered, Part B would measure 2 of its 3 rows, and the one row that moved at seed 7 would be missing.
Fixing it needs either the hook's owner to accept label-only turns from `_TURN_BY_LABEL` (the LBF's `turn_group`
already resolves them), or a pre-registration amendment that points the row at a default-roster turn.
Such an amendment needs a dev-seed read of that turn first.

**Resolved by Amendment 2 (before any gate-seed run).** A seed-7 read of the default-roster candidates `chase`,
`question` and `open` under the GNW lesion, at `3ef5ec08` on pool2 (LTM off), shows the `sw_open` change on `chase`.
The route goes grounded -> withheld, the BG action SPEAK -> STAY_SILENT, and the reply kind grounded -> withheld_abstain.
The intact rebuild is identical to intact
(`research/findings/raw/_open_ended_gated/smoke_gnwturn/s7/smoke_summary.json`, `changes_vs_intact.gnw_lesion.chase`).
`question` and `open` route off-KB and hold in every arm. The pre-registration's Amendment 2 re-points the row to
`chase`, and the hook now merges all three rows (`LBF_ROW_MERGE_REPORT.keys_added`, none parked).
That summary was first scored at `639cce2f9`, before the row was re-pointed, so its row block read `sw_open` (not
exercised) and contradicted this paragraph. It was re-scored from the same arm files at `6994e80f2` and now reads the
row on `chase`, decided on `route` only (Amendment 3).
**"none parked" described the hook as it stood at `338d9ecee` (turn-membership parking only). Corrected by Amendment
4, below, for the hook as merged from main afterward (`eed3652a0`, REQUIRED_ENV parking).**

## Corrections after the 2026-09-24 review (PREREG Amendment 3)

- The GNW paragraph above overstated the result: the lesion moves the gated turn's route label on KB-hit turns, and
  the reply change is the pipeline's own abstain. The three "decision-field diffs" were one cause.
- The gate lesion's marker change on `emo` is by construction, not an effect.
- The faculty-drive row cannot discriminate as a single-shot read and is now descriptive only.
- Part A, which has not run, was re-designed before any gate seed: CONT became a manipulation check and the Part A
  verdict is the reply-level metric (see the pre-registration's Amendment 3).

## Corrections after a second 2026-09-24 review (PREREG Amendment 4)

- **"none parked" (above) described the hook merged at `338d9ecee`, before this branch merged main's later
  `eed3652a0` (REQUIRED_ENV opt-in parking).** With main merged, at production defaults (`BRAIN_OPEN_ENDED_GATED`
  unset) all three row keys are parked and OUT of `load_bearing_fraction.FACULTY_LESIONS`/`FACULTY_PROBES`
  (confirmed by direct import); with the flag set, `open-ended-turn-affect-drive` and `open-ended-turn-gnw-drive`
  enter it (`open-ended-turn-faculty-drive` stays parked either way -- next bullet). Measured with
  `tools/lb_shard.py jobs ... --extra-env BRAIN_OPEN_ENDED_GATED=1 --faculties <rows>`.
- **`open-ended-turn-faculty-drive` is now `PARKED`, not merely `DESCRIPTIVE_ONLY`.** `load_bearing_fraction.py`'s
  own scoring does not read `DESCRIPTIVE_ONLY` (only the dev-smoke `score_row()` above does), so the coin-flip row
  would have silently entered a real headline battery's load-bearing counts once `REQUIRED_ENV` was met. The module
  now declares `PARKED = {"open-ended-turn-faculty-drive": "..."}`, which the registry hook honours unconditionally
  (`load_bearing_fraction.py` itself is unmodified). It stays measurable only via `--bg-curve` and this row's own
  `score_row()` dev-smoke read; it can never contribute to a load-bearing fraction number. `FACULTIES` (the
  b2b-caps `--faculties` list) now names the two keys that actually enter the registry.
- **The wrong number in `DESCRIPTIVE_ONLY`'s reason string is fixed.** It read "intact P(SPEAK) 1/24 at (0, 1)"; the
  pooled dev bg-curve at s = 0.0 is SPEAK 0 / STAY_SILENT 23 / none 1 of 24
  (`research/findings/raw/_open_ended_gated/bg_curve/bg_curve_dev_s7_11_13.json`, `pooled["0.0"]`) -- this
  finding's own "STAY_SILENT on 23/24 dev races" above was already correct; only that one string had the stray "1".
  Now reads "intact P(SPEAK) 0/24".
- **The owner decision quoted at the top of this finding is re-labelled.** "Qwen stays fact-free" is the owner's
  2026-09-24 statement. "So the flag stays default OFF whatever the gates read" is this lane's own inference from
  it (pre-registration A3.3), not a separate owner ruling on the flag.

## What this does not show

- Nothing about the gate seeds. Part A (the capability gate) and Part B (the LBF rows) have not run.
- Nothing about rendered Qwen text. The LLM was disabled in every run.
- That the affect afferent drives the reply. At seed 7 it did not.
- That the gated turn changes a reply under the GNW lesion. On the KB-hit turns the reply change is the pipeline's.
