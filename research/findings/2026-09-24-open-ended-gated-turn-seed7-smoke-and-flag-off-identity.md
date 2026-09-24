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
(committed `b4588775a`; Amendment 1 committed `98c4912e2`, before any gate-seed run).
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

## Results

**The null is clean.** intact_a and intact_b agree on every conditioning field and on the reply, on all three turns
(`smoke_summary.json`: `null_clean` true, `null_diffs` empty).

**The GNW workspace lesion moves the turn's decision.** On `sw_open` ("what does the dog chase") the route goes
grounded -> withheld, the BG action SPEAK -> STAY_SILENT, the reply kind grounded -> withheld_abstain.
The answer goes from "the dog chases the cat" to "I don't know about that."
(`smoke_summary.json`, `changes_vs_intact.gnw_lesion.sw_open`).
The cut held at read time: `cut.ws_lesion` is true in `gnw_lesion.json`.
The row `open-ended-turn-gnw-drive` scores load-bearing at seed 7: 3 decision-field diffs, 0 in the null
(`smoke_summary.json`, `rows.open-ended-turn-gnw-drive`).

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

**The gate lesion (the afferent cut) moves the SETTLE marker on the affective turn, not the BG action.** On `emo` the
marker register goes 2 -> none (`changes_vs_intact.gate_lesion.emo`). The saliences are cut to (0.5, 0.5) on all three
turns (`cut.baseline_applied` true), and the BG action is unchanged on each at seed 7.
The row `open-ended-turn-faculty-drive` probes `unknown`, where it scores not load-bearing at seed 7.
Two of its three fields cannot carry much there. The marker is never read on `unknown`, because a neutral mood
(level 0) is not sent to the circuit, so `marker_level` is 0 in every arm by design. The intact race sits at s = 0.

**Flag-off identity holds on the turns checked.** With the flag unset, the response JSON of 5 turns (`well`, `unknown`,
`sw_open`, `rich_well`, `rich_open`; LTM off) is identical at `3ef5ec08` and at `306ef27d`.
Both hash to sha256 `fc3b716d195e...` (`compare_off5_ltmoff_s7_306ef27d_vs_3ef5ec08.json`, `identical` true).
The same 5-turn check was identical at `928d0d82` vs `f554056e` (`research/findings/raw/_open_ended_gated/identity/compare_off5_ltmoff_s7.json`).
**Scope deviation, disclosed:** the pre-registration's integrity section names the full probe roster.
These 5 turns reach both hooked paths (single-fact and rich), but they are not the roster.

## Against the plan's S16 success check

The check reads: conditioning state changes under the affect lesion and under the GNW lesion, a clean null, and
`byte_identical_off` true.
- GNW lesion: met at the decision level.
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
A seed-7 read of the default-roster candidates `chase`, `question` and `open` under the GNW lesion was launched
for that purpose (`research/findings/raw/_open_ended_gated/smoke_gnwturn/`).

## What this does not show

- Nothing about the gate seeds. Part A (the capability gate) and Part B (the LBF rows) have not run.
- Nothing about rendered Qwen text. The LLM was disabled in every run.
- That the affect afferent drives the reply. At seed 7 it did not.
