# Amendment A0.1 to Step 4b: engagement bar (a) is read at the reward time (Claude, 2026-10-06)

Amends `docs/drafts/long-delay-credit/STEP4B.md` (Claude, 2026-10-04), section 2 "Engagement check" bar (a) and the
two section-4 rows that read it. Answers NEEDS `[T1]` ASK 2026-10-05 05:12. Written before any Step 4b constant is
frozen and before any seed 8-15 line exists, so no grid result has been seen. The STEP4B text is left as written; this
file is the dated change.

## What the bar said, and why it could never pass

STEP4B bar (a): `lhb_rate_unrewarded > 2 x lhb_rate_rewarded`, both read as the LHb mean rate over the whole outcome
window (`_ldc_omission_pilot.py`: LHb spikes in OUTCOME_MS / (n_lhb x OUTCOME_MS)).

Inside the outcome window the two trial types differ only while the reward is delivered: reward_us is on for
REWARD_MS = 50 ms, from REWARD_ON_MS. Before and after that, both trial types receive the same drive (the outcome
screen ctx_o). So even if the reward silences the LHb completely, the whole-window ratio cannot exceed
OUTCOME_MS / (OUTCOME_MS - REWARD_MS):

- 250 / 200 = 1.25 at the Step 4a timing the tune uses (onset 150, outcome 250). Measured: 1.02-1.04 with the
  starting release weights (round 1), 1.22-1.23 with the release arm tripled (rounds 2-3), so the measurement sits
  right under the cap.
- 100 / 50 = 2.0 at the registered default timing (onset 30, outcome 100). The bar is a strict "> 2", so it was
  unreachable there too.

So the bar as written does not test the hypothesis. This is an error in the instrument (Claude's, 2026-10-04), not a
failed prediction. The hypothesis itself is unchanged: "the LHb fires in the outcome window unless the reward
arrives" (STEP4B section 2).

## Biology: when the LHb answers a brief reward

In primates the lateral habenula is phasically inhibited by reward delivery and phasically excited when an expected
reward is omitted. Both responses are time-locked to the time of the outcome and last on the order of 100-200 ms;
they are not a change in the mean rate over the whole post-choice period (Matsumoto and Hikosaka 2007, Nature,
PMID 17522629). The LHb is needed for the dopamine-neuron dip at reward omission (Tian and Uchida 2015, Neuron,
PMID 26365765). The right comparison is therefore the LHb rate AT THE REWARD TIME: the reward period on rewarded
trials, against the same moment (the expected reward time) on unrewarded trials. A 50 ms reward pulse in this model
stands for a brief reward; a whole-window mean dilutes a response the biology expects to be brief.

## The amended bar (a), fixed now

- Window: steps k with REWARD_ON_MS <= k < REWARD_ON_MS + REWARD_MS of the outcome period, the same steps on every
  trial. Read on TRAIN trials, as before (reward_us is driven only while learning).
- Bar (a): `lhb_rate_rewardwin_unrewarded > 2 x lhb_rate_rewardwin_rewarded`. The factor 2 is unchanged.
  `lhb_engagement_a_rewardwin.pass` in each record computes it. `pass: null` (the LHb is silent at the expected
  reward time on unrewarded trials, or a trial type is missing) is UNDEFINED, never a pass.
- Bar (b) is unchanged: route-on unrewarded DA min below route-off by at least 0.02.
- The whole-window rates `lhb_rate_rewarded` / `lhb_rate_unrewarded` stay in every record as diagnostics; no bar reads
  them.
- STEP4B section 4 rows that name `lhb_rate_unrewarded > 2 x lhb_rate_rewarded` ("the route engages and the readout
  does not follow") and "the LHb does not separate rewarded from unrewarded" read the reward-time rates, i.e.
  `lhb_engagement_a_rewardwin.pass`.

Implemented in `research/runners/_ldc_omission_pilot.py` (`lhb_spikes_rewardwin` per trial, `lhb_window_rates`,
`engagement_a`); pinned by `tests/test_ldc_omission_rewardwin.py`, which also shows the whole-window cap of 1.25 and
that a weak reward-time suppression (about 17%, the round-1 release arm) still FAILS the amended bar. The change
counts spikes only: no circuit constant, drive or timing changed, and every other record field is computed as before.

## What was considered and refused

- Lowering the factor, or reading the whole window with a smaller factor: a bar moved to fit a measurement.
- A slower LHb release (GABA_B on the LHb) so the suppression outlasts the pulse: that changes the circuit to fit an
  instrument. It may become a method later if the DA bars ask for it; it is not a fix for bar (a).

## What this does not certify (read with the grid)

On every trial the outcome screen drives the LHb from outcome onset, so in this build the LHb also fires in the
0-150 ms before the reward time on rewarded trials, and the RMTg relay pauses the SNc there too. In vivo the omission
excitation is timed to the expected reward. The amended bar shows reward-specific suppression at the reward time; it
does not show that the excitation is timed. Outcome-specificity at the DA level stays with the R bar and the `cut` arm
(STEP4B section 4). The LHb rate (about 63 Hz on unrewarded trials in rounds 1-3) is also above typical in-vivo LHb
rates; it is recorded, not barred, in this step.

## Prediction (written before the re-run; a check, not a result)

From the whole-window rates, if the rewarded-trial deficit falls inside the reward window, r5 (30/27/24/20/3) loses
about 3.0 of about 3.2 spikes per neuron there, so its reward-time ratio is far above 2; round 1 (release 9/12) loses
about 0.55, a ratio near 1.2, which fails. The re-run measures this; the estimate is not used for any decision.

## Check after the text above was fixed (2026-10-06; dev seed 7 smokes, not evidence)

The window and the bar above were written before any of these runs. Seed 7, credit_immediate, onset 150 / outcome
250, clear 60, w_outc_strio 30, route 30/x/y/20/3, n-train 6-12 (smokes only; every reading needs n-train 200):

| w_rus_lhbinh / w_lhbinh_lhb | whole-window U/R (Hz) | reward-time U/R (Hz) | bar (a) |
|---|---|---|---|
| 27 / 24 (r5) | 63.4 / 51.4 (r5 exactly) | 66 / 39, ratio 1.69 | FAIL |
| 27 / 48 | 63.4 / 47.8 | 66 / 35, ratio 1.89 | FAIL |
| 54 / 48 | 63.4 / 45.2 | 66 / 27, ratio 2.44 | pass |
| 54 / 96 | 63.4 / 42.8 | 66 / 27, ratio 2.44 | pass |

The prediction above was wrong: at r5 only about 1.35 of the 3.0 missing spikes per neuron fall inside the reward
window. The rest fall after it, because the reward-driven inhibition (reward_us -> lhb_inh -> lhb) starts about 20 ms
into the pulse and outlasts it. So r5 does NOT pass the amended bar, and the amended bar separates release
strengths. With the release arm doubled the rewarded rate floors at 27 Hz, which is that onset latency; the
whole-window ratio stays under 1.25 in every row, as the cap says. The DA bars did not move in these smokes
(rewarded max about 0.613, unrewarded min about 0.460), since the release arm acts only on rewarded trials, but
12-trial values are not readings.
