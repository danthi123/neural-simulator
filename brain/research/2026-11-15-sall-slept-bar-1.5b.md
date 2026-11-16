# Pre-registration: refit the S_all bar on slept live reads at 1.5B (DEV bank), then a six-seed weekly S_all vs S0

Written 2026-11-15, before any run. Capability: Self and identity, and Memory selection; board #280.

**Record brief.**
- After the night, S0 loses "What's your name?" to the sister telling on 6/6 slept homes
  (`results/2026-11-15-self-recall-after-night-1.5b.md`). A question key was tried in its place and stole every
  question (`results/2026-11-15-self-key-question-1.5b.md`, NOT A LEAD).
- On the same homes, S_all read through the slept cortex ranks the self episode first on 6/6. The 1.5B bar
  (0.22 / 0.01) passes it on only 2/6, and the told controls hold on 5/6.
- That bar was fit on unslept reads of the DEV bank (`brain.py` `KEY_BARS_BY_SIZE`,
  `research/2026-11-03-hippo-routing-1.5b.md`). The night changes S_all scores (P0 6/6), so slept reads are a new
  operating point.
- S_all leads S0 by about one fact on 1.5B told measures at no false-recall cost. It was not made the default
  because a mixed day fell short of its margin, +0.83 against a bar of 1 (`results/2026-11-14-key-default-1.5b.md`).
- The 0.1B live refit procedure exists: `brain/experiments/sall_live_bar_fit.py` (2026-11-03,
  `results/2026-11-03-sall-live-bar-refit.md`). It is extended here, not rebuilt.
- Prerequisites met. Host decision: none.

## Stage 1: fit (AWS L4, DEV bank, seeds 42/43/44)

Command: `python -m brain.experiments.sall_live_bar_fit score --size-sleep --model 1.5b --device cuda --seed <s>`.
- Tell 12 shuffled DEV_FACTS36 statements to a fresh 1.5B brain (all defaults, self_completion on), then restart.
  The restart sleeps once at the 1.5B size defaults.
- Then record the slept S_all scores of every episode for these cues: the told questions (orig, para_a, para_b),
  the untold DEV questions, DEV_NEVER_TOLD, and the three self questions (kind "self", right = the self episode).
- DEV facts are disjoint from the weekly's and the 11-14 probe's facts.
- Fit with `fit`, unchanged: the same grid, objective J = told selection (orig + para_a) − false fire (untold +
  never), ties to the larger bar, then the larger margin. Self selection at the fit is reported, not optimised.
- The pair is frozen in `Brain.KEY_BARS_SLEPT_BY_SIZE[("S_all", "1.5b")]`, used by `key_bar="slept"` (new switch
  value, default OFF), before stage 2 runs.

## Stage 2: weekly (AWS L4, seeds 42/43/44/100/101/102)

`brain.eval.weekly --model 1.5b --shared-sleep --seed <s>`, run twice: as is (S0, the default) and with
`--key S_all --key-bar slept` (new weekly flags). The seeds are the registered ones. They differ from the fit only in
facts, not seed values, since the fit used DEV facts. Scored by `brain/experiments/sall_weekly_score.py`, written
before the runs.

**Criteria (intact arm, six seeds).**
- W1 (routing): session-2 "What's your name?" recalls the own-name self episode (the `_(recalled: ...)_` tag) on
  >= 5/6 seeds.
- W2 (answer): session-2 own name right on >= 5/6 seeds and on >= as many seeds as S0.
- W3 (no cost): session-2 keyed mean >= S0 − 0.5. The day-1 own name and the session-2 "remember my name?" are each
  right on >= S0 − 1 seeds.
- W4 (answer rate): abstentions ("I don't know about that.") over both sessions are not above S0 on any seed.
- The hippocampus-off arm is reported, not a criterion: the key cannot act with the hippocampus off.

**Decision.**
- W1-W4 hold: S_all with the slept bar fixes the name after the night in the weekly. Next is the 11-14 mixed-day test
  (part B) with the slept bar; if its K2 passes there, the key default flips for 1.5B. This weekly alone flips
  nothing.
- W1 holds and W2 fails: routing is fixed and the answer is not. Read the reinstated reply.
- W1 fails: the slept bar does not carry the self episode in a live conversation. Report the scores.
- W3 or W4 fails: S_all costs answers in the weekly; NOT A LEAD at this bar.

**Expected.** The fit lands below 0.22, since slept self scores sit at 0.15-0.24. W1 is then likely. W3 is the risk:
a lower bar fires on more cues.
