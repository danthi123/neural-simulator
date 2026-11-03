# 2026-11-03 S_all bar refit on live-brain cues -- PRE-REGISTRATION (written before any run)

Branch `research/sall-live-bar-refit` (from main @ 7cb3223e2). Capability: memory selection (`brain/PREREQUISITES.md`).
Record: `2026-11-03-sall-in-brain.md` (S_all at the offline dev bar 0.185 / 0.05 in the live brain: P1-P3 FAIL; S_all
fired on 5/12 rewordings vs S0 10/12; next step there: refit the bar on live cues, rerun the probe unchanged) and
`2026-11-03-meaning-selection-realpath.md` (the offline fit: objective, grid, dev bank). `tools/before_you_build.sh`
reported a stale local corpus (published main not in local main), so the record search was not run; the two writeups
above are the record for this question.

## What the saved probe data can and cannot do
The saved `2026-11-03-sall-in-brain/seed*.json` keep, per question, only the top-1 weighted score and the recalled
episode, not the runner-up, so the margin rule cannot be replayed from them; and their questions are the TEST facts, so
fitting on them would fit the bar on the evaluation set. They are used only for a descriptive read-out (top-1 S_all score
by question kind, test cues, fire rate vs bar with margin ignored), not for the fit.

## Fit (dev facts told in the live brain; seeds 42/43/44 only) -- `brain/experiments/sall_live_bar_fit.py`
Per fit seed: `random.Random(seed)` shuffles `DEV_FACTS36` (no word or cue shared with the test bank, asserted in
`facts_bank.py`); the first 12 statements are told in conversation exactly as the live probe tells its facts
(`live_brain_probe.tell_day`: default key, 0.1B, CPU, sleep defaults 240 steps lr 5e-4 kl 1.0 rank 8, seed = fit seed);
the restart sleeps once (the probe's primary arm a is the slept home). A fresh `Brain(home, key="S_all",
sleep_on_restart=False)` then gives `hippocampus.scores(...)` (strength-weighted S_all similarities against every stored
episode, including the brain's 3 permanent self episodes) for every cue, with no recall side effects.
Told cues: the 12 told facts' same-words question and light rewording (the two cue kinds the live probe asks); the heavy
rewording is scored and reported, not fitted. Untold cues: the 24 untold dev facts' same-words questions + the 12
`DEV_NEVER_TOLD` (36). Right episode = the episode whose text is `User: <statement>`.
Rule (= `Hippocampus._compete`, question cues): fire iff top >= FAMILIARITY and top - runner-up >= MARGIN.
Grid (as the offline fit): FAMILIARITY -0.10..0.80 step 0.005 x MARGIN {0, .005, .01, .02, .03, .05, .08, .12}.
Objective (as the offline fit, restricted to the live probe's cue kinds): J = P(fire and right episode | told cue,
orig + light pooled) - P(fire | untold cue), pooled over the 3 fit seeds; ties to the larger bar, then the larger margin.
The fitted pair is frozen before any evaluation run. If it equals 0.185 / 0.05 no rerun is made (the verdict is then that
the live refit returns the offline bar).

## Evaluation (unchanged probe, held-out seeds 100/101/102)
`brain/experiments/sall_live_probe.py` and `sall_live_table.py` unchanged; the fitted pair is set as
`Brain.KEY_BARS["S_all"]` for the run only (wrapper `sall_live_bar_eval.py`; each JSON records `bars`). Same arms,
questions, scorer, seeds per run. Primary: seeds 100/101/102 (never used in the fit). Also run 42/43/44 on the TEST facts
(their fit used only dev facts, so no test content leaked) for the six-seed tally the six-seed rule needs; reported.

## Criteria (from the original pre-registration, primary arm a, on 100/101/102)
- P1 (paraphrase): strict paraphrase hits a_Sall - a_S0 mean >= +1.0 /12, and a_Sall >= a_S0 on 3/3 seeds
  (the original's 5/6 allows one loss in six; with three seeds no loss is allowed).
- P2 (no false-recall rise): mean false recall a_Sall <= mean a_S0 + 0.5 (/16), and a_Sall > a_S0 on <= 1/3 seeds.
- P3 (originals not lost): mean strict recall a_Sall >= mean a_S0 - 0.5.
Six-seed tally (reported): the original criteria verbatim on all six seeds (`sall_live_table.py`).
Verdict: P1-P3 hold on 100/101/102 AND the six-seed tally passes = a lead for S_all at the live bar (not a GO, not a
default flip). P1-P3 hold on 100/101/102 only = held-out pass, not a lead. Otherwise each failing criterion is reported.
Reported, not criteria: b arms, selection / never-told fire per arm, correct_answer, lesion c; fidelity check: the fit
script's top-1 score on the test homes equals the probe's recorded `similarity` (same rule, same code path).
