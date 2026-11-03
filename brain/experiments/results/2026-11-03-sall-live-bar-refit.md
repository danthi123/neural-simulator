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

---
# RESULTS (0.1B, CPU; numbers from `2026-11-03-sall-live-bar-refit/`)
Instrument unchanged after the pre-registration commit (66881f9ce). Pipeline smoke: fit script at 2 sleep steps, seed 7
(not scored). Fit frozen and committed (8410bf0d7) before any evaluation run.

## Fit (dev facts told live, slept, seeds 42/43/44 pooled; `fit_result.json`, raw `fit_s4{2,3,4}.json`)
**Live-fit bar: FAMILIARITY 0.16, MARGIN 0.05** (offline dev fit was 0.185 / 0.05). Dev J 0.61 (offline bar on the same
cues 0.52; S0's default bar 0.12 / 0.02 applied to S_all 0.43; forced top-1 -0.07).

| S_all on live dev cues | sel orig | sel light | sel heavy (not fit) | fire untold (24) | fire never (12) |
|---|---|---|---|---|---|
| live fit 0.16 / 0.05 | 0.75 | 0.64 | 0.33 | 0.07 | 0.11 |
| offline bar 0.185 / 0.05 | 0.67 | 0.53 | 0.25 | 0.07 | 0.08 |
| forced top-1 | 0.92 | 0.94 | 0.69 | 1.00 | 1.00 |

The live refit lowers the bar by 0.025 (margin unchanged): on live dev cues the offline bar was too strict, by about one
rewording in ten.

## Evaluation (unchanged probe, bar 0.16 / 0.05; each JSON records `bars`)
Held-out seeds 100/101/102 (`table_heldout_100-102.txt`), primary arm a:

| arm | recall /12 | paraphrase /12 | false recall /16 | sel told / para | never-told fired /16 |
|---|---|---|---|---|---|
| a_S0 | 6.33 [6,7,6] | 5.33 [4,6,6] | 12.33 [13,13,11] | 11 / 7 | 14 |
| a_Sall (0.16) | 6.00 [6,6,6] | 5.67 [5,6,6] | 14.00 [15,14,13] | 9 / 6 | 6.67 |

- P1 FAIL: paraphrase a_Sall - a_S0 = +0.33 (needed >= +1.0); a_Sall >= a_S0 on 3/3 seeds.
- P2 FAIL: false recall +1.67 (allowed <= +0.5); a_Sall > a_S0 on 3/3 seeds (allowed <= 1).
- P3 PASS: recall -0.33 (allowed >= -0.5).

Six-seed tally (original criteria verbatim, `table_6seeds.txt`): paraphrase 5.17 vs 5.17 (d +0.00, >= on 5/6) P1 FAIL;
false recall 13.00 vs 12.83 (d +0.17, higher on 4/6) P2 FAIL; recall 5.67 vs 6.33 (d -0.67) P3 FAIL. Secondary b
(hippocampus alone): false recall 5.33 vs 9.33 (lower on 6/6), paraphrase 3.33 vs 3.83, recall 4.83 vs 6.33. Selection
at 0.16: S_all 9 originals / 6 rewordings (was 9 / 5 at 0.185), never-told fire 6.67/16 (was 4.5); S0 11 / 7, 14/16.
Lesion: hippocampus off changes the reply on 121/240 questions vs a_Sall.
Fidelity: the fit script's top-1 S_all score on the slept test homes equals the probe's recorded similarity within
0.0007 (rounding) on all 40 cues of seeds 100/101/102 (`check/`).

## Verdict
Not a lead. Refitting the S_all bar on live cues moves it from 0.185 to 0.16 and buys one correctly selected rewording
(5 -> 6 of 12 on the test facts), but strict paraphrase recall in the slept brain stays level with S0 (held-out +0.33,
six seeds +0.00) and false recall in the reply rises (held-out +1.67). The bar was a small part of the failure.
What the read-outs show:
1. The bar is not where the test rewordings are lost. On the test home (seed 100, `check/check100.json`, read after the
   verdict, not used) S_all at 0.16 selects 9 originals / 6 rewordings; it needs a bar near S0's 0.12 / 0.02 to select
   11 / 9 (S0: 11 / 7). It then fires on 9/16 never-told questions (S0 14/16). That bar is below what live dev cues
   support (dev J 0.43 there), so the dev-to-test gap is in the test facts' scores (for example "What is my cat called?" scores
   0.138 against its own telling), not in the conversation context.
2. As in the 0.185 run, hippocampal selection does not set false recall after sleep: the hippocampus-off brain says a
   fact word on 11.7/16 never-told questions, and both keys sit at 12.8-13.0 over six seeds. More S_all firing
   (6.7 vs 4.5 never-told) moved it up slightly. Without sleep (b) S_all again cuts false recall 9.3 -> 5.3.
Caveats: one sampled reply per question; 12 test facts; 3 held-out seeds for the primary test. a_S0 in this rerun is
not byte-identical to the 11-03 run on every question (seed 100 paraphrase 4 vs 5; sleep float order under different
thread counts and CPU load). The arms are paired within each run, so the comparison is unaffected.
Next step: keep the switch OFF and stop refitting the S_all bar. Two levers remain: (a) a better WHETHER signal than the
top-1 score (told-vs-untold AUROC 0.62-0.65 offline), and (b) the false-recall gate on the consolidated path (sleep's
synapses), which drives reply-level false recall at 0.1B whichever key is used.
