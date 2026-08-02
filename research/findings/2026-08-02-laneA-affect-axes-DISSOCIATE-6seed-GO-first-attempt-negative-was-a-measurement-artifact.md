---
type: finding
status: contributing
date: 2026-08-02
mechanism: neuromodulator-affect-axes
artifacts:
  - research/findings/raw/lanes/affect/affect_warmup1500_s42.json
  - research/findings/raw/lanes/affect/affect_warmup1500_s43.json
  - research/findings/raw/lanes/affect/affect_warmup1500_s44.json
  - research/findings/raw/lanes/affect/affect_warmup1500_s100.json
  - research/findings/raw/lanes/affect/affect_warmup1500_s101.json
  - research/findings/raw/lanes/affect/affect_warmup1500_s102.json
---

# lane A (affect axes): the three neuromodulator affect axes DO dissociate — 6-seed GO — once the block is measured AFTER the reward-expectation converges; the first-attempt 6-seed NEGATIVE was a MEASUREMENT ARTIFACT (warmup shorter than the surprise-expectation window), NOT opponent-coupling (the board's guess) and NOT a wrong 500ms biology (the window-shortening an agent proposed would have tuned the biology to fit the metric)

<!--derived-->
**One-line verdict.** The three affect axes (mood/5-HT, arousal/NA, eagerness/ACh) each respond MAXIMALLY to their own
matched driver and form the unique-perfect axis-driver permutation — G1 (own-is-max 3/3), G2 (unique permutation), G3
(lesion collapse), G4 (mood lags DA) ALL True on 6/6 seeds — with the arousal surprise-rule's reward-expectation window
LEFT AT ITS 500ms biology. The first-attempt 6-seed NEGATIVE (`affect_axes_6seed.json`, own-is-max 2/3, arousal reading
`sustained` as more arousing than `surprise`) was a MEASUREMENT ARTIFACT: the block was scored after a 200-step warmup,
shorter than the ~3x500ms the surprise reward-expectation EMA needs to converge, so a fully-EXPECTED constant stream
still read as surprising (a decaying transient). Measuring after warmup >= 1500 (3 tau) removes it. No `sim/` biology
edit — a runner measurement-protocol fix (`--warmup` default 200 -> 1500).

## Result — 6 seeds, warmup=1500 (biology window_ms=500 UNCHANGED)

<!--derived-->
| seed | own_is_max | G1 dissociation | G2 unique-perm | G3 lesion collapse | G4 mood-lag | GO |
|---|---|---|---|---|---|---|
| 42  | 3/3 | T | T | T | T | **True** |
| 43  | 3/3 | T | T | T | T | **True** |
| 44  | 3/3 | T | T | T | T | **True** |
| 100 | 3/3 | T | T | T | T | **True** |
| 101 | 3/3 | T | T | T | T | **True** |
| 102 | 3/3 | T | T | T | T | **True** |

<!--derived-->
6/6 GO. The lever is a decaying transient, not a coupling: at warmup 200 the arousal row is `sustained 1.52 >
surprise 1.25` (fails), by warmup 1500 it is `sustained 0.19 < surprise 1.29` (passes), and by warmup 2500
`sustained 0.015` — the spurious sustained-arousal response DECAYS as the expectation EMA converges, exactly the
signature of an unconverged statistic. mood and eagerness rows were already diagonal at every warmup. Artifact
`research/findings/raw/lanes/affect/affect_warmup1500_s42.json`; command
`SIM_BACKEND=numpy .venv/bin/python -m research.runners._neuromodulator_affect_axes_derisk --seeds <s> --warmup 1500 --n-steps 4000`.

## Why this is the HONEST fix (the discipline that produced it)

<!--derived-->
Two other "fixes" were on the table and both were wrong for this defect. (1) The board's residual said "finer OPPONENT
tuning" — but this probe has NO opponent / cross-coupling between axes (the four axes are independent
`NeuromodulatorManager` configs); there was nothing to tune, so that residual was a mischaracterization. (2) An agent's
scout found that SHORTENING the surprise window (`window_ms` 500 -> 100) also makes own-is-max pass — but that works by
changing the LC/NA reward-expectation ADAPTATION TIMESCALE to fit the 200-step measurement window, i.e. tuning the
biology to the metric. Testing the companion process directly (does LENGTHENING the warmup, biology untouched, fix it?)
showed it does — so the real defect was the INSTRUMENT (scoring before convergence), not the mechanism or its timescale.
This is the project's recurring lesson ("what else does the real system run alongside this that we replaced with a
constant / measured too early?") applied before tuning: the surprise expectation is a companion process with its own
convergence time, and the measurement must respect it.

## Honest scope

<!--derived-->
This validates that the affect axes DISSOCIATE as a functional-correlate readout (each neuromodulator axis is selectively
driven by its own driver, lesion-dependent, with mood's slow timescale) — it does NOT assert felt affect; the axes are a
measured functional decomposition. The prior `affect_axes_6seed.json` NEGATIVE is superseded (its warmup under-measured
the arousal axis). The fix is a measurement-protocol default; the 500ms surprise-expectation biology is unchanged.
